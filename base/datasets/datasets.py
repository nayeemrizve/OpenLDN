import numpy as np
from PIL import Image, ImageFilter, ImageOps
import random
from torchvision import datasets, transforms
import torch
import pickle
import os
from .randaugment import RandAugmentMC
import math


# normalization parameters
cifar10_mean, cifar10_std = (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
cifar100_mean, cifar100_std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
normal_mean, normal_std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
tinyimagenet_mean, tinyimagenet_std = (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262)
imgnet_mean, imgnet_std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def get_dataset(args):
    if args.dataset == 'cifar10':
        return get_cifar10(args)
    elif args.dataset == 'cifar100':
        return get_cifar100(args)
    elif args.dataset == 'svhn':
        return get_svhn(args)
    elif args.dataset == 'tinyimagenet':
        return get_tinyimagenet(args)
    elif args.dataset in ['aircraft', 'stanfordcars', 'oxfordpets', 'imagenet100', 'herbarium']:
        return get_dataset224(args)


def get_cifar10(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        base_dataset = datasets.CIFAR10(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'cifar10_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = CIFAR10SSL(args.data_root, train_labeled_idxs, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS32(mean=cifar10_mean, std=cifar10_std))
    train_pl_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val)
    test_dataset_known = CIFAR10SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(0,args.no_known)))
    test_dataset_novel = CIFAR10SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = CIFAR10SSL_TEST(args.data_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, train_pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


def get_cifar100(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        base_dataset = datasets.CIFAR100(args.data_root, train=True, download=True)
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset.targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'cifar100_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = CIFAR100SSL(args.data_root, train_labeled_idxs, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS32(mean=cifar100_mean, std=cifar100_std))
    train_pl_dataset = CIFAR100SSL(args.data_root, train_unlabeled_idxs, train=True, transform=transform_val)
    test_dataset_known = CIFAR100SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(0,args.no_known)))
    test_dataset_novel = CIFAR100SSL_TEST(args.data_root, train=False, transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = CIFAR100SSL_TEST(args.data_root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, train_pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


def get_svhn(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=normal_mean, std=normal_std)
    ])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        base_dataset = datasets.SVHN(args.data_root, split='train', download=True)
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset.labels, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'svhn_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = SVHNSSL(args.data_root, train_labeled_idxs, split='train', transform=transform_labeled)
    train_unlabeled_dataset = SVHNSSL(args.data_root, train_unlabeled_idxs, split='train', transform=TransformWS32(mean=normal_mean, std=normal_std))
    train_pl_dataset = SVHNSSL(args.data_root, train_unlabeled_idxs, split='train', transform=transform_val)
    test_dataset_known = SVHNSSL_TEST(args.data_root, split='test', transform=transform_val, download=True, labeled_set=list(range(0,args.no_known)))
    test_dataset_novel = SVHNSSL_TEST(args.data_root, split='test', transform=transform_val, download=False, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = SVHNSSL_TEST(args.data_root, split='test', transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, train_pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


def get_tinyimagenet(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=tinyimagenet_mean, std=tinyimagenet_std)])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if not os.path.exists(args.ssl_indexes):
        base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
        base_dataset_targets = np.array(base_dataset.imgs)
        base_dataset_targets = base_dataset_targets[:,1]
        base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'tinyimagenet_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()

    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, transform=transform_labeled)
    train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformWS64(mean=tinyimagenet_mean, std=tinyimagenet_std))
    train_pl_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=transform_val)
    test_dataset_known = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(0,args.no_known)))
    test_dataset_novel = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, train_pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


def get_dataset224(args):
    # augmentations
    transform_labeled = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=imgnet_mean, std=imgnet_std)])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imgnet_mean, std=imgnet_std)])

    # generate random labeled/unlabeled split or use a saved labeled/unlabeled split
    if args.ssl_indexes is None:
        base_dataset = datasets.ImageFolder(os.path.join(args.data_root, 'train'))
        base_dataset_targets = np.array(base_dataset.imgs)
        base_dataset_targets = base_dataset_targets[:,1]
        base_dataset_targets= list(map(int, base_dataset_targets.tolist()))
        train_labeled_idxs, train_unlabeled_idxs, train_val_idxs = x_u_split_known_novel(base_dataset_targets, args.lbl_percent, args.no_class, list(range(0,args.no_known)), list(range(args.no_known, args.no_class)))

        f = open(os.path.join(args.split_root, f'{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl'),"wb")
        label_unlabel_dict = {'labeled_idx': train_labeled_idxs, 'unlabeled_idx': train_unlabeled_idxs, 'val_idx': train_val_idxs}
        pickle.dump(label_unlabel_dict,f)
        f.close()
    else:
        label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
        train_labeled_idxs = label_unlabel_dict['labeled_idx']
        train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']

    # balance the labeled and unlabeled data
    if len(train_unlabeled_idxs) > len(train_labeled_idxs):
        exapand_labeled = len(train_unlabeled_idxs) // len(train_labeled_idxs)
        train_labeled_idxs = np.hstack([train_labeled_idxs for _ in range(exapand_labeled)])

        if len(train_labeled_idxs) < len(train_unlabeled_idxs):
            diff = len(train_unlabeled_idxs) - len(train_labeled_idxs)
            train_labeled_idxs = np.hstack((train_labeled_idxs, np.random.choice(train_labeled_idxs, diff)))
        else:
            assert len(train_labeled_idxs) == len(train_unlabeled_idxs)

    # generate datasets
    train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, transform=transform_labeled)
    train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformWS224(mean=imgnet_mean, std=imgnet_std))
    train_pl_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=transform_val)
    test_dataset_known = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(0, args.no_known)))
    test_dataset_novel = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, train_pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


def x_u_split_known_novel(labels, lbl_percent, no_classes, lbl_set, unlbl_set, val_percent=1):
    labels = np.array(labels)
    labeled_idx = []
    unlabeled_idx = []
    val_idx = []
    for i in range(no_classes):
        idx = np.where(labels == i)[0]
        n_lbl_sample = math.ceil(len(idx)*(lbl_percent/100))
        n_val_sample = max(int(len(idx)*(val_percent/100)), 1)
        np.random.shuffle(idx)
        if i in lbl_set:
            labeled_idx.extend(idx[:n_lbl_sample])
            unlabeled_idx.extend(idx[n_lbl_sample:-n_val_sample])
            val_idx.extend(idx[-n_val_sample:])
        elif i in unlbl_set:
            unlabeled_idx.extend(idx[:-n_val_sample])
            val_idx.extend(idx[-n_val_sample:])
    return labeled_idx, unlabeled_idx, val_idx


class TransformWS32(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformWS64(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect')
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=64,
                                  padding=int(64*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class TransformWS224(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            ])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index]


class CIFAR10SSL_TEST(datasets.CIFAR10):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index]


class CIFAR100SSL_TEST(datasets.CIFAR100):
    def __init__(self, root, train=False,
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(100):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class SVHNSSL(datasets.SVHN):
    def __init__(self, root, indexs, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.labels = np.array(self.labels)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.labels))

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.moveaxis(img, 0, -1))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index]


class SVHNSSL_TEST(datasets.SVHN):
    def __init__(self, root, split='test',
                 transform=None, target_transform=None,
                 download=False, labeled_set=None):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.labels = np.array(self.labels)
        indexs = []
        if labeled_set is not None:
            for i in range(10):
                idx = np.where(self.labels == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.labels = np.array(self.labels)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.moveaxis(img, 0, -1))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class GenericSSL(datasets.ImageFolder):
    def __init__(self, root, indexs,
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        if indexs is not None:
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.indexs[index]


class GenericTEST(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None, labeled_set=None, no_class=200):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])

        self.targets = np.array(self.targets)
        indexs = []
        if labeled_set is not None:
            for i in range(no_class):
                idx = np.where(self.targets == i)[0]
                if i in labeled_set:
                    indexs.extend(idx)
            indexs = np.array(indexs)
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = self.loader(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target
