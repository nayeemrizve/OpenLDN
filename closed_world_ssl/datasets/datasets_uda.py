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

    # load the saved labeled/unlabeled split
    label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
    train_labeled_idxs = label_unlabel_dict['labeled_idx']
    train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
    train_unlabeled_idxs_ = train_unlabeled_idxs # used to generate iterative pseudo-labels

    # load pseudo-labels
    pl_dict = pickle.load(open(args.pl_dict, 'rb'))
    pseudo_idxs = pl_dict['pseudo_idx']
    pseudo_targets = pl_dict['pseudo_target']

    # incorporate pseudo-labels into the labeled set
    train_unlabeled_idxs = [item for item in train_unlabeled_idxs if item not in pseudo_idxs]
    train_labeled_idxs.extend(pseudo_idxs)

    # generate datasets
    train_labeled_dataset = CIFAR10SSL(args.data_root, train_labeled_idxs, pseudo_idxs, pseudo_targets, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS32(mean=cifar10_mean, std=cifar10_std))
    train_pl_dataset = CIFAR10SSL(args.data_root, train_unlabeled_idxs_, train=True, transform=transform_val)
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

    # load the saved labeled/unlabeled split
    label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
    train_labeled_idxs = label_unlabel_dict['labeled_idx']
    train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
    train_unlabeled_idxs_ = train_unlabeled_idxs # used to generate iterative pseudo-labels

    # load pseudo-labels
    pl_dict = pickle.load(open(args.pl_dict, 'rb'))
    pseudo_idxs = pl_dict['pseudo_idx']
    pseudo_targets = pl_dict['pseudo_target']

    # incorporate pseudo-labels into the labeled set
    train_unlabeled_idxs = [item for item in train_unlabeled_idxs if item not in pseudo_idxs]
    train_labeled_idxs.extend(pseudo_idxs)

    # generate datasets
    train_labeled_dataset = CIFAR100SSL(args.data_root, train_labeled_idxs, pseudo_idxs, pseudo_targets, train=True, transform=transform_labeled)
    train_unlabeled_dataset = CIFAR100SSL(args.data_root, train_unlabeled_idxs, train=True, transform=TransformWS32(mean=cifar100_mean, std=cifar100_std))
    train_pl_dataset = CIFAR100SSL(args.data_root, train_unlabeled_idxs_, train=True, transform=transform_val)
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

    # load the saved labeled/unlabeled split
    label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
    train_labeled_idxs = label_unlabel_dict['labeled_idx']
    train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
    train_unlabeled_idxs_ = train_unlabeled_idxs # used to generate iterative pseudo-labels

    # load pseudo-labels
    pl_dict = pickle.load(open(args.pl_dict, 'rb'))
    pseudo_idxs = pl_dict['pseudo_idx']
    pseudo_targets = pl_dict['pseudo_target']

    # incorporate pseudo-labels into the labeled set
    train_unlabeled_idxs = [item for item in train_unlabeled_idxs if item not in pseudo_idxs]
    train_labeled_idxs.extend(pseudo_idxs)

    # generate datasets
    train_labeled_dataset = SVHNSSL(args.data_root, train_labeled_idxs, pseudo_idxs, pseudo_targets, split='train', transform=transform_labeled)
    train_unlabeled_dataset = SVHNSSL(args.data_root, train_unlabeled_idxs, split='train', transform=TransformWS32(mean=normal_mean, std=normal_std))
    train_pl_dataset = SVHNSSL(args.data_root, train_unlabeled_idxs_, split='train', transform=transform_val)
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

    # load the saved labeled/unlabeled split
    label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
    train_labeled_idxs = label_unlabel_dict['labeled_idx']
    train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
    train_unlabeled_idxs_ = train_unlabeled_idxs # used to generate iterative pseudo-labels

    # load pseudo-labels
    pl_dict = pickle.load(open(args.pl_dict, 'rb'))
    pseudo_idxs = pl_dict['pseudo_idx']
    pseudo_targets = pl_dict['pseudo_target']

    # incorporate pseudo-labels into the labeled set
    train_unlabeled_idxs = [item for item in train_unlabeled_idxs if item not in pseudo_idxs]
    train_labeled_idxs.extend(pseudo_idxs)

    # generate datasets
    train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, pseudo_idxs, pseudo_targets, transform=transform_labeled)
    train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformWS64(mean=tinyimagenet_mean, std=tinyimagenet_std))
    train_pl_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs_, transform=transform_val)
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

    # load the saved labeled/unlabeled split
    label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
    train_labeled_idxs = label_unlabel_dict['labeled_idx']
    train_unlabeled_idxs = label_unlabel_dict['unlabeled_idx']
    train_unlabeled_idxs_ = train_unlabeled_idxs # used to generate iterative pseudo-labels

    # load pseudo-labels
    pl_dict = pickle.load(open(args.pl_dict, 'rb'))
    pseudo_idxs = pl_dict['pseudo_idx']
    pseudo_targets = pl_dict['pseudo_target']

    # incorporate pseudo-labels into the labeled set
    train_unlabeled_idxs = [item for item in train_unlabeled_idxs if item not in pseudo_idxs]
    train_labeled_idxs.extend(pseudo_idxs)

    # generate datasets
    train_labeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_labeled_idxs, pseudo_idxs, pseudo_targets, transform=transform_labeled)
    train_unlabeled_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs, transform=TransformWS224(mean=imgnet_mean, std=imgnet_std))
    train_pl_dataset = GenericSSL(os.path.join(args.data_root, 'train'), train_unlabeled_idxs_, transform=transform_val)
    test_dataset_known = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(0, args.no_known)))
    test_dataset_novel = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val, labeled_set=list(range(args.no_known, args.no_class)))
    test_dataset_all = GenericTEST(os.path.join(args.data_root, 'test'), no_class=args.no_class, transform=transform_val)

    return train_labeled_dataset, train_unlabeled_dataset, train_pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all


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


class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, pseudo_idxs=None, pseudo_targets=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        # replace ground-truth with pseudo-labels
        if pseudo_idxs is not None:
            self.targets[pseudo_idxs] = pseudo_targets

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
    def __init__(self, root, indexs, pseudo_idxs=None, pseudo_targets=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.targets = np.array(self.targets)

        # replace ground-truth with pseudo-labels
        if pseudo_idxs is not None:
            self.targets[pseudo_idxs] = pseudo_targets

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
    def __init__(self, root, indexs, pseudo_idxs=None, pseudo_targets=None, split='train',
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, split=split,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)

        self.labels = np.array(self.labels)

        # replace ground-truth with pseudo-labels
        if pseudo_idxs is not None:
            self.labels[pseudo_idxs] = pseudo_targets

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
    def __init__(self, root, indexs, pseudo_idxs=None, pseudo_targets=None,
                 transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.imgs = np.array(self.imgs)
        self.targets = self.imgs[:, 1]
        self.targets= list(map(int, self.targets.tolist()))
        self.data = np.array(self.imgs[:, 0])
        self.targets = np.array(self.targets)

        # replace ground-truth with pseudo-labels
        if pseudo_idxs is not None:
            self.targets[pseudo_idxs] = pseudo_targets

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
