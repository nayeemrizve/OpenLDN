import argparse
import os
import random
import time
import pickle
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm
from models.build_model import build_model
from utils.utils import AverageMeter, accuracy, set_seed, save_checkpoint, sim_matrix, interleave, de_interleave
from datasets.datasets import get_dataset
from utils.evaluate_utils import hungarian_evaluate
from losses.losses import entropy, symmetric_mse_loss
from utils.pseudo_labeling_utils import pseudo_labeling


def main():
    parser = argparse.ArgumentParser(description='Base Training')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium'], help='dataset name')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')
    parser.add_argument('--pl-percent', type=int, default=10, help='percent of selected pseudo-labels data')
    parser.add_argument('--arch', default='resnet18', type=str, help='model architecture')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run, deafult 50')
    parser.add_argument('--batch-size', default=200, type=int, help='train batchsize')
    parser.add_argument('--lr', default=5e-4, type=float, help='learning rate, default 1e-3')
    parser.add_argument('--lr-simnet', default=1e-4, type=float, help='earning rate for simnet, default 1e-4')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', type=int, default=-1, help="random seed (-1: don't use random seed)")
    parser.add_argument('--mu', default=1, type=int, help='coefficient of unlabeled batch size')
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--no-class', default=10, type=int, help='total classes')
    parser.add_argument('--threshold', default=0.5, type=float, help='pseudo-label threshold, default 0.50')
    parser.add_argument('--split-id', default='split_0', type=str, help='random data split number')
    parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')

    args = parser.parse_args()
    best_acc = 0
    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))

    writer = SummaryWriter(logdir=args.out)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write('************************************************************************\n\n')
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
        ofile.write('\n\n************************************************************************\n')
    
    args.n_gpu = torch.cuda.device_count()
    args.dtype = torch.float32
    if args.seed != -1:
        set_seed(args)

    # set dataset specific parameters
    if args.dataset == 'cifar10':
        args.no_class = 10
    elif args.dataset == 'cifar100':
        args.no_class = 100
    elif args.dataset == 'svhn':
        args.no_class = 10
    elif args.dataset == 'tinyimagenet':
        args.no_class = 200
    elif args.dataset == 'aircraft':
        args.no_class = 100
    elif args.dataset == 'stanfordcars':
        args.no_class = 196
    elif args.dataset == 'oxfordpets':
        args.no_class = 37
    elif args.dataset == 'imagenet100':
        args.no_class = 100
    elif args.dataset == 'herbarium':
        args.no_class = 682

    args.data_root = os.path.join(args.data_root, args.dataset)
    os.makedirs(args.data_root, exist_ok=True)
    os.makedirs(args.split_root, exist_ok=True)

    # load dataset
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)
    lbl_dataset, unlbl_dataset, pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all = get_dataset(args)

    # create dataloaders
    unlbl_batchsize = int((float(args.batch_size) * len(unlbl_dataset))/(len(lbl_dataset) + len(unlbl_dataset)))
    lbl_batchsize = args.batch_size - unlbl_batchsize
    args.iteration = (len(lbl_dataset) + len(unlbl_dataset)) // args.batch_size

    train_sampler = RandomSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=lbl_batchsize, num_workers=args.num_workers, drop_last=True)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=unlbl_batchsize, num_workers=args.num_workers, drop_last=True)
    pl_loader = DataLoader(pl_dataset, sampler=SequentialSampler(pl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_known = DataLoader(test_dataset_known, sampler=SequentialSampler(test_dataset_known), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_novel = DataLoader(test_dataset_novel, sampler=SequentialSampler(test_dataset_novel), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all = DataLoader(test_dataset_all, sampler=SequentialSampler(test_dataset_all), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    # create model
    model, simnet = build_model(args)
    model = model.cuda()
    simnet = simnet.cuda()

    # optimizer
    if torch.cuda.device_count() > 1:
        optimizer = torch.optim.Adam(model.module.params(), lr=args.lr)
        optimizer_simnet = torch.optim.Adam(simnet.module.params(), lr=args.lr_simnet)
    else:
        optimizer = torch.optim.Adam(model.params(), lr=args.lr)
        optimizer_simnet = torch.optim.Adam(simnet.params(), lr=args.lr_simnet)

    start_epoch = 0
    if args.resume:
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        simnet.load_state_dict(checkpoint['state_dict_simnet'])
        optimizer_simnet.load_state_dict(checkpoint['optimizer_simnet'])

    test_accs = []
    model.zero_grad()
    for epoch in range(start_epoch, args.epochs):
        train_loss = train(args, lbl_loader, unlbl_loader, model, optimizer, simnet, optimizer_simnet, epoch)

        test_acc_known = test_known(args, test_loader_known, model, epoch)
        novel_cluster_results = test_cluster(args, test_loader_novel, model, epoch, offset=args.no_known)
        all_cluster_results = test_cluster(args, test_loader_all, model, epoch)
        test_acc = all_cluster_results["acc"]

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(f'epoch: {epoch}, acc-known: {test_acc_known}')
        print(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}')
        print(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}')

        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('test/1.acc_known', test_acc_known, epoch)
        writer.add_scalar('test/2.acc_novel', novel_cluster_results['acc'], epoch)
        writer.add_scalar('test/3.nmi_novel', novel_cluster_results['nmi'], epoch)
        writer.add_scalar('test/4.acc_all', all_cluster_results['acc'], epoch)
        writer.add_scalar('test/5.nmi_all', all_cluster_results['nmi'], epoch)

        model_to_save = model.module if hasattr(model, "module") else model
        simnet_to_save = simnet.module if hasattr(simnet, "module") else simnet
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model_to_save.state_dict(),
            'state_dict_simnet': simnet_to_save.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'optimizer_simnet': optimizer_simnet.state_dict(),
        }, is_best, args.out, tag='base')

        test_accs.append(test_acc)

        with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
            ofile.write(f'epoch: {epoch}, acc-known: {test_acc_known}\n')
            ofile.write(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}\n')
            ofile.write(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}\n')

    # close writer
    writer.close()

    # pseudo-label generation and selection
    model.zero_grad()
    lbl_unlbl_dict = pickle.load(open(f'{args.split_root}/{args.dataset}_{args.lbl_percent}_{args.novel_percent}_{args.split_id}.pkl', 'rb'))
    total_samples = len(lbl_unlbl_dict['labeled_idx']) + len(lbl_unlbl_dict['unlabeled_idx'])
    # no_pl_perclass = int((args.pl_percent*total_samples)/(args.no_class*100))
    no_pl_perclass = int((args.lbl_percent*total_samples)/(args.no_class*100))
    pl_dict, pl_acc, pl_no = pseudo_labeling(args, pl_loader, model, list(range(args.no_known, args.no_class)), no_pl_perclass)
    with open(os.path.join(args.out, 'pseudo_labels_base.pkl'),"wb") as f:
        pickle.dump(pl_dict,f)

    with open(f'{args.out}/score_logger_base.txt', 'a+') as ofile:
        ofile.write(f'acc-pl: {pl_acc}, total-selected: {pl_no}\n')


def train(args, lbl_loader, unlbl_loader, model, optimizer, simnet, optimizer_simnet, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    losses_ce = AverageMeter()
    losses_pair = AverageMeter()
    losses_reg = AverageMeter()

    end = time.time()
    if not args.no_progress:
        p_bar = tqdm(range(args.iteration))

    train_loader = zip(lbl_loader, unlbl_loader)
    for batch_idx, (data_lbl, data_unlbl) in enumerate(train_loader):
        inputs_l, targets_l, _ = data_lbl
        (inputs_u_w, inputs_u_s), _, _ = data_unlbl

        inputs = interleave(
            torch.cat((inputs_l, inputs_u_w, inputs_u_s)), 2*args.mu+1).cuda()
        targets_l = targets_l.cuda()
        batch_l = inputs_l.shape

        model.train()
        # create intermediate model for bi-level optimization
        model_, _ = build_model(args)
        model_ = model_.cuda()
        model_.load_state_dict(model.state_dict())

        feat, logits = model_(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_l = logits[:batch_l[0]]
        logits_u_w, logits_u_s = logits[batch_l[0]:].chunk(2)

        feat = de_interleave(feat, 2*args.mu+1)
        feat_l = feat[:batch_l[0]]
        feat_u_w, _ = feat[batch_l[0]:].chunk(2)
        feats = torch.cat((feat_l, feat_u_w), 0)

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        mask_pl = max_probs_pl.ge(args.threshold).float()

        feats = feats.unsqueeze(0).repeat(feats.shape[0],1,1)
        feats_t = torch.transpose(feats,0,1)
        feat_pairs = torch.cat((feats, feats_t),2).view(-1, feats.shape[2]*2)
        sim_feat = simnet(feat_pairs).view(-1,feats.shape[0])

        class_logit = torch.cat((logits_l, logits_u_w), 0)
        sim_prob = sim_matrix(F.softmax(class_logit,dim=1), F.softmax(class_logit,dim=1), args)

        loss_pair = symmetric_mse_loss(sim_prob.view(-1), sim_feat.view(-1))/sim_feat.view(-1).shape[0]
        loss_reg = entropy(torch.mean(F.softmax(class_logit, dim=1), 0), input_as_probabilities = True)
        loss_ce_supervised = F.cross_entropy(class_logit[:batch_l[0]], targets_l)
        loss_ce_pseudo = (F.cross_entropy(logits_u_s, targets_u_pl, reduction='none') * mask_pl).mean()
        loss_ce = loss_ce_supervised + loss_ce_pseudo

        loss = loss_pair - loss_reg + loss_ce

        model_.zero_grad()
        # compute gradients for the intermediate update
        if torch.cuda.device_count() > 1:
            grads = torch.autograd.grad(loss, (model_.module.params()), create_graph=True)
        else:
            grads = torch.autograd.grad(loss, (model_.params()), create_graph=True)

        # update the model parameters
        if torch.cuda.device_count() > 1:
            model_.module.update_params(lr_inner=1e-3, source_params=grads)
        else:
            model_.update_params(lr_inner=1e-3, source_params=grads)
        
        del grads

        # update simnet parameters
        feat, logits = model_(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_l = logits[:batch_l[0]]
        loss_ce_supervised = F.cross_entropy(logits_l, targets_l)

        optimizer_simnet.zero_grad()
        loss_ce_supervised.backward()
        optimizer_simnet.step()

        # update main model parameters with updated simnet
        feat, logits = model(inputs)
        logits = de_interleave(logits, 2*args.mu+1)
        logits_l = logits[:batch_l[0]]
        logits_u_w, logits_u_s = logits[batch_l[0]:].chunk(2)

        feat = de_interleave(feat, 2*args.mu+1)
        feat_l = feat[:batch_l[0]]
        feat_u_w, _ = feat[batch_l[0]:].chunk(2)
        feats = torch.cat((feat_l, feat_u_w), 0)

        pseudo_label = torch.softmax(logits_u_w.detach(), dim=-1)
        max_probs_pl, targets_u_pl = torch.max(pseudo_label, dim=-1)
        mask_pl = max_probs_pl.ge(args.threshold).float()

        feats = feats.unsqueeze(0).repeat(feats.shape[0],1,1)
        feats_t = torch.transpose(feats,0,1)
        feat_pairs = torch.cat((feats, feats_t),2).view(-1, feats.shape[2]*2)

        # no gradients for the simnet parameters
        with torch.no_grad():
            sim_feat = simnet(feat_pairs).view(-1,feats.shape[0])
        class_logit = torch.cat((logits_l, logits_u_w), 0)
        sim_prob = sim_matrix(F.softmax(class_logit,dim=1), F.softmax(class_logit,dim=1), args)

        loss_pair = symmetric_mse_loss(sim_prob.view(-1), sim_feat.view(-1))/sim_feat.view(-1).shape[0]
        loss_reg = entropy(torch.mean(F.softmax(class_logit, dim=1), 0), input_as_probabilities = True)
        loss_ce_supervised = F.cross_entropy(class_logit[:batch_l[0]], targets_l)
        loss_ce_pseudo = (F.cross_entropy(logits_u_s, targets_u_pl, reduction='none') * mask_pl).mean()
        loss_ce = loss_ce_supervised + loss_ce_pseudo

        final_loss = loss_pair - loss_reg + loss_ce

        losses.update(final_loss.item(), inputs_l.size(0))
        losses_ce.update(loss_ce.item(), inputs_l.size(0))
        losses_pair.update(loss_pair.item(), inputs_l.size(0))
        losses_reg.update(loss_reg.item(), inputs_l.size(0))

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if not args.no_progress:
            p_bar.set_description("train epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. l_ce: {loss_ce:.4f}. l_pair: {loss_pair:.4f}. l_reg: {loss_reg:.4f}".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=args.iteration,
                bt=batch_time.avg,
                loss=losses.avg,
                loss_ce=losses_ce.avg,
                loss_pair=losses_pair.avg,
                loss_reg=losses_reg.avg,
                ))
            p_bar.update()

    if not args.no_progress:
        p_bar.close()

    return losses.avg


def test_known(args, test_loader, model, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s. loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    return top1.avg


def test_cluster(args, test_loader, model, epoch, offset=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    gt_targets =[]
    predictions = []
    model.eval()

    if not args.no_progress:
        test_loader = tqdm(test_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs = model(inputs)
            _, max_idx = torch.max(outputs, dim=1)
            predictions.extend(max_idx.cpu().numpy().tolist())
            gt_targets.extend(targets.cpu().numpy().tolist())
            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                test_loader.set_description("test epoch: {epoch}/{epochs:4}. itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    predictions = np.array(predictions)
    gt_targets = np.array(gt_targets)

    predictions = torch.from_numpy(predictions)
    gt_targets = torch.from_numpy(gt_targets)
    eval_output = hungarian_evaluate(predictions, gt_targets, offset)

    return eval_output


if __name__ == '__main__':
    cudnn.benchmark = True
    main()
