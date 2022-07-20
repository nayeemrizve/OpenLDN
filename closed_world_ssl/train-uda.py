import argparse
import os
import time
import pickle
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tensorboardX import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from utils.evaluate_utils import hungarian_evaluate
from tqdm import tqdm
from utils.pseudo_labeling_utils import pseudo_labeling
from models.build_model import build_model
import torch.backends.cudnn as cudnn
from datasets.datasets_uda import get_dataset
from utils.utils import Bar, AverageMeter, accuracy, SemiLoss, set_seed, WeightEMA, interleave, save_checkpoint

best_acc = 0

def main():
    parser = argparse.ArgumentParser(description='UDA Training')
    parser.add_argument('--data-root', default=f'data', help='directory to store data')
    parser.add_argument('--split-root', default=f'random_splits', help='directory to store datasets')
    parser.add_argument('--out', default=f'outputs', help='directory to output the result')
    parser.add_argument('--gpu-id', default='0', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
    parser.add_argument('--num-workers', type=int, default=4, help='number of workers')
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100', 'svhn', 'tinyimagenet', 'oxfordpets', 'aircraft', 'stanfordcars', 'imagenet100', 'herbarium'], help='dataset name')
    parser.add_argument('--arch', default='resnet18', type=str, choices=['resnet18', 'resnet50'], help='model architecure')
    parser.add_argument('--batch-size', default=64, type=int, help='train batchsize')
    parser.add_argument('--lr', '--learning-rate', default=0.002, type=float, help='initial learning rate')
    parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')
    parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=None, type=int, help="random seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--no-progress', action='store_true', help="don't use progress bar")
    parser.add_argument('--pl-freq', default=10, type=int, help='frequency of iterative pseudo-labeling, default 10')
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--lambda-u', default=1, type=float, help='coefficient of unlabeled loss')
    parser.add_argument('--T', default=0.4, type=float, help='pseudo label temperature')
    parser.add_argument('--threshold', default=0.8, type=float, help='pseudo label threshold')
    parser.add_argument('--epochs', default=1024, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--train-iteration', type=int, default=1024, help='Number of iteration per epoch')
    parser.add_argument('--ssl-indexes', default='', type=str, help='path to random data split')
    parser.add_argument('--lbl-percent', type=int, default=50, help='percent of labeled data')
    parser.add_argument('--novel-percent', default=50, type=int, help='percentage of novel classes, default 50')

    args = parser.parse_args()
    state = {k: v for k, v in args._get_kwargs()}
    print(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
    global best_acc
    writer = SummaryWriter(args.out)

    if args.local_rank == -1:
        device = torch.device('cuda', args.gpu_id)
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.world_size = torch.distributed.get_world_size()
        args.n_gpu = 1

    args.device = device

    if args.seed is not None:
        set_seed(args)

    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
    
    with open(f'{args.out}/score_logger_uda.txt', 'a+') as ofile:
        ofile.write('************************************************************************\n\n')
        ofile.write(' | '.join(f'{k}={v}' for k, v in vars(args).items()))
        ofile.write('\n\n************************************************************************\n')

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

    # use the base pseudo-labels
    args.pl_dict = os.path.join(args.out, 'pseudo_labels_base.pkl')

    # load dataset
    args.no_known = args.no_class - int((args.novel_percent*args.no_class)/100)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    lbl_dataset, unlbl_dataset, pl_dataset, test_dataset_known, test_dataset_novel, test_dataset_all = get_dataset(args)

    if args.local_rank == 0:
        torch.distributed.barrier()

    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler
    lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
    pl_loader = DataLoader(pl_dataset, sampler=SequentialSampler(pl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_known = DataLoader(test_dataset_known, sampler=SequentialSampler(test_dataset_known), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_novel = DataLoader(test_dataset_novel, sampler=SequentialSampler(test_dataset_novel), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)
    test_loader_all = DataLoader(test_dataset_all, sampler=SequentialSampler(test_dataset_all), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=False)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # create model
    model = build_model(args)
    ema_model = build_model(args, ema=True)

    cudnn.benchmark = True
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer= WeightEMA(args, model, ema_model, alpha=args.ema_decay)
    
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
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

    model.zero_grad()
    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))
        train_loss, train_loss_x, train_loss_u = train(args, lbl_loader, unlbl_loader, model, optimizer, ema_optimizer, train_criterion, epoch, True)
        all_cluster_results = test_cluster(args, test_loader_all, ema_model, epoch)
        novel_cluster_results = test_cluster(args, test_loader_novel, ema_model, epoch, offset=args.no_known)
        test_acc_known = test_known(args, test_loader_known, ema_model, epoch)
        test_acc = all_cluster_results["acc"]

        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)

        print(f'epoch: {epoch}, acc-known: {test_acc_known}')
        print(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}')
        print(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}')

        with open(f'{args.out}/score_logger_uda.txt', 'a+') as ofile:
            ofile.write(f'epoch: {epoch}, acc-known: {test_acc_known}\n')
            ofile.write(f'epoch: {epoch}, acc-novel: {novel_cluster_results["acc"]}, nmi-novel: {novel_cluster_results["nmi"]}\n')
            ofile.write(f'epoch: {epoch}, acc-all: {all_cluster_results["acc"]}, nmi-all: {all_cluster_results["nmi"]}, best-acc: {best_acc}\n')

        model_to_save = model.module if hasattr(model, "module") else model
        ema_model_to_save = ema_model.module if hasattr(ema_model, "module") else ema_model

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_model_to_save.state_dict(),
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best, args.out, tag='uda')
        test_accs.append(test_acc)

        if (epoch+1)%args.pl_freq==0:
            #pseudo-label generation and selection
            label_unlabel_dict = pickle.load(open(args.ssl_indexes, 'rb'))
            no_pl_perclass = int(len(label_unlabel_dict['labeled_idx'])/args.no_known)
            pl_dict, pl_acc, pl_no = pseudo_labeling(args, pl_loader, ema_model, list(range(args.no_known, args.no_class)), no_pl_perclass)
            with open(os.path.join(args.out, 'pseudo_labels_uda.pkl'),"wb") as f:
                pickle.dump(pl_dict,f)
            args.pl_dict = os.path.join(args.out, 'pseudo_labels_uda.pkl')

            lbl_dataset, unlbl_dataset, _, _, _, _ = get_dataset(args)
            lbl_loader = DataLoader(lbl_dataset, sampler=train_sampler(lbl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
            unlbl_loader = DataLoader(unlbl_dataset, sampler=train_sampler(unlbl_dataset), batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)
            
            with open(f'{args.out}/score_logger_uda.txt', 'a+') as ofile:
                ofile.write(f'***epoch: {epoch}, acc-pl: {pl_acc}, total-selected: {pl_no}***\n')
            
            writer.add_scalar('pl/1.acc_pl', pl_acc, epoch)
            writer.add_scalar('pl/2.no_selected', pl_no, epoch)

        writer.add_scalar('train/1.train_loss', train_loss, epoch)
        writer.add_scalar('test/1.acc_known', test_acc_known, epoch)
        writer.add_scalar('test/2.acc_novel', novel_cluster_results['acc'], epoch)
        writer.add_scalar('test/3.nmi_novel', novel_cluster_results['nmi'], epoch)
        writer.add_scalar('test/4.acc_all', all_cluster_results['acc'], epoch)
        writer.add_scalar('test/5.nmi_all', all_cluster_results['nmi'], epoch)
    
    # close writer
    writer.close()

    print('Best acc:')
    print(best_acc)

    print('Mean acc:')
    print(np.mean(test_accs[-20:]))


def train(args, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.train_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()
    for batch_idx in range(args.train_iteration):
        try:
            inputs_x, targets_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, _ = labeled_train_iter.next()

        try:
            (inputs_u_w, inputs_u_s), _, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u_w, inputs_u_s), _, _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, args.no_class).scatter_(1, targets_x.view(-1,1).long(), 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u_w = inputs_u_w.cuda()
            inputs_u_s = inputs_u_s.cuda()

        l = np.random.beta(args.alpha, args.alpha)
        l = max(l, 1-l)
        idx = torch.randperm(inputs_x.size(0))

        # mixup for only the labeled samples
        input_a, input_b = inputs_x, inputs_x[idx]
        target_a, target_b = targets_x, targets_x[idx]

        mixed_input_x = l * input_a + (1 - l) * input_b
        mixed_target_x = l * target_a + (1 - l) * target_b

        mixed_input = torch.cat([mixed_input_x, inputs_u_w, inputs_u_s], dim=0)
        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)
        logits_u_w, logits_u_s = logits_u.chunk(2)

        Lx = -torch.mean(torch.sum(F.log_softmax(logits_x, dim=1) * mixed_target_x, dim=1))
        
        targets_u = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
        max_probs, _ = torch.max(targets_u, dim=-1)
        mask = max_probs.ge(args.threshold).float()

        Lu = (-(targets_u * torch.log_softmax(logits_u_s, dim=-1)).sum(dim=-1) * mask).mean()

        loss = Lx + args.lambda_u * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Loss_x: {loss_x:.4f} | Loss_u: {loss_u:.4f}'.format(
                    batch=batch_idx + 1,
                    size=args.train_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_x.avg, losses_u.avg,)


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
            outputs = model(inputs)
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
            outputs = model(inputs)
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
    main()
