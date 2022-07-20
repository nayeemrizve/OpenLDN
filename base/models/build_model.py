import torch

def build_model(args):
    if args.arch == 'resnet18':
        if args.dataset in ['cifar10', 'cifar100', 'svhn']:
            from . import resnet_cifar as models    
        elif args.dataset == 'tinyimagenet':
            from . import resnet_tinyimagenet as models
        else:
            from . import resnet as models
        model = models.resnet18(no_class=args.no_class)
        simnet = models.SimNet(1024, 100, 1)

    # use dataparallel if there's multiple gpus
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        simnet = torch.nn.DataParallel(simnet)
    return model, simnet
