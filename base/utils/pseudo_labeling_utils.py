import random
import time
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .utils import AverageMeter
from .evaluate_utils import hungarian_evaluate


def pseudo_labeling(args, data_loader, model, novel_classes, no_pl_perclass):
    batch_time = AverageMeter()
    end = time.time()
    pseudo_idx = []
    pseudo_target = []
    pseudo_maxval = []
    gt_target = []
    model.eval()

    if not args.no_progress:
        data_loader = tqdm(data_loader)

    with torch.no_grad():
        for batch_idx, (inputs, targets, indexs) in enumerate(data_loader):
            inputs = inputs.cuda()
            targets = targets.cuda()
            _, outputs = model(inputs)
            out_prob = F.softmax(outputs, dim=1)
            max_value, max_idx = torch.max(out_prob, dim=1)

            pseudo_target.extend(max_idx.cpu().numpy().tolist())
            pseudo_maxval.extend(max_value.cpu().numpy().tolist())
            pseudo_idx.extend(indexs.numpy().tolist())
            gt_target.extend(targets.cpu().numpy().tolist())

            batch_time.update(time.time() - end)
            end = time.time()
            if not args.no_progress:
                data_loader.set_description("pseudo-labeling itr: {batch:4}/{iter:4}. btime: {bt:.3f}s.".format(
                    batch=batch_idx + 1,
                    iter=len(data_loader),
                    bt=batch_time.avg,
                ))
        if not args.no_progress:
            data_loader.close()

    pseudo_target = np.array(pseudo_target)
    gt_target = np.array(gt_target)
    pseudo_maxval = np.array(pseudo_maxval)
    pseudo_idx = np.array(pseudo_idx)

    #class balance the selected pseudo-labels
    blnc_idx_list = []
    for class_idx in novel_classes:
        current_class_idx = np.where(pseudo_target==class_idx)
        if len(np.where(pseudo_target==class_idx)[0]) > 0:
            current_class_maxval = pseudo_maxval[current_class_idx]
            sorted_idx = np.argsort(current_class_maxval)[::-1]
            current_class_idx = current_class_idx[0][sorted_idx[:no_pl_perclass]] 
            blnc_idx_list.extend(current_class_idx)

    blnc_idx_list = np.array(blnc_idx_list)
    pseudo_target = pseudo_target[blnc_idx_list]
    pseudo_idx = pseudo_idx[blnc_idx_list]
    gt_target = gt_target[blnc_idx_list]

    pl_eval_output = hungarian_evaluate(torch.from_numpy(pseudo_target), torch.from_numpy(gt_target), args.no_known) # for sanity check only
    pseudo_label_dict = {'pseudo_idx': pseudo_idx.tolist(), 'pseudo_target':pseudo_target.tolist()}
 
    return pseudo_label_dict, pl_eval_output["acc"], len(pseudo_idx)