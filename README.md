# OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning

Implementation of [OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning](https://arxiv.org/abs/2207.02261).

Semi-supervised learning (SSL) is one of the dominant approaches to address the annotation bottleneck of supervised learning. Recent SSL methods can effectively leverage a large repository of unlabeled data to improve performance while relying on a small set of labeled data. One common assumption in most SSL methods is that the labeled and unlabeled data are from the same underlying data distribution. However, this is hardly the case in many real-world scenarios, which limits their applicability. In this work, instead, we attempt to solve the recently proposed challenging open-world SSL problem that does not make such an assumption. In the open-world SSL problem, the objective is to recognize samples of known classes, and simultaneously detect and cluster samples belonging to novel classes present in unlabeled data. This work introduces OpenLDN that utilizes a pairwise similarity loss to discover novel classes. Using a bi-level optimization rule this pairwise similarity loss exploits the information available in the labeled set to implicitly cluster novel class samples, while simultaneously recognizing samples from known classes. After discovering novel classes, OpenLDN transforms the open-world SSL problem into a standard SSL problem to achieve additional performance gains using existing SSL methods. Our extensive experiments demonstrate that OpenLDN outperforms the current state-of-the-art methods on multiple popular classification benchmarks while providing a better accuracy/training time trade-off.


## Training
```shell
# For CIFAR10 50% Labels and 50% Novel Classes 
python3 train.py --dataset cifar10 --lbl-percent 50 --novel-percent 50 --arch resnet18

# For CIFAR100 50% Labels and 50% Novel Classes 
python3 train.py --dataset cifar100 --lbl-percent 50 --novel-percent 50 --arch resnet18
```

## Citation
```
@inproceedings{rizve2022openldn,
Title={OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning},
Author={Mamshad Nayeem Rizve and Navid Kardan and Salman Khan and Fahad Shahbaz Khan and Mubarak Shah},
booktitle={European Conference on Computer Vision},
Year={2022}
```
