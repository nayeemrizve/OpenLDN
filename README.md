# OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openldn-learning-to-discover-novel-classes/open-world-semi-supervised-learning-on-cifar)](https://paperswithcode.com/sota/open-world-semi-supervised-learning-on-cifar?p=openldn-learning-to-discover-novel-classes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openldn-learning-to-discover-novel-classes/open-world-semi-supervised-learning-on-cifar-1)](https://paperswithcode.com/sota/open-world-semi-supervised-learning-on-cifar-1?p=openldn-learning-to-discover-novel-classes)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/openldn-learning-to-discover-novel-classes/open-world-semi-supervised-learning-on-1)](https://paperswithcode.com/sota/open-world-semi-supervised-learning-on-1?p=openldn-learning-to-discover-novel-classes)

Implementation of [OpenLDN: Learning to Discover Novel Classes for Open-World Semi-Supervised Learning](https://arxiv.org/abs/2207.02261).

Semi-supervised learning (SSL) is one of the dominant approaches to address the annotation bottleneck of supervised learning. Recent SSL methods can effectively leverage a large repository of unlabeled data to improve performance while relying on a small set of labeled data. One common assumption in most SSL methods is that the labeled and unlabeled data are from the same underlying data distribution. However, this is hardly the case in many real-world scenarios, which limits their applicability. In this work, instead, we attempt to solve the recently proposed challenging open-world SSL problem that does not make such an assumption. In the open-world SSL problem, the objective is to recognize samples of known classes, and simultaneously detect and cluster samples belonging to novel classes present in unlabeled data. This work introduces OpenLDN that utilizes a pairwise similarity loss to discover novel classes. Using a bi-level optimization rule this pairwise similarity loss exploits the information available in the labeled set to implicitly cluster novel class samples, while simultaneously recognizing samples from known classes. After discovering novel classes, OpenLDN transforms the open-world SSL problem into a standard SSL problem to achieve additional performance gains using existing SSL methods. Our extensive experiments demonstrate that OpenLDN outperforms the current state-of-the-art methods on multiple popular classification benchmarks while providing a better accuracy/training time trade-off.


## Training
```shell
# For CIFAR10 50% Labels and 50% Novel Classes 
python3 train.py --dataset cifar10 --lbl-percent 50 --novel-percent 50 --arch resnet18

# For CIFAR100 50% Labels and 50% Novel Classes 
python3 train.py --dataset cifar100 --lbl-percent 50 --novel-percent 50 --arch resnet18

For training on the other datasets, please download the dataset and put under the "name_of_the_dataset" folder and put the train and validation/test images under "train" and "test" folder. After that, please set the value of data_root argument as "name_of_the_dataset".

# For Tiny ImageNet 50% Labels and 50% Novel Classes
python3 train.py --dataset tinyimagenet --lbl-percent 50 --novel-percent 50 --arch resnet18

# For ImageNet-100 50% Labels and 50% Novel Classes
python3 train.py --dataset imagenet100 --lbl-percent 50 --novel-percent 50 --arch resnet50

# For Oxford-IIIT Pet 50% Labels and 50% Novel Classes
python3 train.py --dataset oxfordpets --lbl-percent 50 --novel-percent 50 --arch resnet18

# For FGVC-Aircraft 50% Labels and 50% Novel Classes
python3 train.py --dataset aircraft --lbl-percent 50 --novel-percent 50 --arch resnet18

# For Stanford-Cars 50% Labels and 50% Novel Classes
python3 train.py --dataset stanfordcars --lbl-percent 50 --novel-percent 50 --arch resnet18

# For Herbarium19 50% Labels and 50% Novel Classes
python3 train.py --dataset herbarium --lbl-percent 50 --novel-percent 50 --arch resnet18

# For SVHN 10% Labels and 50% Novel Classes
python3 train.py --dataset svhn --lbl-percent 10 --novel-percent 50 --arch resnet18
```

## Citation
```
@inproceedings{rizve2022openldn,
  title={Openldn: Learning to discover novel classes for open-world semi-supervised learning},
  author={Rizve, Mamshad Nayeem and Kardan, Navid and Khan, Salman and Shahbaz Khan, Fahad and Shah, Mubarak},
  booktitle={European Conference on Computer Vision},
  pages={382--401},
  year={2022},
  organization={Springer}
}
```
