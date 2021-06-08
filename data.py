from PIL import Image
import os
import os.path
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import CIFAR10, SVHN
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
from collections import Counter

#special class for tagging data as normal and anomalous
# By default it is in semisupervised setting with all training data as normal samples. If you turn on unsupervised setting, it will add a %ge of anomalies, but they are indistinguishable from normal samples.

AUG_LIMIT = 0.5
class CIFAR10Anom(CIFAR10):
        
    def __init__(self, root, stage='train',  transform=None, target_transform=None,
                 download=True, anom_classes=None, valid_split= 0.05,anom_ratio=0,
                 aug=False, aug_transform= None, setting = 'ss',seed=0):
        np.random.seed(seed)
        train_or_valid = True if stage == 'train' or stage == 'valid' else False
        self.aug_transform = aug_transform
        super(CIFAR10Anom, self).__init__(root, train=train_or_valid, transform=transform, target_transform=target_transform,
                 download=download)
        
        if len(set(anom_classes) & set(self.class_to_idx.keys()))==0:
            print('No anomaly class found, will be trained on all the classes')

        anom_class = -1
        norm_class = 1
        
        print(self.class_to_idx)
        
        anom_mapping= dict((i,anom_class) if c in anom_classes else (i,norm_class) for c,i in self.class_to_idx.items())
        print(anom_mapping)
        
        if train_or_valid:
            self.targets = [anom_mapping[key] for key in self.targets]
            norm_indices = np.where(np.array(self.targets) == norm_class)[0]
            anom_indices = np.where(np.array(self.targets) == anom_class)[0]
            if aug:
                imgs = self.data[norm_indices]
                arr = np.arange(imgs.shape[0])
                np.random.shuffle(arr)
                alpha = np.random.uniform(AUG_LIMIT,1,imgs.shape[0]).reshape(-1,1,1,1)
                imgs2 = imgs[arr]
                imgs_i = (alpha*imgs).astype(np.uint8) + ((1-alpha)*imgs2).astype(np.uint8)
                self.data[norm_indices] = imgs_i
                
            valid_indices = np.random.choice(norm_indices, int(len(norm_indices)*valid_split), replace=False)
            train_indices = list(set(norm_indices) - set(valid_indices))
            train_anom_indices = np.random.choice(anom_indices, int(len(train_indices)*anom_ratio), replace=False)
            valid_anom_indices = np.random.choice(list(set(anom_indices)-set(train_anom_indices)), int(len(valid_indices)*0.1), replace=False)
            
            train_indices = np.concatenate((train_indices, train_anom_indices))
            valid_indices = np.concatenate((valid_indices, valid_anom_indices))
            print(len(train_indices),len(valid_indices), len(train_anom_indices), len(valid_anom_indices))
            
            np.random.shuffle(train_indices)
            np.random.shuffle(valid_indices)
                
            if stage == 'train':
                self.data = self.data[train_indices]
                self.targets = np.array(self.targets)[train_indices]
            elif stage == 'valid':
                self.data = self.data[valid_indices]
                self.targets = np.array(self.targets)[valid_indices]
                
        elif stage == 'test':
            self.targets = [anom_mapping[key] for key in self.targets]
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        
        if target == 1 and self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.transform is not None:
            img = self.transform(img)
            

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR10_OOD(CIFAR10):
        
    def __init__(self, root, anom_root, stage='train',  transform=None, target_transform=None,
                 download=True, valid_split= 0.01,anom_ratio=0, aug=False, 
                 aug_transform= None, corruption = 0,setting = 'ss',seed=0):
        np.random.seed(seed)
        train_or_valid = True if stage == 'train' or stage == 'valid' else False

        self.aug_transform = aug_transform
        super(CIFAR10_OOD, self).__init__(root, train=train_or_valid, transform=transform, target_transform=target_transform,
                 download=download)
        anom_class = -1
        norm_class = 1
        
        if train_or_valid:
            if aug:
                normal_set = CIFAR10(root=root, train=True,transform=None)
                imgs = normal_set.data
                arr = np.arange(normal_set.data.shape[0])
                np.random.shuffle(arr)
                alpha = np.random.uniform(AUG_LIMIT,1,imgs.shape[0]).reshape(-1,1,1,1)
                imgs2 = imgs[arr]
                imgs_i = (alpha*imgs).astype(np.uint8) + ((1-alpha)*imgs2).astype(np.uint8)
                normal_set.data = imgs_i
            else:
                normal_set = CIFAR10(root=root, train=True,transform=transform,download=True)
            normal_set.targets = np.array(np.ones(len(normal_set.targets)),dtype=int)
            anom_set = SVHN(root= anom_root,transform=transform, download=True,split='train')
            anom_set.data = anom_set.data.transpose(0,2,3,1)
            anom_set.labels = np.array(-1*np.ones(len(anom_set.labels)),dtype=int)
            
            anom_count = int(len(normal_set.targets)*0.2)
            print(anom_count)
            
            self.data = np.vstack((normal_set.data,anom_set.data[500:500+anom_count]))
            self.targets = np.concatenate((normal_set.targets,anom_set.labels[500:500+anom_count]), axis=0)
            
            norm_indices = np.where(self.targets==norm_class)[0]
            anom_indices =  np.where(self.targets==anom_class)[0]
            valid_indices = np.random.choice(norm_indices, int(len(norm_indices)*valid_split), replace=False)
            train_indices = list(set(norm_indices) - set(valid_indices))
            train_anom_indices = np.random.choice(anom_indices, int(len(train_indices)*anom_ratio), replace=False)
            valid_anom_indices = np.random.choice(list(set(anom_indices)-set(train_anom_indices)), int(len(valid_indices)*0.1), replace=False)
            train_indices = np.concatenate((train_indices, train_anom_indices))
            valid_indices = np.concatenate((valid_indices, valid_anom_indices))
            print(len(train_indices),len(valid_indices), len(train_anom_indices), len(valid_anom_indices))
            
            np.random.shuffle(train_indices)
            np.random.shuffle(valid_indices)
            if setting == 'unsupervised':
                self.targets = np.zeros_like(self.targets) + norm_class
            if stage == 'train':
                self.data = self.data[train_indices]
                self.targets = np.array(self.targets)[train_indices]
                if corruption > 0:
                    self.targets[0:int(corruption*len(self.targets))] = norm_class
                print(self.data.shape, self.targets.shape)
            elif stage == 'valid':
                self.data = self.data[valid_indices]
                self.targets = np.array(self.targets)[valid_indices]
        
        elif stage == 'test':
            normal_test = CIFAR10(root=root, train=False,transform=transform)
            normal_test.targets = np.array(np.ones(len(normal_test.targets)),dtype=int)
            anom_test = SVHN(root= anom_root,transform=transform, download=True,split='test')
            anom_test.data = anom_test.data.transpose(0,2,3,1)
            anom_test.labels = np.array(-1*np.ones(len(anom_test.labels)),dtype=int)
            anom_count = int(0.1*len(normal_test.targets))

            self.data = np.vstack((normal_test.data,anom_test.data[:anom_count+1]))
            self.targets = np.concatenate((normal_test.targets,anom_test.labels[:anom_count+1]), axis=0)
            indices = np.array(list(range(len(self.targets))))[torch.randperm(len(self.targets))]
            print(len(indices))
            self.data = self.data[indices]
            self.targets = np.array(self.targets)[indices]
        else:
            print(f'invalid stage: {stage}')
            raise
                
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if target == 1 and self.aug_transform is not None:
            img = self.aug_transform(img)

        if self.transform is not None:
            img = self.transform(img)
            

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

                
def load_data(dataset, ood, anom_classes,anom_ratio, corruption, seed, augmentation = False,learning_setting = 'ss', valid_split = 0.1):
    if dataset == 'cifar10':
        root = './data/CIFAR10'
        anom_root = './data/SVHN'
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        

        aug_transform = None

        if ood:
            print('In CIFAR10 OOD area')
            
            dataset_train = CIFAR10_OOD(root=root, anom_root = anom_root,stage='train',
                                 transform = transform,
                                 valid_split= valid_split, aug = augmentation, aug_transform = aug_transform,
                                 anom_ratio=anom_ratio, seed = seed, corruption = corruption)
            dataset_valid = CIFAR10_OOD(root=root, anom_root = anom_root,stage='valid',
                                 transform = transform,
                                 valid_split= valid_split, aug = augmentation, aug_transform = aug_transform,
                                 anom_ratio=anom_ratio, seed = seed)
            dataset_test =  CIFAR10_OOD(root=root, anom_root = anom_root,stage='test',
                                 transform = transform, seed = seed)
                
        else:
            dataset_train = CIFAR10Anom(root=root,stage='train',
                                     transform = transform,anom_classes=anom_classes,
                                     valid_split= valid_split, aug = augmentation, aug_transform = aug_transform,
                                     anom_ratio=anom_ratio, seed = seed)
            dataset_valid = CIFAR10Anom(root=root,stage='valid',
                                     transform = transform,anom_classes=anom_classes,
                                     valid_split= valid_split, aug = augmentation, aug_transform = aug_transform,
                                     anom_ratio=anom_ratio, seed = seed)
            dataset_test = CIFAR10Anom(root=root,stage='test',
                                     transform = transform,anom_classes=anom_classes, seed = seed)
            
    else:
        print(f'invalid dataset: {dataset}')
        raise
            
    return dataset_train, dataset_valid, dataset_test
