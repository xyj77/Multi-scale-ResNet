# -*- coding:utf-8 -*-
from base.base_data_loader import BaseDataLoader

import os
import numpy as np
from PIL import Image
import scipy.io as sio
from SampleData import sampledata

import torch
from torchvision import transforms

from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


class MRINPY(BaseDataLoader):

    def __init__(self, config, train = True, transform=None, target_transform=None):
        super(MRINPY, self).__init__(config)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        
        self.path = os.path.join(config.data_path, config.data_type)
        self.Num_train = config.Num_train
        self.Num_test = config.Num_test
        self.model_type = config.exp_name 
        self.isAug = config.isAug
        
        # 二分类：I、II  &  III、IV
        if 'Ed' in config.data_type and config.classes == 2:   
            self.target_transform = self.binary
        
        #提取指定模态数据并进行预处理
        self.dict = config.dict
        self.Fusion = config.Fusion
        
        if self.train:
            (self.train_data, self.train_labels) = self.loadData(self.train, config.isSample)
            self.train_labels = torch.from_numpy(self.train_labels).type(torch.LongTensor)
        else:
            (self.test_data, self.test_labels) = self.loadData(self.train, config.isSample)
            self.test_labels = torch.from_numpy(self.test_labels).type(torch.LongTensor)
            
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # 提取融合数据
        img = self.extrData(img, self.dict, self.Fusion)
        if len(self.Fusion)<=3 and self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
        
    def binary(self, target):
        if target < 2:
            return 0
        else:
            return 1

    def getData(self, train=True, array=True):
        if train:
            if array:
                return self.train_data.numpy(), self.train_labels.numpy()
            else:
                return self.train_data, self.train_labels
        else:
            if array:
                return self.test_data.numpy(), self.test_labels.numpy()
            else:
                return self.test_data, self.test_labels

    def loadData(self, train, isSample):
        path = str(self.path) 
        if train:
            if isSample:
                Num_train = self.Num_train
                sampledata(path+'/train.txt', Num_train, 0, path+'/trainSample.txt') 
                fp = open(os.path.join(path, "trainSample.txt"), 'r')
            else:
                fp = open(os.path.join(path, "train.txt"), 'r')
        else:
            if isSample:
                Num_test = self.Num_test
                sampledata(path+'/test.txt', Num_test, 0, path+'/testSample.txt')
                fp = open(os.path.join(path, "testSample.txt"), 'r') 
            else:
                fp = open(os.path.join(path, "test.txt"), 'r')
            
        data = []  
        labels = []  
            
        line = fp.readline()
        while len(line):
            Level = int(line[0])
            imgpath = line[2:-1]
            
            # 是否进行数据扩充
            if (not self.isAug) and self.lineSearch(imgpath):
                line = fp.readline()
                continue
            
            if 'npy' in imgpath:
                mat = np.load(imgpath)
            else:
                mat = Image.open(imgpath) 
                
            data.append(mat)
            labels.append(Level)
            
            line = fp.readline()
            
        fp.close()
        return (data, np.array(labels, dtype='float32'))
    
    def extrData(self, img, dict, Fusion):
        image = []
        for modual in Fusion:
            image.append(img[:,:,dict[modual]])
        image = np.array(image)
        if len(Fusion) == 1:
            image = np.squeeze(image)
            image = Image.fromarray(image)
        elif len(Fusion) == 3:
            image = image.transpose (1, 2, 0)
            image = Image.fromarray(np.dot(255, image).astype('int8'), mode='RGB')
        else:
            image = torch.from_numpy(image).type(torch.FloatTensor)
            return image
        # 调整尺寸特别小的数据的大小
        w, h = image.size
        m = min(w, h)
        if m < 32:
            w, h = int(32.0*w/m), int(32.0*h/m)
            image = image.resize((w, h))  #重设宽，高
        
        return image
    
    def lineSearch(self, line, strlist=['_90','_270','_180','_lr','ud','tr','tr2']):
        for str in strlist:
            if str in line:
                return False
        return True


class SiameseNPY(Dataset):
    """
    Train: For each sample creates randomly a positive or a negative pair
    Test: Creates fixed pairs for testing
    """

    def __init__(self, npy_dataset):
        self.npy_dataset = npy_dataset
        self.train = npy_dataset.train
        self.transform = npy_dataset.transform
        self.extrData = npy_dataset.extrData
        
        #提取指定模态数据并进行预处理
        self.dict = npy_dataset.dict
        self.Fusion = npy_dataset.Fusion

        if self.train:
            self.train_labels = self.npy_dataset.train_labels
            self.train_data = self.npy_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}
        else:
            # generate fixed pairs for testing
            self.test_labels = self.npy_dataset.test_labels
            self.test_data = self.npy_dataset.test_data
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            positive_pairs = [[i,
                               random_state.choice(self.label_to_indices[self.test_labels[i]]),
                               1]
                              for i in range(0, len(self.test_data), 2)]

            negative_pairs = [[i,
                               random_state.choice(self.label_to_indices[
                                                       np.random.choice(
                                                           list(self.labels_set - set([self.test_labels[i]]))
                                                       )
                                                   ]),
                               0]
                              for i in range(1, len(self.test_data), 2)]
            self.test_pairs = positive_pairs + negative_pairs

    def __getitem__(self, index):
        obj_label = []  #用于记录图像真实标签
        if self.train:
            target = np.random.randint(0, 2)
            img1, label1 = self.train_data[index], self.train_labels[index]
            if target == 1:
                siamese_index = index
                while siamese_index == index:
                    siamese_index = np.random.choice(self.label_to_indices[label1])
            else:
                siamese_label = np.random.choice(list(self.labels_set - set([label1])))
                siamese_index = np.random.choice(self.label_to_indices[siamese_label])
            img2 = self.train_data[siamese_index]
            
            # 记录图像对标签
            obj_label.append(label1)
            obj_label.append(self.train_labels[siamese_index])
            
        else:
            img1 = self.test_data[self.test_pairs[index][0]]
            img2 = self.test_data[self.test_pairs[index][1]]
            target = self.test_pairs[index][2]
            
            # 记录图像对标签
            obj_label.append(self.test_labels[self.test_pairs[index][0]])
            obj_label.append(self.test_labels[self.test_pairs[index][1]])   

        # 提取融合数据
        img1 = self.extrData(img1, self.dict, self.Fusion)
        img2 = self.extrData(img2, self.dict, self.Fusion)
        if len(self.Fusion)<=3 and self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            
        return (img1, img2), target

    def __len__(self):
        return len(self.npy_dataset)
        

class TripletNPY(Dataset):
    """
    Train: For each sample (anchor) randomly chooses a positive and negative samples
    Test: Creates fixed triplets for testing
    """

    def __init__(self, npy_dataset):
        self.npy_dataset = npy_dataset
        self.train = npy_dataset.train
        self.transform = npy_dataset.transform
        self.extrData = npy_dataset.extrData
        
        #提取指定模态数据并进行预处理
        self.dict = npy_dataset.dict
        self.Fusion = npy_dataset.Fusion 

        if self.train:
            self.train_labels = self.npy_dataset.train_labels
            self.train_data = self.npy_dataset.train_data
            self.labels_set = set(self.train_labels.numpy())
            self.label_to_indices = {label: np.where(self.train_labels.numpy() == label)[0]
                                     for label in self.labels_set}

        else:
            self.test_labels = self.npy_dataset.test_labels
            self.test_data = self.npy_dataset.test_data
            # generate fixed triplets for testing
            self.labels_set = set(self.test_labels.numpy())
            self.label_to_indices = {label: np.where(self.test_labels.numpy() == label)[0]
                                     for label in self.labels_set}

            random_state = np.random.RandomState(29)

            triplets = [[i,
                         random_state.choice(self.label_to_indices[self.test_labels[i]]),
                         random_state.choice(self.label_to_indices[
                                                 np.random.choice(
                                                     list(self.labels_set - set([self.test_labels[i]]))
                                                 )
                                             ])
                         ]
                        for i in range(len(self.test_data))]
            self.test_triplets = triplets

    def __getitem__(self, index):
        obj_label = []  #用于记录图像真实标签
        if self.train:
            img1, label1 = self.train_data[index], self.train_labels[index]
            positive_index = index
            while positive_index == index:
                positive_index = np.random.choice(self.label_to_indices[label1])
            negative_label = np.random.choice(list(self.labels_set - set([label1])))
            negative_index = np.random.choice(self.label_to_indices[negative_label])
            img2 = self.train_data[positive_index]
            img3 = self.train_data[negative_index]
            
            # 记录图像对标签
            obj_label.append(label1)
            obj_label.append(positive_index)
            obj_label.append(negative_index)
            
        else:
            img1 = self.test_data[self.test_triplets[index][0]]
            img2 = self.test_data[self.test_triplets[index][1]]
            img3 = self.test_data[self.test_triplets[index][2]]
            
            # 记录图像对标签
            obj_label.append(self.test_labels[self.test_triplets[index][0]])
            obj_label.append(self.test_labels[self.test_triplets[index][1]])
            obj_label.append(self.test_labels[self.test_triplets[index][2]])

        # 提取融合数据
        img1 = self.extrData(img1, self.dict, self.Fusion)
        img2 = self.extrData(img2, self.dict, self.Fusion)
        img3 = self.extrData(img3, self.dict, self.Fusion)
        if len(self.Fusion)<=3 and self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []#, obj_label

    def __len__(self):
        return len(self.npy_dataset)

    
class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a mri-like dataset, samples n_classes and within these classes samples n_samples.
    Returns batches of size n_classes * n_samples
    """

    def __init__(self, dataset, n_classes, n_samples):
        if dataset.train:
            self.labels = dataset.train_labels
        else:
            self.labels = dataset.test_labels
        self.labels_set = list(set(self.labels.numpy()))
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0]
                                 for label in self.labels_set}
        for l in self.labels_set:
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.dataset = dataset
        self.batch_size = self.n_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < len(self.dataset):
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.n_samples])
                self.used_label_indices_count[class_] += self.n_samples
                if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.n_samples

    def __len__(self):
        return len(self.dataset) // self.batch_size

 
if __name__ == "__main__":
    pass 