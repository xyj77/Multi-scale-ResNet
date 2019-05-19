# -*- coding:utf-8 -*-
import os
import sys
import pickle
import numpy as np

DATA_DIR = '../../Data'

def getSampleSet(path):
    path = os.path.join(path, 'sample.bin')
    with open(path, 'rb') as fp:
        data=pickle.load(fp)#顺序导出变量
    return data

def randomSplit(sampleSet, ratio):
    length = len(sampleSet)
    sampleSet = list(sampleSet)
    index = np.random.permutation(length)
    train, test = [], []
    for i in range(int(length*ratio)):
        train.append(sampleSet[i])
    for i in range(int(length*ratio), length):
        test.append(sampleSet[i])
    return train, test
    
def parseName(imgName):
    img = imgName.split('_')
    return img[0] + '_' + img[1] + '_' + img[2]

# def file_del(path):
    # for img in os.listdir(path):
        # if (str(img[:10]) in ['004316201', '132126761']) & (int(path[-1]) == 0):
            # os.remove(os.path.join(path, img))
            # print ("    Delete File: " + os.path.join(path, img))
                
def split_dir(dir, trainSet, testSet):
    fp_train = open(os.path.join(dir, "train.txt"), 'w')
    fp_test = open(os.path.join(dir, "test.txt"), 'w')
    print('*'*20, len(trainSet), len(testSet))
    for sub_dir in os.listdir(dir):
        path = os.path.join(dir, sub_dir)
        if os.path.isdir(path):
            for img in os.listdir(path):
                Level = sub_dir
                p = os.path.join(path, img)
                img = parseName(img)
                if img in testSet:
                    fp_test.write(Level + ' ' + p + '\n')
                else:
                    fp_train.write(Level + ' ' + p + '\n')
    fp_test.close()
    fp_train.close()

def splitDataSet(path, ratio = 0.6):
    sampleSet = getSampleSet(path)
    train, test = randomSplit(sampleSet, ratio)
    if os.path.isdir(path):
        split_dir(path, train, test)
        print path + '  done!'

def test(TYPE = 'WHO', ratio = 0.6):
    root = os.path.join(DATA_DIR, TYPE)
    for dir in os.listdir(root):
        path = os.path.join(root, dir)
        sampleSet = getSampleSet(path)
        train, test = randomSplit(sampleSet, ratio)
        if os.path.isdir(path):
            split_dir(path, train, test)
            print path + '  done!'

if __name__ == '__main__':
    pass
    # test()
    # test('Edmondson')
    # test('Binary')
