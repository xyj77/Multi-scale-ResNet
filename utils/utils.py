# -*- coding:utf-8 -*-
from __future__ import division
import os
import csv
import torch
import itertools
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import interp
from scipy.misc import imsave

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc, recall_score, precision_score, f1_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

import argparse
def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args

def plot_confusion_matrix(cm, classes,
                          save_tag = '',
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('experiments/img/'+ save_tag + '_cfm.png')
    plt.close('all') # 关闭图    

def plot_roc_curve(y_true, y_pred, classes, save_tag):
    # # 绘制ROC曲线
    if len(classes) == 2:
        fpr, tpr, thresholds = roc_curve(y_true[:,1], y_pred[:,1])
        fpr[0], tpr[0] = 0, 0
        fpr[-1], tpr[-1] = 1, 1
        Auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (Auc))
        # # 记录ROC曲线以及曲线下面积              
        f = open('experiments/img/'+ save_tag + '_roc_record01.txt', 'ab+')
        f.write(save_tag + '   AUC:' +  str(Auc) + '\n')
        f.write('FPR:' + str(list(fpr)) + '\n')
        f.write('TPR:' + str(list(tpr)) + '\n\n')
        f.close()

        # # #字典中的key值即为csv中列名
        # # dataframe = pd.DataFrame({'FPR':fpr,'TPR':tpr})
        # # #将DataFrame存储为csv,index表示是否显示行名，default=True
        # # dataframe.to_csv('experiments/img/roc_record.csv', index=False, sep=',')  
    else:
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in classes:
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred[:, i])
            fpr[i][0], tpr[i][0] = 0, 0
            fpr [i][-1], tpr[i][-1] = 1, 1
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in classes]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in classes:
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= len(classes)

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot all ROC curves
        plt.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC (area = {0:0.2f})'
                       ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=2)

        plt.plot(fpr["macro"], tpr["macro"],
                 label='macro-average ROC (area = {0:0.2f})'
                       ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=2)

        colors = itertools.cycle(['aqua', 'darkorange', 'cornflowerblue'])
        for i, color in zip(classes, colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                     label='ROC of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i]))
            
            # # 记录ROC曲线以及曲线下面积          
            f = open('experiments/img/' + save_tag + '_roc_record.txt', 'ab+')
            f.write(save_tag + '  AUC of class {0}:{1:f}\n'.format(i, roc_auc[i]))
            f.write('FPR:' + str(list(fpr[i])) + '\n')
            f.write('TPR:' + str(list(tpr[i])) + '\n\n')
            f.close()
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color=(0.6, 0.6, 0.6), alpha=.8)
    plt.xlim([-0.05, 1.05])  
    plt.ylim([-0.05, 1.05])  
    plt.xlabel('False Positive Rate')  
    plt.ylabel('True Positive Rate')  
    plt.title('Receiver operating curve')  
    plt.legend(loc="lower right") 
    plt.savefig('experiments/img/'+ save_tag + '_roc.png')
    plt.close('all') # 关闭图    
    
def cnf_roc(y_true, y_pred, classes, isPlot, save_tag = ''):
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))    
    for i in range(len(y_true)): 
        y[i] = np.argmax(y_true[i,:])
        y_[i] = np.argmax(y_pred[i,:])
    cnf_mat = confusion_matrix(y, y_)
    print cnf_mat
    
    if isPlot:
        # # 绘制混淆矩阵
        plot_confusion_matrix(cnf_mat, range(classes), save_tag=save_tag)
        # # 绘制ROC曲线
        plot_roc_curve(y_true, y_pred, range(classes), save_tag)

    if classes > 2: 
        # 计算多分类评价值
        Sens = recall_score(y, y_, average='macro')
        Prec = precision_score(y, y_, average='macro')
        F1 = f1_score(y, y_, average='weighted') 
        Support = precision_recall_fscore_support(y, y_, beta=0.5, average=None)
        print Support
        return Sens, Prec, F1, cnf_mat
    else:
        Acc = 1.0*(cnf_mat[1][1]+cnf_mat[0][0])/len(y_true)
        Sens = 1.0*cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[1][0])
        Spec = 1.0*cnf_mat[0][0]/(cnf_mat[0][0]+cnf_mat[0][1])
        # 计算AUC值
        Auc = roc_auc_score(y_true[:,1], y_pred[:,1])
        return Acc, Sens, Spec, Auc 

def save_cnf_roc(y_true, y_pred, classes, isPlot, save_tag = ''):
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))    
    for i in range(len(y_true)): 
        y[i] = np.argmax(y_true[i,:])
        y_[i] = np.argmax(y_pred[i,:])
    cnf_mat = confusion_matrix(y, y_)
    print cnf_mat
    
    # # 记录混淆矩阵
    f = open('experiments/img/confuse_matrixes.txt', 'ab+')
    if save_tag[-1] == '0':
        f.write(save_tag+'\n')
    f.write('No.' + save_tag[-1] + '\n')
    f.write(str(cnf_mat) + '\n')
    f.close()

    # # 记录ROC曲线
    plot_roc_curve(y_true, y_pred, range(classes), 'all/'+save_tag)  

###########################
# 计算TP、TN、FP、FN
def cnf2TpTnFpFn(c, mat):

    #将正类预测为正类
    TP = mat[c][c]
    #将负类预测为负类
    TN = sum([mat[x][y]  for x in range(len(mat)) if  x != c for y in range(len(mat[0])) if y != c])
    #将负类预测为正类
    FP = sum([mat[x][y]  for x in range(len(mat)) if  x != c for y in range(len(mat[0])) if y == c])
    #将正类预测为负类
    FN = sum([mat[x][y]  for x in range(len(mat)) if  x == c for y in range(len(mat[0])) if y != c])
    return TP, TN, FP, FN        
    
def oneVsAll(y_true, y_pred, classes, save_tag = ''):
    # 计算混淆矩阵
    y = np.zeros(len(y_true))
    y_ = np.zeros(len(y_true))    
    for i in range(len(y_true)): 
        y[i] = np.argmax(y_true[i,:])
        y_[i] = np.argmax(y_pred[i,:])
    cnf_mat = confusion_matrix(y, y_)
    print cnf_mat
    
    if classes > 2: 
        # 计算多分类一对多Acc、Sens、Spec、AUC
        acc = []
        sens = []
        spec = []
        _auc = []
        
        f = open('experiments/img/oneVsAll.txt', 'ab+')
        if save_tag[-1] == '0':
            f.write(save_tag+'\n')
        f.write('No.' + save_tag[-1] + '\n')
        f.close()
        for i in range(classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
            fpr[0], tpr[0] = 0, 0
            fpr[-1], tpr[-1] = 1, 1
            _auc.append(auc(fpr, tpr))
            
            tp, tn, fp, fn = cnf2TpTnFpFn(i, cnf_mat)
            acc.append(1.0 * (tp + tn) / (tp + tn + fp + fn))
            sens.append(1.0 * (tp) / (tp + fn))
            spec.append(1.0 * (tn) / (tn + fp))
            
        # # 记录oneVsall结果          
        f = open('experiments/img/oneVsAll.txt', 'ab+')
        f.write('    Acc:' + str(acc) + '\n')
        f.write('    Sens:' + str(sens) + '\n')
        f.write('    Spec:' + str(spec) + '\n')
        f.write('    AUC:' + str(_auc) + '\n')
        f.close()
        return acc, sens, spec, _auc

        
# plot figure use t-SNE
cuda = torch.cuda.is_available()
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch.autograd import Variable

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def plot_embeddings(embeddings, targets, W=None, b=None, classes=[], save_tag = 'train', xlim=None, ylim=None):
    plt.figure(figsize=(10,10))
    for i in range(len(classes)):
        inds = np.where(targets==i)[0]
        plt.scatter(embeddings[inds,0], embeddings[inds,1], alpha=0.5, color=colors[i])
        if W is not None:
            x = np.linspace(-3,3)
            if b is None:
                y = -1.0*(W[i][0]*x)/W[i][1]
            else:
                y = -1.0*(W[i][0]*x+b[i])/W[i][1]
            plt.plot(y, x, alpha=0.7, color=colors[i])
            plt.scatter(0, 0, color='red')
    if xlim:
        plt.xlim(xlim[0], xlim[1])
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    plt.legend(classes)
    if b is None:
        save_tag = save_tag + 'withoutB'
    else:
        save_tag = save_tag + 'withB'
    plt.savefig(save_tag+'.jpg')
    plt.close('all')


def extract_embeddings(dataloader, model):
    model.eval()
    embeddings = np.zeros((len(dataloader.dataset), 2))
    labels = np.zeros(len(dataloader.dataset))
    k = 0
    for images, target in dataloader:
        images = Variable(images, volatile=True)
        if cuda:
            images = images.cuda()
        embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()[:, 0:2]
        labels[k:k+len(images)] = target.numpy()
        k += len(images)
    return embeddings, labels
    
    
import code
def printData(data, type='', only_shape=False):
    print '*'*20 + type + '*'*20
    if type == 'normal' or type == 'softmax':
        for d, label in data:
            print d.shape, label
    elif type == 'siamese':
        for d, label, obj_l in data:
            if only_shape:
                print d[0].shape, d[1].shape, label, [obj_l[0], obj_l[1]]
            else:
                print d[0].shape, d[1].shape, torch.stack([label, obj_l[0], obj_l[1]], dim=1)
    elif type == 'triplet':
        for d, label, obj_l in data:
            print d[0].shape, d[1].shape, d[2].shape, label.shape, obj_l[0].shape, obj_l[1].shape, obj_l[2].shape
    
    print len(data)
    raw_input() #等待输入，用于暂停程序，回车继续运行
    
    # 控制台交互，在命令行打印想要输出的结果，ctrl+D继续、exit()退出
    # code.interact(local=locals())

def view_img(kernel, (k_w, k_h, k_s), (row, col), tag):
    margin = 1
    width = k_w * col + (col - 1) * margin
    height = k_h * row + (row - 1) * margin
    stitched_filters = np.ones((width, height))
    # fill the picture with our saved filters
    for i in range(row):
        for j in range(col):
            stitched_filters[(k_w+margin)*i : (k_w+margin)*i + k_w,
                             (k_h+margin)*j : (k_h+margin)*j + k_h] = kernel[i*row+j, 0, :, :]
    # save the result to disk
    imsave('figures/'+tag+'data.png', stitched_filters)

def plot_tsic(data, targets, classes):
    # print(data.shape, targets.shape)
    time = [0, 16, 26, 60, 180, 300]
    colors = ['#13b652', '#17becf', '#e377c2', '#9467bd', '#7f7f7f']
    # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              # '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              # '#bcbd22', '#17becf']
    plt.figure(figsize=(30,15))
    for i in range(len(classes)):
        plt.subplot(231+i)
        inds = np.where(targets==i)[0]
        for j in inds:
            plt.plot(time, data[j,:], '.-', alpha=0.5, color=colors[i])
        # for j in range(data.shape[1]):
            # plt.scatter(j*np.ones(len(inds)), data[inds,j], alpha=0.3, color=colors[i])
        
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=0.8, color='r')
        plt.title(classes[i])
    plt.savefig('figures/tsic.jpg')
    
    
    if len(classes) == 5:
        plt.figure(figsize=(10,10))
        plt.subplot(211)
        for i in range(len(classes)-1):
            inds = np.where(targets==i)[0]
            plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[i], label=classes[i])
            print(len(inds))
            
        inds = np.where(targets==len(classes)-1)[0]
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=0.5, color=colors[4], label=classes[len(classes)-1])
        print(len(inds))
        
        plt.title("Time-Intensity Curve of HCC")
        plt.xlabel("DCE-MRI Series")
        plt.ylabel("Average Intensity")
        # 设置y刻度：用文字来显示刻度
        plt.xticks(time, ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])
        plt.legend(loc='best') 
        
        plt.subplot(212)
        inds = np.concatenate((np.where(targets==0)[0], np.where(targets==1)[0]), axis=0)
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[1], label='I-II')
        inds = np.concatenate((np.where(targets==2)[0], np.where(targets==3)[0]), axis=0)
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[3], label='III-IV')
        inds = np.where(targets==(len(classes)-1))[0]
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[4], label='BG')

        plt.legend(loc='best')    
        plt.savefig('figures/tsic_ed_mean.jpg')
    else:
        plt.figure(figsize=(8, 4))
        for i in range(len(classes)-1):
            inds = np.where(targets==i)[0]
            plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[i], label=classes[i])
            print(len(inds))
        inds = np.where(targets==len(classes)-1)[0]
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=0.5, color=colors[4], label=classes[len(classes)-1])
        print(len(inds))
        
        plt.title("Time-Intensity Curve of HCC")
        plt.xlabel("DCE-MRI Time Series")
        plt.ylabel("Average Intensity")
        # 设置x刻度：用文字来显示刻度
        plt.xticks(time, ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])
        plt.legend(loc='best') 
        plt.savefig('figures/tsic_who_mean.jpg')
        
        plt.figure(figsize=(8, 4))
        inds = np.where(targets==(len(classes)-1))[0]
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[4], label='BG')
        inds = np.where(targets!=(len(classes)-1))[0]
        plt.plot(time, data[inds,:].mean(0), 'o-', alpha=1, color=colors[1], label='HCC')
        plt.title("Time-Intensity Curve of HCC")
        plt.xlabel("DCE-MRI Time Series")
        plt.ylabel("Average Intensity")
        # 设置x刻度：用文字来显示刻度
        plt.xticks(time, ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'])
        plt.legend(loc='best')    
        plt.savefig('figures/tsic_mean_vs.jpg')
        
    
    plt.close('all')

    
import time
# 计时装饰器
def timer(func): 
    def inner_wrapper():
        start_time = time.time()
        func()
        stop_time = time.time()   
        print('Used time {}'.format(stop_time-start_time))  
    return inner_wrapper

@timer
def test1():
    time.sleep(1)
    print('Test the timer!')

if __name__ == '__main__':
    test1()