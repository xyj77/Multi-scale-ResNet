# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.utils import printData
import numpy as np
import pandas as pd
import scipy
import os

def eval(config, data_loader, model, loss_fn, cuda, metrics=[]):

    for batch_idx, (data, target) in enumerate(data_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        data = tuple(Variable(d, volatile=True) for d in data)

        outputs = model(*data)
        
        if type(outputs) not in (tuple, list):
            outputs = (outputs,)
        loss_inputs = outputs
        if target is not None:
            target = Variable(target, volatile=True)
            target = (target,)
            loss_inputs += target

        loss_outputs = loss_fn(*loss_inputs)
        for metric in metrics:
            metric(outputs, target, loss_outputs)
    
    for metric in metrics:
        print(metric.name(), metric.value())
        
    return metrics
        
        
# def sigmoid(inX):
	# return 1.0 / (1 + np.exp(-inX))        

# def embedding(data_loader, model, cuda, train=True):
    # '''
    
    # '''
    # for batch_idx, (data, target) in enumerate(data_loader):
        # target = target if len(target) > 0 else None
        # if not type(data) in (tuple, list):
            # data = (data,)
        # if cuda:
            # data = tuple(d.cuda() for d in data)
            # if target is not None:
                # target = target.cuda()
        # data = tuple(Variable(d, volatile=True) for d in data)

        # outputs = model.get_embedding(*data)
        
        # # print('Feature shape: ', outputs.shape)
        # # print(target.cpu().numpy())
        # if train:
            # img = os.path.join('./figures', 'features_train.jpg')
            # txt = os.path.join('./figures', 'features_train.csv')
        # else:
            # img = os.path.join('./figures', 'features_test.jpg')
            # txt = os.path.join('./figures', 'features_test.csv')
        
        # data_array = outputs.data.cpu().numpy()
        # target_array = np.expand_dims(target.cpu().numpy(), 1)   #扩充标签的维度才能合并
        # map = np.concatenate((data_array, target_array), axis=1) #拼接标签和特征图
        
        # scipy.misc.imsave(img, sigmoid(map))   #保存图像
        # pd_data = pd.DataFrame(map, columns=[str(i) for i in range(map.shape[1])])
        # pd_data.to_csv(txt, index=False)       #保存csv
        
        # return outputs, target
        
# def eval(config, train_loader, test_loader, model, cuda, metrics=[]):
    # """
    # arg:
        
    # """
    # for metric in metrics:
        # metric.reset()
    # model.eval()
    
    # # 所有数据
    # # printData(train_loader, type='normal', only_shape=True)
    
    # train_feature, train_label = embedding(train_loader, model, cuda, train=True)
    # test_feature, test_label = embedding(test_loader, model, cuda, train=False)
    
    # # 矩阵相乘
    # # score_matrix = train_feature.mm(test_feature.t())
    # # print(score_matrix)
    
    # # score_matrix = torch.Tensor(test_feature.size(0), train_feature.size(0))
    # score_matrix = np.zeros((test_feature.size(0), train_feature.size(0)))
    # for i in range(test_feature.size(0)):
        # anchor = test_feature[i]
        # for j in range(train_feature.size(0)):
            # diff = (test_feature[i] - train_feature[j]).data.cpu().numpy()
            # score_matrix[i][j] = np.sum(pow(diff, 2))  #MSE
        
    # print(score_matrix.shape)
    # scipy.misc.imsave('./figures/pred_score.jpg', sigmoid(score_matrix))   #保存图像
    
    # count = 0
    # for i in range(test_feature.size(0)):
        # similarity = score_matrix[i]
        # score, pred = 1, -1
        # for c in range(config.classes):
            # mask = torch.eq(train_label,c)
            # s = 1.0*sum(similarity*mask.cpu().numpy())/sum(mask)
            # if s < score:
                # score, pred = s, c

        # if(pred == test_label[i]):
            # count = count + 1
    
    # print(1.0*count/len(test_label))
    
    # return 1.0*count/len(test_label)
    
# def score(config, train_loader, test_loader, model, cuda, metrics=[]):
    
    # train_data, train_label, test_data, test_label = None, None, None, None
    # for (x, y) in train_loader:
        # train_data, train_label = x, y
    # for (x, y) in test_loader:
        # test_data, test_label = x, y
    # score_matrix = np.zeros((test_data.size(0), train_data.size(0)))
    # for i in range(test_data.size(0)):
        # for j in range(train_data.size(0)):
            # data = tuple([Variable(test_data[i].unsqueeze(0).cuda(), volatile=True),
                          # Variable(train_data[j].unsqueeze(0).cuda(), volatile=True)])
            # score_matrix[i][j] = F.sigmoid(model(*data)).data.cpu().numpy()[0][1]
            # # print(model(*data).data.cpu().numpy())
    
    # print(score_matrix.shape)
    # count = 0
    # for i in range(test_data.size(0)):
        # similarity = score_matrix[i]
        # score, pred = 1, -1
        # for c in range(config.classes):
            # mask = torch.eq(train_label,c)
            # s = 1.0*sum(similarity*mask.cpu().numpy())/sum(mask)
            # if s < score:
                # score, pred = s, c

        # if(pred == test_label[i]):
            # count = count + 1
    
    # print(1.0*count/len(test_label))
    # return 1.0*count/len(test_label)