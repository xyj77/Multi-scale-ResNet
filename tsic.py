#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import numpy as np
from data_loader.mri_data import MRIData
from data_loader.datasets import SiameseMRI, TripletMRI
from utils.config import process_config
from utils.dirs import create_dirs
from utils.utils import get_args, timer
from utils.utils import printData, view_img, plot_tsic

import torch
cuda = torch.cuda.is_available()



S = [ # 融合方案

# ['E5', 'F5', 'G5', 'H5', 'I5']

['E', 'F', 'G', 'H', 'I', 'J'],
# ['E5', 'F5', 'G5', 'H5', 'I5', 'J5']

]

@timer
def main():
    # 获取配置文件路径
    # 运行：python tsic.py -c configs/tsic_who_config.json
    #       python tsic.py -c configs/tsic_ed_config.json
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([])
    for Fusion in S:
        print('Create the data generator.')
        train_dataset = MRIData(config, train = True)   
        test_dataset = MRIData(config, train = False)
        
        # printData(train_dataset, type='normal')
        # printData(test_dataset, type='normal')
        
        # 获取训练集和测试集
        train_data, train_label = train_dataset.getData(train=True, array=True)
        test_data, test_label = test_dataset.getData(train=False, array=True)
        print(type(train_data), type(train_label), train_data.shape, train_label.shape)
        
        # 截取E、F、G、H、I、J
        print(train_data[:, :, :, 3:9].shape)
        train_data = train_data[:, :, :, 3:9] #36*16*16*6
        test_data = test_data[:, :, :, 3:9]   #
        
        # 调整维度：样本-时间序列-宽-高
        train_data = train_data.transpose((0,3,1,2))
        test_data = test_data.transpose((0,3,1,2))
        train_data_shape = train_data.shape
        test_data_shape = test_data.shape
        print(train_data_shape, test_data_shape)
        
        # 可视化图像
        view_img(train_data, (16, 16, 3), (6, 6), str(Fusion))
        
        # 将宽-高拉长成一维
        train_data = train_data.reshape(train_data_shape[0], train_data_shape[1], -1)
        test_data = test_data.reshape(test_data_shape[0], test_data_shape[1], -1)
        print(train_data.shape, test_data.shape)
        
        # 计算每个序列平均灰度值
        train_mean, test_mean = train_data.mean(2), test_data.mean(2)
        
        # 合并训练集和测试集
        data_mean = np.concatenate((train_mean, test_mean), axis=0)
        label = np.concatenate((train_label, test_label), axis=0)
        
        # 绘制时间-强度曲线
        plot_tsic(data_mean, label, classes = config.classes_name)
    
if __name__ == "__main__": 
    main()