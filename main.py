# -*- coding: UTF-8 -*-
import os
import pickle
import argparse
import numpy as np

import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
cuda = torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from data_loader.mri_data import MRIData
from data_loader.mri_t2wi import MRIT2WI
from data_loader.datasets import SiameseMRI, TripletMRI
from data_loader.datasets import BalancedBatchSampler
from data_loader.npy_data import MRINPY
from data_loader.SplitData import splitDataSet
from data_loader.npy_data import SiameseNPY, TripletNPY

from models import resnet, resnetSpp, weights_init
from models.networks import EmbeddingNet3D, EmbeddingNet2D, ClassificationNet
from models.networks import SiameseNet, TripletNet, CompareNet
from models.losses import ContrastiveLoss, TripletLoss, OnlineContrastiveLoss, OnlineTripletLoss
from models.metrics import AccuracyMetric, RecallMetric, PrecisionMetric, F_scoreMetric
from models.metrics import SensitivityMetric, SpecificityMetric, AUCMetric

from trainers.trainer import fit
from evaluaters import softmax_eval, siamese_eval

from utils.selector import HardNegativePairSelector
from utils.selector import HardestNegativeTripletSelector 
from utils.utils import timer, printData, extract_embeddings, plot_embeddings
from utils.dirs import create_dirs
from utils.config import get_args, process_config
from utils.avg_acc import train_curve, plot_avg_acc

# 实验结果存储
EXP = './experiments/'

def main2(configFile, classes):
    # 获取配置文件路径
    # 运行：
    #   Or: 
    
    # 可视化: tensorboard --logdir=experiments/Compare/logs
    try:
        config = process_config(configFile)
        
    except:
        print("missing or invalid arguments")
        exit(0)
    create_dirs([])
    
    # 10次实验数据记录
    max_score= 0
    Acc_train, Acc_test = [], []   # 保存训练过程
    Pred, Label = [], []           # 保存预测值和标签
    Acc, Sens, Spec, Prec, Fscore, AUC = [], [], [], [], [], [] # 保存测试集结果
    
    # 实验标签
    tag = config.data_type[:2]+'_'+config.exp_name+'_'+''.join(config.Fusion)+'_'+str(config.embedding_size)
    figure_tag, result_tag, model_tag = EXP+'figures/'+tag, EXP+'results/'+tag, EXP+'fine_tuning/'+tag
    if config.isPretrain:
        figure_tag, result_tag, model_tag = figure_tag+'_pre', result_tag+'_pre', model_tag+'_pre'
    if config.withSPP:
        figure_tag, result_tag, model_tag = figure_tag+'_spp', result_tag+'_spp', model_tag+'_spp'
        
    # 重复10次实验
    for i in range(config.repeat): 
        # (1)载入数据
        # 划分数据集
        splitDataSet(os.path.join(config.data_path, config.data_type), 0.6)
        print('Create the data generator.')
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize(36),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
                # transforms.Normalize((0.1307,), (0.3081,))
            ])
        train_dataset = MRINPY(config, train = True, transform = train_transform) 
        test_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize(36),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
                # transforms.Normalize((0.1307,), (0.3081,))
            ])       
        test_dataset = MRINPY(config, train = False, transform = test_transform)
        print('*'*20, len(train_dataset), len(test_dataset))
        # printData(test_dataset, type='normal') 

        if (config.exp_name == 'siamese' or config.exp_name == 'compare') and not config.isHardMining:
            train_dataset = SiameseNPY(train_dataset) 
            test_dataset = SiameseNPY(test_dataset) 
            # printData(test_dataset, type=config.exp_name, only_shape=True) 

        elif config.exp_name == 'triplet' and not config.isHardMining:
            # Returns triplets of images
            train_dataset = TripletNPY(train_dataset) 
            test_dataset = TripletNPY(test_dataset)
                 
        # 批数据 Set up data loaders
        if (config.exp_name == 'siamese' or config.exp_name == 'triplet') and config.isHardMining:
            train_batch_sampler = BalancedBatchSampler(train_dataset, n_classes=config.classes, n_samples=8)
            test_batch_sampler = BalancedBatchSampler(test_dataset, n_classes=config.classes, n_samples=8)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_sampler)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

        # printData(test_loader, type=config.exp_name, only_shape = False)
            
        # (2)建立模型
        # # Set up the network and training parameters
        if config.isPretrain:
            #载入预训练Resnet
            print('Loading pretrained model ...')
            if config.withSPP:
                pretrain_dict = torch.load(config.model_path)
                # # Apply SPP
                scales = [1, 2]
                res_model = resnetSpp.resnet20(len(config.Fusion), config.embedding_size,
                                                scales, num_classes = 10)
                res_model.load_state_dict(pretrain_dict)
            else:
                res_model = resnet.resnet20(len(config.Fusion), config.embedding_size,
                                                num_classes = 10)
            #提取fc层中固定的参数
            features_num = res_model.linear.in_features
            #修改类别
            res_model.linear = nn.Linear(features_num, config.classes, bias=True)
            
        else:
            print('Building resnet model ...')
            if config.withSPP:
                # # Apply SPP
                scales = [1, 2, 3]
                res_model = resnetSpp.resnet20(len(config.Fusion), config.embedding_size,
                                               scales, num_classes = config.classes)
            else:
                res_model = resnet.resnet20(len(config.Fusion), num_features = config.embedding_size,
                                            num_classes = config.classes)
        print(res_model)
        
        # (3) 训练模型    
        if config.exp_name == 'softmax': 
            if cuda:
                res_model.cuda()
            loss_fn = nn.NLLLoss().cuda()
            
            if i == 0 and config.isPretrain and (config.transfer_type == 'siamese' or config.transfer_type == 'triplet'):
                figure_tag, result_tag, model_tag = figure_tag + '_' + config.transfer_type,\
                                                    result_tag + '_' + config.transfer_type,\
                                                    model_tag + '_' + config.transfer_type
                # 训练一部分
                '''
                print('#'*50)
                aaa = [x.shape for x in list(res_model.parameters())]
                print(bbb[-7:])
                raw_input()
                '''
                for para in list(res_model.parameters())[:-4]:
                    para.requires_grad=False
                optimizer = torch.optim.Adam(params=[
                                                    res_model.embedding_net.linear.weight,
                                                    res_model.embedding_net.linear.bias,
                                                    res_model.linear.weight,
                                                    res_model.linear.bias],
                                             lr=config.lr, weight_decay=1e-4)
            else:
                # 全部微调
                optimizer = torch.optim.Adam(params=res_model.parameters(), lr=config.lr, weight_decay=1e-4)
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.patience, gamma=0.1, last_epoch=-1)
            trainAcc, testAcc, trainLoss, testLoss = fit(train_loader, test_loader, res_model, loss_fn,\
                                    optimizer, scheduler, config.num_epochs, cuda, config.log_interval,\
                                    metrics=[AccuracyMetric()])
            Acc_train.append(trainAcc)  #记录训练过程
            Acc_test.append(testAcc)
            
            # (4) 评估模型
            pred, label, AccMetric = softmax_eval.eval(config, test_loader, res_model, loss_fn,\
                                                     cuda, metrics=[AccuracyMetric()])
            Pred.append(pred)      # 最终测试集结果
            Label.append(label)
            Acc.append(AccMetric[0].value())
            
            if config.classes > 2: # 多分类
                metricsList = [RecallMetric(), PrecisionMetric(), F_scoreMetric(1.0)]
                metricsList = softmax_eval.metric(label, pred, cuda, metricsList)
                Sens.append(metricsList[0].value())
                Prec.append(metricsList[1].value())
                Fscore.append(metricsList[2].value())
            else:                  # 二分类
                metricsList = [SensitivityMetric(), SpecificityMetric(), AUCMetric()] 
                metricsList = softmax_eval.metric(label, pred, cuda, metricsList)
                Sens.append(metricsList[0].value())
                Spec.append(metricsList[1].value())
                AUC.append(metricsList[2].value())
            
        elif config.exp_name == 'siamese':
            # 构造Siamese Model
            if config.isHardMining:
                loss_fn = OnlineContrastiveLoss(1.0, HardNegativePairSelector())
            else:
                res_model = SiameseNet(res_model)
                loss_fn = ContrastiveLoss()
            
            if cuda:
                res_model.cuda()
                
            optimizer = optim.Adam(res_model.parameters(), lr=config.lr)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)                
            trainAcc, testAcc, trainLoss, testLoss = fit(train_loader, test_loader, res_model, loss_fn,\
                                                         optimizer, scheduler, config.num_epochs,\
                                                         cuda, log_interval = config.log_interval)
            # (4) 评估模型
            AccMetric = siamese_eval.eval(config, test_loader, res_model, loss_fn,\
                                          cuda, metrics=[AccuracyMetric()])
            Acc.append(AccMetric[0].value())

        elif config.exp_name == 'triplet':
            # 构造Triplet Model
            if config.isHardMining:
                loss_fn = OnlineTripletLoss(1.0, HardestNegativeTripletSelector(1.0))
            else:
                res_model = TripletNet(res_model)
                loss_fn = ContrastiveLoss()
            
            if cuda:
                res_model.cuda()
   
            optimizer = optim.Adam(res_model.parameters(), lr=config.lr)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)                
            trainAcc, testAcc, trainLoss, testLoss = fit(train_loader, test_loader, res_model, loss_fn,\
                                                         optimizer, scheduler, config.num_epochs,\
                                                         cuda, log_interval = config.log_interval)
            # siamese_eval.eval(config, train_loader, test_loader, res_model, cuda)
            trainAcc, testAcc, trainLoss, testLoss = [], [], [], []
            AccMetric = [AccuracyMetric()]
                
        elif config.exp_name == 'compare':
            model = CompareNet(embedding_net, 2*config.embedding_size)
            model.apply(weights_init)
            print(model)
            if cuda:
                model.cuda()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, metrics=[AccuracyMetric()])
            siamese_eval.score(config, train_loader, test_loader, model, cuda)      

        # (5) 可视化并保存模型                
        if AccMetric[0].value() >= max_score:
            max_score = AccMetric[0].value()
            train_curve(trainAcc, trainLoss, testAcc, testLoss, figure_tag+'_acc')

            # 读取未扩容未采样的数据
            kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
            config.isSample = False #不再采样
            config.isAug = False    #不扩容
            train_dataset = MRINPY(config, train = True, transform = train_transform)   
            test_dataset = MRINPY(config, train = False, transform = test_transform)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, **kwargs)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, **kwargs)
                
            # 提取线性分类面的权重
            # linearWeights = res_model.state_dict()['linear.weight'].cpu().numpy()
            # linearBias = res_model.state_dict()['linear.bias'].cpu().numpy()
            linearWeights = None
            linearBias = None
                
            # 特征可视化
            train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
            plot_embeddings(train_embeddings_cl, train_labels_cl, linearWeights, linearBias, classes, figure_tag+'_train_SNE')
            val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
            plot_embeddings(val_embeddings_cl, val_labels_cl, linearWeights, linearBias, classes, figure_tag+'_test_SNE')
                
            # Save model dict
            torch.save(res_model.state_dict(), model_tag+'.pkl') 

    # (6) 保存结果
    # 绘制平均ACC曲线
    # plot_avg_acc(Acc_train, Acc_test, figure_tag+'_avgAcc')
    # 保存训练过程
    with open(result_tag+'_avgAcc.bin', 'wb') as fp:
        pickle.dump(Acc_train, fp) #顺序存入变量
        pickle.dump(Acc_test, fp)
    
    # 保存预测值和标签
    with open(result_tag+'_predL.bin', 'wb') as fp:
        pickle.dump(Pred, fp)  #顺序存入变量
        pickle.dump(Label, fp)
        
    # 保存均值方差到TXT文件
    if config.classes > 2:
        # 存入txt
        with open(result_tag+'.txt', 'ab') as fp:
            fp.write('Acc:%s\n'%(str(Acc))) 
            fp.write('Average Acc:%.4f  Std: +- %.4f\n\n'%(np.mean(Acc, axis=0), np.std(Acc, axis=0)))
            fp.write('Recall:%s\n'%(str(Sens)))            
            fp.write('Average Recall:%.4f  Std: +- %.4f\n\n'%(np.mean(Sens, axis=0), np.std(Sens, axis=0)))
            fp.write('Prec:%s\n'%(str(Prec)))
            fp.write('Average Prec:%.4f  Std: +- %.4f\n\n'%(np.mean(Prec, axis=0), np.std(Prec, axis=0)))
            fp.write('Fscore:%s\n'%(str(Fscore)))
            fp.write('Average Fscore:%.4f  Std: +- %.4f\n\n'%(np.mean(Fscore, axis=0), np.std(Fscore, axis=0)))
    else:
        with open(result_tag+'.txt', 'ab') as fp:
            fp.write('Acc:%s\n'%(str(Acc))) 
            fp.write('Average Acc:%.4f  Std: +- %.4f\n\n'%(np.mean(Acc, axis=0), np.std(Acc, axis=0)))
            fp.write('Sens:%s\n'%(str(Sens)))            
            fp.write('Average Sens:%.4f  Std: +- %.4f\n\n'%(np.mean(Sens, axis=0), np.std(Sens, axis=0)))
            fp.write('Spec:%s\n'%(str(Spec)))
            fp.write('Average Spec:%.4f  Std: +- %.4f\n\n'%(np.mean(Spec, axis=0), np.std(Spec, axis=0)))
            fp.write('AUC:%s\n'%(str(AUC)))
            fp.write('Average AUC:%.4f  Std: +- %.4f\n\n'%(np.mean(AUC, axis=0), np.std(AUC, axis=0)))

@timer
def main1():
    # 获取配置文件路径
    # 运行：python main.py -c configs/ed_config.json         #for softmax
    #       python main.py -c configs/ed_siamese_config.json #for siamese
    
    #   Or: python main.py -c configs/who_config.json  #for WHO
    # 可视化: tensorboard --logdir=experiments/Compare/logs
    
    parser = argparse.ArgumentParser("""Image classifical!""")
    # parser.add_argument('-c', '--config', default='configs/transfer_ed_config.json')
    # classes=('I II','III IV')
    # # classes=('I', 'II', 'III', 'IV')

    parser.add_argument('-c', '--config', default='configs/transfer_who_config.json')
    classes=('1','2', '3')
    
    try:
        args = parser.parse_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([])
    No, max_score= -1, 0
    save_tag = config.exp_name
    # 重复10次实验
    for i in range(config.repeat): 
        # (1)载入数据
        print('Create the data generator.')
        # transform = transforms.Compose([
                # transforms.RandomHorizontalFlip(),
                # transforms.RandomCrop(32, 4),
                # transforms.ToTensor(),  # 将numpy数据类型转化为Tensor
                # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
            # ])
        # Load data
        train_dataset = MRIT2WI(config, train = True)   
        test_dataset = MRIT2WI(config, train = False)
        # printData(test_dataset, type='normal') 

        if config.exp_name == 'siamese' or config.exp_name == 'compare':
            train_dataset = SiameseMRI(train_dataset) 
            test_dataset = SiameseMRI(test_dataset) 
            # printData(test_dataset, type=config.exp_name, only_shape=True) 

        elif config.exp_name == 'triplet':
            # Returns triplets of images
            train_dataset = TripletMRI(train_dataset) 
            test_dataset = TripletMRI(test_dataset)
            
        # 批数据        
        # Set up data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        # printData(test_loader, type=config.exp_name, only_shape = False)
            
        # (2)构建模型
        # Set up the network and training parameters
        if config.exp_name == 'softmax':
            if config.isPretrain:
                #载入预训练Resnet
                print('Loading pretrained model ...')
                pretrain_dict = torch.load('models/cifar_resnet_2_dict-withSPP.pkl')
                
                # res_model = resnet.resnet20(len(config.Fusion), num_features = config.embedding_size,
                                            # num_classes = 10)
                # # Apply SPP
                scales = [1, 2]
                res_model = resnetSpp.resnet20(len(config.Fusion), config.embedding_size,
                                            scales, num_classes = 10)
                res_model.load_state_dict(pretrain_dict)
                print(res_model)

                #提取fc层中固定的参数
                features_num = res_model.linear.in_features
                #修改类别
                res_model.linear = nn.Linear(features_num, config.classes, bias=False)
            else:
                print('Building resnet model ...')
                res_model = resnet.resnet20(len(config.Fusion), num_features = config.embedding_size,
                                            num_classes = config.classes)
            # print(res_model)
                
            if cuda:
                res_model.cuda()
            loss_fn = nn.NLLLoss().cuda()
            optimizer = torch.optim.Adam(res_model.parameters(), lr=config.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.patience, gamma=0.1, last_epoch=-1)
            
            fit(train_loader, test_loader, res_model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, metrics=[AccuracyMetric()])
            
        elif config.exp_name == 'siamese':
            # Set up the network and training parameters
            res_model = resnet.resnet20(len(config.Fusion), num_features = config.embedding_size, num_classes = config.classes)
            print(res_model)
            model = SiameseNet(res_model)

            model.cuda()
            loss_fn = ContrastiveLoss(config.margin).cuda()
                    
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)                
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, obj_label=True)
        
        elif config.exp_name == 'triplet':
            model = TripletNet(embedding_net)
            model.apply(weights_init)
            print(model)
            if cuda:
                model.cuda()
            loss_fn = TripletLoss(config.margin)
                    
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)                
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, obj_label=True)
                
        elif config.exp_name == 'compare':
            model = CompareNet(embedding_net, 2*config.embedding_size)
            model.apply(weights_init)
            print(model)
            if cuda:
                model.cuda()
            if cuda:
                model.cuda()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, metrics=[AccuracyMetric()], obj_label=True)
        
        # 读取未扩容未采样的数据
        # Set up data loaders
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        config.isSample = False #不再采样
        config.isAug = True    #不扩容
        train_dataset = MRIT2WI(config, train = True)   
        test_dataset = MRIT2WI(config, train = False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, **kwargs)
        
        # 提取线性分类面的权重
        linearWeights = res_model.state_dict()['linear.weight'].cpu().numpy()
        # linearBias = res_model.state_dict()['linear.bias'].cpu().numpy()
        linearBias = None
        print(linearWeights, linearBias)
        
        # 特征可视化
        train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, res_model)
        plot_embeddings(train_embeddings_cl, train_labels_cl, linearWeights, linearBias, classes=classes, save_tag='train')
        val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, res_model)
        plot_embeddings(val_embeddings_cl, val_labels_cl, linearWeights, linearBias, classes=classes, save_tag='test')
            
        # Save model dict
        torch.save(res_model.state_dict(), 'fine_tuning/resnet20_2.pkl')     
          
        # (3)评估siamese模型
        model_dict = torch.load('fine_tuning/resnet20_2.pkl')
        # model = resnet.resnet20(len(config.Fusion), num_features = config.embedding_size,
                                    # num_classes = config.classes)
        # # Apply SPP
        scales = [1, 2]
        model = resnetSpp.resnet20(len(config.Fusion), config.embedding_size,
                                       scales, num_classes = config.classes)
        if cuda:
            model.cuda()
        model.load_state_dict(model_dict)
        if config.exp_name == 'compare':
            siamese_eval.score(config, train_loader, test_loader, model, cuda)
        else:
            siamese_eval.eval(config, train_loader, test_loader, model, cuda)

@timer
def main():
    # 获取配置文件路径
    # 运行：python main.py -c configs/ed_config.json         #for softmax
    #       python main.py -c configs/ed_siamese_config.json #for siamese
    
    #   Or: python main.py -c configs/who_config.json  #for WHO
    # 可视化: tensorboard --logdir=experiments/Compare/logs
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    create_dirs([])
    No, max_score= -1, 0
    save_tag = config.exp_name
    # 重复10次实验
    for i in range(config.repeat): 
        # (1)载入数据
        print('Create the data generator.')
        train_dataset = MRIData(config, train = True)   
        test_dataset = MRIData(config, train = False)

        # printData(test_dataset, type='normal') 
        '''
        data_num
        (5L, 5L, 32L, 32L) 0.0
        (5L, 5L, 32L, 32L) 2.0
                ...
        (5L, 5L, 32L, 32L) 1.0
        (5L, 5L, 32L, 32L) 3.0
                ...
        '''   
        if config.exp_name == 'siamese' or config.exp_name == 'compare':
            # Set up data loaders
            # Returns pairs of images and target same/different
            # if config.isSelect and False:
                # from data_loader.datasets import BalancedBatchSampler
                # train_batch_dataset = BalancedBatchSampler(train_dataset, n_classes=config.classes, n_samples=16)
                # test_batch_dataset = BalancedBatchSampler(test_dataset, n_classes=config.classes, n_samples=16)
            # else:
            train_dataset = SiameseMRI(train_dataset) 
            test_dataset = SiameseMRI(test_dataset)
                
            # printData(test_dataset, type=config.exp_name, only_shape=True) 
            '''
            data_num
            (5L, 5L, 32L, 32L) (5L, 5L, 32L, 32L) 1 [2.0, 2.0]
            (5L, 5L, 32L, 32L) (5L, 5L, 32L, 32L) 1 [0.0, 0.0]
                              ...
            (5L, 5L, 32L, 32L) (5L, 5L, 32L, 32L) 0 [3.0, 0.0]
            (5L, 5L, 32L, 32L) (5L, 5L, 32L, 32L) 0 [1.0, 0.0]
                              ...
            '''
        elif config.exp_name == 'triplet':
            # Set up data loaders
            # Returns triplets of images
            train_dataset = TripletMRI(train_dataset) 
            test_dataset = TripletMRI(test_dataset)
            
            
        # 批数据        
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {} 
        # if config.exp_name == 'siamese' and config.isSelect and False:
            # train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_dataset, **kwargs)
            # test_loader = torch.utils.data.DataLoader(test_dataset, batch_sampler=test_batch_dataset, **kwargs)
        # else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
            
        # printData(test_loader, type=config.exp_name, only_shape = False)
        '''
        for siamese:
        data_num/batch_size
        (16L, 5L, 5L, 32L, 32L) (16L, 5L, 5L, 32L, 32L) (16L,) [(16L,), (16L,)]
                                ...
        (8L, 5L, 5L, 32L, 32L) (8L, 5L, 5L, 32L, 32L) (16L,) [(8L,), (8L,)]
        '''
            
        # (2)构建模型
        # # Set up the network and training parameters
        embedding_net = EmbeddingNet3D(len(config.Fusion), config.embedding_size)
        if config.exp_name == 'softmax':
            model = ClassificationNet(embedding_net, n_classes=config.classes)
            model.apply(weights_init)
            print(model)
                
            if cuda:
                model.cuda()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, metrics=[AccuracyMetric()])
        elif config.exp_name == 'siamese':
            # if config.isSelect and False:
                # model = embedding_net
                # print(model)
                # if cuda:
                    # model.cuda()
                # from models.losses import OnlineContrastiveLoss
                # # Strategies for selecting pairs within a minibatch
                # from utils.selector import AllPositivePairSelector, HardNegativePairSelector 
                # loss_fn = OnlineContrastiveLoss(margin, HardNegativePairSelector())
            # else:
            model = SiameseNet(embedding_net)
            model.apply(weights_init)
            print(model)
            if cuda:
                model.cuda()
            loss_fn = ContrastiveLoss(config.margin)
                    
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)                
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, obj_label=True)
        
        elif config.exp_name == 'triplet':
            model = TripletNet(embedding_net)
            model.apply(weights_init)
            print(model)
            if cuda:
                model.cuda()
            loss_fn = TripletLoss(config.margin)
                    
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)                
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, obj_label=True)
                
        elif config.exp_name == 'compare':
            model = CompareNet(embedding_net, 2*config.embedding_size)
            model.apply(weights_init)
            print(model)
            if cuda:
                model.cuda()
            if cuda:
                model.cuda()
            loss_fn = torch.nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-2)
            scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
            fit(train_loader, test_loader, model, loss_fn, optimizer, scheduler, config.num_epochs,\
                cuda, log_interval = config.log_interval, metrics=[AccuracyMetric()], obj_label=True)
        
        
        # 读取未扩容未采样的数据
        # Set up data loaders
        kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
        config.isSample = False #不再采样
        config.isAug = False    #不扩容
        train_dataset = MRIData(config, train = True)   
        test_dataset = MRIData(config, train = False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False, **kwargs)
        
        # 特征可视化
        train_embeddings_cl, train_labels_cl = extract_embeddings(train_loader, model)
        plot_embeddings(train_embeddings_cl, train_labels_cl, save_tag = 'train', n_classes=config.classes)
        val_embeddings_cl, val_labels_cl = extract_embeddings(test_loader, model)
        plot_embeddings(val_embeddings_cl, val_labels_cl, save_tag = 'test', n_classes=config.classes)
            
        # 保存整个模型
        torch.save(model, 'models/model.pkl')       
          
        # (3)评估siamese模型
        model = torch.load('models/model.pkl') #加载模型
        if config.exp_name == 'compare':
            siamese_eval.score(config, train_loader, test_loader, model, cuda)
        else:
            siamese_eval.eval(config, train_loader, test_loader, model, cuda)

if __name__ == '__main__':
    # # 使用mat 3D数据
    # main()
    # # 使用mat 2D数据
    # main1()
    
    
    # 使用npy数据
    classes=('BG', 'HCC')
    
    # 单模态
    # main2('configs/Bi/A_config.json', classes)
    # main2('configs/Bi/B_config.json', classes)
    # main2('configs/Bi/K_config.json', classes)
    # main2('configs/Bi/E_config.json', classes)
    # main2('configs/Bi/F_config.json', classes)
    # main2('configs/Bi/G_config.json', classes)
    # main2('configs/Bi/H_config.json', classes)
    # main2('configs/Bi/I_config.json', classes)
    # main2('configs/Bi/J_config.json', classes)
    
    # main2('configs/Bi/EGI_config.json', classes)
    # main2('configs/Bi/AFI_config.json', classes)
    # main2('configs/Bi/ABK_config.json', classes)
    # main2('configs/Bi/AEG_config.json', classes)
    # # main2('configs/Bi/All_config.json', classes)#batch_size设为1
    
    
    # # BG和HCC分类
    # main2('configs/Bi/cifar_transfer_pre_spp_config.json', classes)
    # main2('configs/Bi/cifar_transfer_spp_config.json', classes)
    # main2('configs/Bi/cifar_transfer_pre_config.json', classes)
    # main2('configs/Bi/cifar_transfer_config.json', classes)
    
    
    
    # # WHO三分类
    classes=('poorly', 'moderately', 'well')
    
    # # 单模态
    # main2('configs/WH/A_config.json', classes)
    # main2('configs/WH/B_config.json', classes)
    # main2('configs/WH/K_config.json', classes)
    # main2('configs/WH/E_config.json', classes)
    # main2('configs/WH/F_config.json', classes)
    # main2('configs/WH/G_config.json', classes)
    # main2('configs/WH/H_config.json', classes)
    # main2('configs/WH/I_config.json', classes)
    # main2('configs/WH/J_config.json', classes)
    
    # main2('configs/WH/mnist_transfer_config.json', classes)
    # main2('configs/WH/mnist_transfer_spp_config.json', classes)
    
    # main2('configs/WH/mnist_transfer_siamese_config.json', classes)
    # main2('configs/WH/mnist_transfer_triplet_config.json', classes)
    # main2('configs/WH/mnist_transfer_softmax_siamese_config.json', classes)
    # main2('configs/WH/mnist_transfer_softmax_triplet_config.json', classes)
    
    # 多模态
    # 融合
    # main2('configs/WH/All_spp_config.json', classes) #无法转化成图片
    # main2('configs/WH/ABK_spp_config.json', classes)
    # main2('configs/WH/AGI_spp_config.json', classes)
    # main2('configs/WH/EGI_spp_config.json', classes)
    # main2('configs/WH/ABI_spp_config.json', classes)
    
    main2('configs/WH/ABK_pre_spp_config.json', classes)
    main2('configs/WH/AGI_pre_spp_config.json', classes)
    main2('configs/WH/EGI_pre_spp_config.json', classes)
    main2('configs/WH/ABI_pre_spp_config.json', classes)
    
    
    # 四组
    # main2('configs/WH/cifar_transfer_pre_spp_config.json', classes) #84%
    # main2('configs/WH/cifar_transfer_spp_config.json', classes)
    # main2('configs/WH/cifar_transfer_pre_config.json', classes)
    # main2('configs/WH/cifar_transfer_config.json', classes)
    
    
    # main2('configs/WH/FHI_siamese_config.json', classes)
    # main2('configs/WH/FHI_triplet_config.json', classes)
    

    # # 四分类
    # classes=('I II', 'III IV')
    # # classes=('I', 'II', 'III', 'IV')

