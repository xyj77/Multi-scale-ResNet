#-*- coding:utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt

from avg_acc import avgAcc
from avg_roc import avgRoc

color = ['red', 'blue', 'green', 'fuchsia', 'cyan']
shadow = ['violet', 'lightsteelblue', 'lightgreen', 'pink' ,'aquamarine']

def plotAcc(path, fusion, ylim):
    result = path + fusion +'_2_avgAcc.bin'
    resultSpp = path + fusion +'_2_spp_avgAcc.bin'
    resultPre = path + fusion +'_2_pre_avgAcc.bin'
    resultPreSpp = path + fusion +'_2_pre_spp_avgAcc.bin'

    with open(result, 'rb') as fp:
        resT = pickle.load(fp)
        res = pickle.load(fp)
    with open(resultPre, 'rb') as fp:
        resPreT = pickle.load(fp)
        resPre = pickle.load(fp)
    with open(resultSpp, 'rb') as fp:
        resSppT = pickle.load(fp)
        resSpp = pickle.load(fp)
    with open(resultPreSpp, 'rb') as fp:
        resPreSppT = pickle.load(fp)
        resPreSpp = pickle.load(fp)
        
    name = ['Fine-tuned Resnet+SPP', 'Fine-tuned Resnet', 'Resnet+SPP', 'Resnet']
    avgAcc([resPreSpp, resPre, resSpp, res], name, path+fusion+'_test', ylim)  
    avgAcc([resPreSppT, resPreT, resSppT, resT], name, path+fusion+'_train', ylim)

    
    '''
    # 'wb'和'ab'区别，ab.bin的大小是wb.bin的10倍
    for _ in range(10):
        with open('../figures/wb.bin', 'wb') as fp:
            pickle.dump(resPre, fp) #覆盖

        with open('../figures/ab.bin', 'ab') as fp:
            pickle.dump(resPre, fp) #顺序存入变量
    '''
  
def plotRoc(path, fusion):
    result = path + fusion +'_2_predL.bin'
    resultSpp = path + fusion +'_2_spp_predL.bin'
    resultPre = path + fusion +'_2_pre_predL.bin'
    resultPreSpp = path + fusion +'_2_pre_spp_predL.bin'

    with open(result, 'rb') as fp:
        resP = pickle.load(fp)
        resL = pickle.load(fp)
    with open(resultPre, 'rb') as fp:
        resPreP = pickle.load(fp)
        resPreL = pickle.load(fp)
    with open(resultSpp, 'rb') as fp:
        resSppP = pickle.load(fp)
        resSppL = pickle.load(fp)
    with open(resultPreSpp, 'rb') as fp:
        resPreSppP = pickle.load(fp)
        resPreSppL = pickle.load(fp)
        
    name = ['Fine-tuned+SPP', 'Fine-tuned', 'Resnet+SPP', 'Resnet']
    avgRoc([resPreSppP, resPreP, resSppP, resP],\
           [resPreSppL, resPreL, resSppL, resL], name, path+fusion+'_ROC')  
  
if __name__ == "__main__":
    '''
    BG HCC
    '''
    # # fusion = 'AKG'
    # fusion = 'AEG'
    # path = '../experiments/results/Bi_softmax_'
    # ylim = [0.5, 1.0]
    
    # # 绘制平均Acc
    # plotAcc(path, fusion, ylim)
    
    # # 绘制平均Roc
    # plotRoc(path, fusion)

    
    '''
    HCC
    '''
    # # fusion = 'AKG'
    # fusion = 'AEG'
    fusion = 'E'
    path = '../experiments/results/WH_softmax_'
    ylim = [0.5, 1.0]
    
    # 绘制平均Acc
    plotAcc(path, fusion, ylim)

    
    
    