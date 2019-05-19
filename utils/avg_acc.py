#-*- coding:utf-8 -*-
import pickle
import numpy as np
import matplotlib.pyplot as plt

color = ['red', 'blue', 'green', 'fuchsia', 'cyan']
shadow = ['violet', 'lightsteelblue', 'lightgreen', 'pink' ,'aquamarine']

def plot_avg_acc(AccsT, Accs, save_tag):
    epoch = range(len(AccsT[0]))
    plt.figure(figsize=(10,10))
    
    plt.subplot(211)   #训练曲线
    # for acc in AccsT:
        # plt.plot(epoch, acc, lw=0.5, alpha=0.3)

    mean_acc = np.mean(AccsT, axis=0)
    plt.plot(epoch, mean_acc, color='red', label=r'Mean acc',lw=1.2, alpha=.8)
    
    std_acc = np.std(AccsT, axis=0)
    upper = np.minimum(mean_acc + std_acc, 1)
    lower = np.maximum(mean_acc - std_acc, 0)
    plt.fill_between(epoch, lower, upper, color='gray', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, len(epoch)-1])  
    plt.ylim([0.5, 1]) 
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training Average accuracy', size=12)
    plt.legend(loc="lower right")
    
    plt.subplot(212)   #测试曲线
    # for acc in Accs:
        # plt.plot(epoch, acc, lw=0.5, alpha=0.3)
    mean_acc = np.mean(Accs, axis=0)
    plt.plot(epoch, mean_acc, color='blue', label=r'Mean acc',lw=1.2, alpha=.8)

    std_acc = np.std(Accs, axis=0)
    upper = np.minimum(mean_acc + std_acc, 1)
    lower = np.maximum(mean_acc - std_acc, 0)
    plt.fill_between(epoch, lower, upper, color='gray', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, len(epoch)-1])  
    plt.ylim([0.5, 1]) 
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Testing Average accuracy', size=12)
    plt.legend(loc="lower right")
    
    
    plt.savefig(save_tag + '.png')
    plt.close('all') # 关闭图 
        
def train_curve(acc, loss, val_acc, val_loss, save_tag):
    # # 绘图
    plt.figure(figsize=(15, 15))
    epoch = len(acc)
    if epoch > 0:
        plt.subplot(211)
        plt.plot(range(epoch), acc, label='Train')
        plt.plot(range(epoch), val_acc, label='Test')
        plt.title('Accuracy', size=15)
        plt.legend(loc = 'lower right')
        plt.grid(True)
    epoch = len(loss)
    if epoch > 0:
        plt.subplot(212)
        plt.plot(range(epoch), loss, label='Train')
        plt.plot(range(epoch), val_loss, label='Test')
        plt.title('Loss', size=15)
        plt.legend(loc = 'upper right')
        plt.grid(True)
    
    plt.savefig(save_tag + '.png')
    # plt.show()
    plt.close('all') # 关闭图
    
def avgAcc(accs, labels=[], save_tag='', ylim=[0.5, 1.0]):
    epoch = range(1, len(accs[0][0])+1)
    for i, acc in enumerate(accs):
        mean_acc = np.mean(acc, axis=0)
        plt.plot(epoch, mean_acc, color=color[i], label=labels[i], lw=1.2, alpha=.8)

        std_acc = np.std(acc, axis=0)
        upper = np.minimum(mean_acc + std_acc, 1)
        lower = np.maximum(mean_acc - std_acc, 0)
        plt.fill_between(epoch, lower, upper, color=shadow[i], alpha=.2,
                         label=labels[i]+r' std.')
    
    plt.xlim([1, len(epoch)])  
    plt.ylim(ylim) 
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Average accuracy', size=12)
    plt.legend(loc="lower right")
    plt.savefig(save_tag + '_avgAcc.png')
    plt.close('all') # 关闭图 
    print('Avg Acc curve saved !')
