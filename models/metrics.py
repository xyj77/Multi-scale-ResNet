#-*- coding:utf-8 -*-
import numpy as np
from sklearn.metrics import recall_score, precision_score, roc_auc_score, fbeta_score
from sklearn.metrics import confusion_matrix


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class AccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / (self.total + 1e-8)

    def name(self):
        return 'Accuracy'

class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'

class SensitivityMetric(Metric):
    """
    Works with classification model
    Same with RecallMetric
    Sensitivity&Recall
    敏感度 or 召回率
    适用于二分类
    """

    def __init__(self):
        self.val = 0
        self.batches = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        label = target[0].data.view_as(pred)
        self.val += recall_score(label, pred)
        self.batches += 1.0
        return self.value()

    def reset(self):
        self.val = 0
        self.batches = 0

    def value(self):
        return 100 * float(self.val) / self.batches

    def name(self):
        return 'Sensitivity'

class SpecificityMetric(Metric):
    """
    Works with classification model
    Specificity
    特异度
    
    计算方法：
    #cnf_mat是混淆矩阵
    cnf_mat = confusion_matrix(y_true, y_pred)
    Acc = 1.0*(cnf_mat[1][1]+cnf_mat[0][0])/len(y_true)
    Sens = 1.0*cnf_mat[1][1]/(cnf_mat[1][1]+cnf_mat[1][0])
    Spec = 1.0*cnf_mat[0][0]/(cnf_mat[0][0]+cnf_mat[0][1])
    # 计算AUC值
    Auc = roc_auc_score(y_true, y_pred)
    
    适用于二分类
    """

    def __init__(self):
        self.val = 0
        self.batches = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        label = target[0].data.view_as(pred)
        cnf_mat = confusion_matrix(label, pred)
        self.val += 1.0*cnf_mat[0][0]/(cnf_mat[0][0]+cnf_mat[0][1])
        self.batches += 1.0
        return self.value()

    def reset(self):
        self.val = 0
        self.batches = 0

    def value(self):
        return 100 * float(self.val) / self.batches

    def name(self):
        return 'Sensitivity'

class AUCMetric(Metric):
    """
    Works with classification model
    AUC: Area Under ROC
    
    适用于二分类
    """

    def __init__(self):
        self.val = 0
        self.batches = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        label = target[0].data.view_as(pred)
        self.val += roc_auc_score(label, pred)
        self.batches += 1
        return self.value()

    def reset(self):
        self.val = 0
        self.batches = 0

    def value(self):
        return 100 * float(self.val) / self.batches

    def name(self):
        return 'AUC'

class RecallMetric(Metric):
    """
    Works with classification model
    Same with Sensitivity
    Sensitivity&Recall
    敏感度 or 召回率
    """

    def __init__(self):
        self.val = 0
        self.batches = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        label = target[0].data.view_as(pred)
        self.val += recall_score(label, pred, average='macro', labels=np.unique(label))
        self.batches += 1.0
        return self.value()

    def reset(self):
        self.val = 0
        self.batches = 0

    def value(self):
        return 100 * float(self.val) / self.batches

    def name(self):
        return 'Recall'
        
class PrecisionMetric(Metric):
    """
    Works with classification model
    Precision
    精确度
    """

    def __init__(self):
        self.val = 0
        self.batches = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        label = target[0].data.view_as(pred)
        self.val += precision_score(label, pred, average='macro', labels=np.unique(pred))
        self.batches += 1
        return self.value()

    def reset(self):
        self.val = 0
        self.batches = 0

    def value(self):
        return 100 * float(self.val) / self.batches

    def name(self):
        return 'Precision'
        
class F_scoreMetric(Metric):
    """
    Works with classification model
    F_score
    beta值越小，表示越看中precision
    beta值越大，表示越看中recall
    """

    def __init__(self, beta=1):
        self.beta = beta
        self.val = 0
        self.batches = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        label = target[0].data.view_as(pred)
        self.val += fbeta_score(label, pred, beta=self.beta, average='macro')
        self.batches += 1
        return self.value()

    def reset(self):
        self.val = 0
        self.batches = 0

    def value(self):
        return 100 * float(self.val) / self.batches

    def name(self):
        return 'F_'+str(self.beta)