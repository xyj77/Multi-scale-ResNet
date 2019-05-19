# -*- coding:utf-8 -*-
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import scipy
import os
        
def eval(config, test_loader, model, loss_fn, cuda, metrics=[]):
    """
    arg:
        
    """
    pred, label = [], []
    for metric in metrics:
        metric.reset()
    
    model.eval()
    for batch_idx, (data, target) in enumerate(test_loader):
        label.extend(target)
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()
        data = tuple(Variable(d, volatile=True) for d in data)
        outputs = model(*data)
        pred.extend(outputs.data.cpu().numpy())

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

    return pred, label, metrics
    
def metric(target, pred, cuda, metrics=[]):
    """
    arg:
        
    """
    loss = None
    
    for metric in metrics:
        metric.reset()
    
    target, pred = np.array(target), np.array(pred)
    target, pred = torch.from_numpy(target), torch.from_numpy(pred)
    
    if cuda and target is not None:
        target = target.cuda()
    if cuda and pred is not None:
        pred = pred.cuda()

    if target is not None:
        target = Variable(target, volatile=True)
        target = (target,)
    if pred is not None:
        pred = Variable(pred, volatile=True)
        pred = (pred,)

    for metric in metrics:
        metric(pred, target, loss)
        print(metric.name(), metric.value())

    return metrics