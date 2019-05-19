#-*- coding:utf-8 -*-
from torch.autograd import Variable
import numpy as np
import torch

def fit(train_loader, val_loader, model, loss_fn, optimizer, scheduler, n_epochs, cuda,\
        log_interval, metrics=[], start_epoch=0, obj_label=False):
    """
    Loaders, model, loss function and metrics should work together for a given task,
    i.e. The model should be able to process data output of loaders,
    loss function should process target output of loaders and outputs from the model

    Examples: Classification: batch loader, classification model, NLL loss, accuracy metric
    Siamese network: Siamese loader, siamese model, contrastive loss
    Online triplet learning: batch loader, embedding model, online triplet loss
    """
    trainLoss, testLoss = [], []
    trainRes, testRes = [], []
    
    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, obj_label=obj_label)

        message = 'Epoch: {}/{}. Train set: Average loss: {:.4f}'.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            trainRes.append(metric.value()/100.0) #依次放入metric结果
            
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics, obj_label=obj_label)
        val_loss /= len(val_loader)

        message += '\nEpoch: {}/{}. Validation set: Average loss: {:.4f}'.format(epoch + 1, n_epochs,
                                                                                 val_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            testRes.append(metric.value()/100.0)
        print(message)
        
        trainLoss.append(train_loss)
        testLoss.append(val_loss)
        
    return trainRes, testRes, trainLoss, testLoss

def train_epoch(train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics, obj_label):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0

    if obj_label:
        for batch_idx, (data, target, obj_l) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            data = tuple(Variable(d) for d in data)

            optimizer.zero_grad()
            outputs = model(*data)
            
            # print(target)
            # print(obj_l)
            # raw_input()

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = Variable(target)
                target = (target,)
                loss_inputs += target
                
            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.data[0])
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

                print(message)
                losses = []

        total_loss /= (batch_idx + 1)    
    else:
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()
            data = tuple(Variable(d) for d in data)

            optimizer.zero_grad()
            outputs = model(*data)

            if type(outputs) not in (tuple, list):
                outputs = (outputs,)

            loss_inputs = outputs
            if target is not None:
                target = Variable(target)
                target = (target,)
                loss_inputs += target

            loss_outputs = loss_fn(*loss_inputs)
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            losses.append(loss.data[0])
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            for metric in metrics:
                metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    batch_idx * len(data[0]), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), np.mean(losses))
                for metric in metrics:
                    message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

                print(message)
                losses = []

        total_loss /= (batch_idx + 1)
    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics, obj_label):
    for metric in metrics:
        metric.reset()
    model.eval()
    val_loss = 0
    if obj_label:
        for batch_idx, (data, target, obj_l) in enumerate(val_loader):
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
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.data[0]

            for metric in metrics:
                metric(outputs, target, loss_outputs)    
    else:
        for batch_idx, (data, target) in enumerate(val_loader):
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
            loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs
            val_loss += loss.data[0]

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
