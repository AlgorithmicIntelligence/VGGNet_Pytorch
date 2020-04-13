#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 15:56:01 2020

@author: lds
"""

from data_preprocessing.DataLoader_ILSVRC import ILSVRC2012 as Dataset
from models.VGGNet import VGGNet as VGGNet

import time, os
import torch
import torch.nn as nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler

import numpy as np
import matplotlib.pyplot as plt

train_dir = '/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_train'
val_dir = '/media/nickwang/StorageDisk/Dataset/ILSVRC2012/ILSVRC2012_img_val'
dirname_to_classname_path = './data_preprocessing/dirname_to_classname'
net_mode = 'A'

if net_mode == 'A':
    pretrained_weights = None
else:
    pretrained_weights = './weights/VGG_A_numCls100.pth'
  
start_epoch = 0
num_epoch = 74
batch_size_train = 32
momentum = 0.9
learning_rate = 0.01
num_classes = 100

trainset = Dataset(train_dir, dirname_to_classname_path, num_classes)
testset = Dataset(val_dir, dirname_to_classname_path, num_classes)
train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)
test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_train, shuffle=False, num_workers=8)


net = VGGNet(num_classes, mode=net_mode).cuda()
net.init_weights('KAMING') # ['VGG', 'XAVIER', 'KAMING']
if pretrained_weights != None:
    net_pretrain = torch.load(pretrained_weights)
    keys = list(net_pretrain.keys())
    idx_transfer = list(range(4*2)) + list(range(-3*2, 0, 1))
    for idx in idx_transfer:
        print(keys[idx], 'initilze')
        net.state_dict()[keys[idx]].copy_(net_pretrain[keys[idx]])
    
criterion = nn.CrossEntropyLoss().cuda()
optimizer= optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=0.0005)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=4, verbose=True)

train_loss_list = list()
train_accuracy_list = list()
test_loss_list = list()
test_accuracy_list = list()

for epoch in range(start_epoch, num_epoch):
    time_s = time.time()
    print('Epoch : ', epoch + 1)

    net.train()
    
    for batch_idx, (img, y_GT) in enumerate(train_dataloader):
        img = img.permute(0, 3, 1, 2).float()
         
        y_PD = net(img.cuda())
        loss = criterion(y_PD, y_GT.long().cuda())
        acc_batch = np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1))
        optimizer.zero_grad()    
        loss.backward()
        optimizer.step()

        if (batch_idx+1) % (len(train_dataloader)//40) == 0:
            print("Epoch {}, Training Data Num {}, Loss {}, Batch Accuracy {}%".format(epoch+1, (batch_idx + 1) * batch_size_train, loss.item(), np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))/len(y_GT)*100))
            print("labels(GT) = ", y_GT[:10].numpy())
            print("labels(PD) = ", np.argmax(y_PD.cpu().data.numpy()[:10], axis=1))
    
    net.eval()
    
    acc_train = 0
    loss_train = 0 
    for batch_idx, (img, y_GT) in enumerate(train_dataloader):
        img = img.permute(0, 3, 1, 2).float()
        with torch.no_grad():
            y_PD = net(img.cuda())
        loss = criterion(y_PD, y_GT.long().cuda())
        acc_train += np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))
        loss_train += loss.item()
    
    acc_train /= len(trainset)
    loss_train /= len(trainset) / batch_size_train
    train_loss_list.append(loss_train)
    train_accuracy_list.append(acc_train)
    print("Train Loss : ", loss_train, "Accuracy : %.2f%%" %(acc_train * 100))
    
    scheduler.step(loss_train) # adjsut learning rate. 
    
    acc_test = 0
    loss_test = 0   
    for batch_idx, (img, y_GT) in enumerate(test_dataloader):
        img = img.permute(0, 3, 1, 2).float()
        with torch.no_grad():
            y_PD = net(img.cuda())
        loss = criterion(y_PD, y_GT.long().cuda())
        acc_test += np.sum(np.equal(y_GT.numpy(), np.argmax(y_PD.cpu().data.numpy(), axis=1)))
        loss_test += loss.item()
    acc_test /= len(testset)
    loss_test /= len(testset) / batch_size_train
    test_loss_list.append(loss_test)
    test_accuracy_list.append(acc_test)
    print("Test Loss : ", loss_test, "Accuracy : %.2f%%" %(acc_test * 100))
    if not os.path.isdir('./weights'):
        os.mkdir('weights')
    torch.save(net.state_dict(), 'weights/VGG_{}_numCls{}_epoch{}.pth'.format(net_mode, num_classes, epoch+1))   
    print("Time Elapsed : ", time.time() - time_s)
torch.save(net.state_dict(), 'weights/VGG_{}_numCls{}.pth'.format(net_mode, num_classes))   
    
if not os.path.isdir('./records'):
    os.mkdir('records')    
x = np.arange(len(train_accuracy_list) + 1)
plt.figure()
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.plot(x, [0] + list(train_accuracy_list))
plt.plot(x, [0] + list(test_accuracy_list))
plt.legend(['training accuracy', 'testing accuracy'], loc='upper right')
plt.grid(True)
plt.savefig('./README/Accuracy_{}_numCls{}.png'.format(net_mode, num_classes)) 
plt.show()   
np.save('./records/train_acc_{}.npy'.format(net_mode), train_accuracy_list)
np.save('./records/train_loss_{}.npy'.format(net_mode), train_loss_list)

plt.figure()
plt.xlabel('epochs')
plt.ylabel('Loss')
plt.plot(x, list(train_loss_list[0:1]) + list(train_loss_list))
plt.plot(x, list(test_loss_list[0:1]) + list(test_loss_list))
plt.legend(['training loss', 'testing loss'], loc='upper right')
plt.savefig('./README/Loss_{}_numCls{}.png'.format(net_mode, num_classes))
plt.show()    
np.save('./records/test_acc_{}_epoch{}-{}.npy'.format(net_mode), test_accuracy_list)
np.save('./records/test_loss_{}_epoch{}-{}.npy'.format(net_mode), test_loss_list)
