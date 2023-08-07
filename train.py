#!/usr/bin/env python
# coding: utf-8

import cv2
import audioread
import logging
import gc
import os
import sys
#sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import random
import time
import warnings

import librosa
import colorednoise as cn
import numpy as np
import pandas as pd
import soundfile as sf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as torchdata

from contextlib import contextmanager
from joblib import Parallel, delayed
from pathlib import Path
from typing import Optional
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score

from albumentations.core.transforms_interface import ImageOnlyTransform
from torchlibrosa.stft import LogmelFilterBank, Spectrogram
from torchlibrosa.augmentation import SpecAugmentation
from tqdm import tqdm

import albumentations as A   
import albumentations.pytorch.transforms as T

import matplotlib.pyplot as plt
import argparse
from model import TimmSED

from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler,SubsetRandomSampler


import transformers
from torch.cuda.amp import autocast, GradScaler



import glob

from data_setting import CFG,AudioParams
from utils import AverageMeter,MetricMeter
from loss import *
from train_time_augmentation import *
from mydataset import WaveformDataset
import copy

import ast

#target_list = ['crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh']
target_list = ['barpet','ercfra','elepai','crehon','hawgoo','hawhaw','hawpet1','maupar','puaioh']
train = pd.read_csv('../bird_cleff/Birdcall_data/train_enlargedata_v8.csv')
#train = pd.read_csv('stage__analyze/fixed_v8.csv')
#enlarge = pd.read_csv('../bird_cleff/Birdcall_data/target_enlarge.csv')

#meta = pd.read_csv('../bird_cleff/Birdcall_data/train_metadata.csv')
'''
train['add_label'] = ['[]' for i in range(len(train))]
train['remove_label'] = ['[]' for i in range(len(train))]
enlarge['add_label'] = ['[]' for i in range(len(enlarge))]
enlarge['remove_label'] = ['[]' for i in range(len(enlarge))]

df_list = []
for bird in CFG.target_columns:
    if bird in target_list:
        df_list.append(enlarge[enlarge['primary_label'] == bird].reset_index(drop = True))
    else:
        df_list.append(train[train['primary_label'] == bird].reset_index(drop = True))
train = pd.concat(df_list,axis = 0,ignore_index = True)
'''

index_dict,samplenum = {},[]

init_num = []
for bird in CFG.target_columns:
    sub_df = train[train['primary_label'] == bird]
    init_num.append(len(sub_df))


        

'''
df_list = []

for bird in CFG.target_columns:
    bird_df = train[train['primary_label'] == bird].reset_index(drop = True)
    if len(bird_df) >= 500:
        bird_df = bird_df.sample(n = 500, random_state = CFG.seed).reset_index(drop = True)
    df_list.append(bird_df)

train = pd.concat(df_list,axis = 0,ignore_index = True)
'''




# split fold for train and validation
Fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for n, (trn_index, val_index) in enumerate(Fold.split(train, train['primary_label'])):
    train.loc[val_index, 'kfold'] = int(n)
train['kfold'] = train['kfold'].astype(int)

train.to_csv('train_folds.csv', index=False)

# set hyper parameters 
import os
parser = argparse.ArgumentParser()
parser.add_argument('--device', type = int , default = 0,
                    help = 'the device for training')
parser.add_argument('--lr',type = float , default = 1e-3,
                    help = 'learning rate')
parser.add_argument('--slr',type = float , default = 1e-4,
                    help = 'small learning rate for better finetune')
parser.add_argument('--flr',type = float , default = 1e-5,
                    help = 'small learning rate for better finetune with scored birds')
parser.add_argument('--weight_decay',type = float,default = 1e-6)
parser.add_argument('--epochs',type = int, default = 35,
                    help = 'epochs for training')
parser.add_argument('--fepochs',type = int, default = 5,
                    help = 'epochs for training')
parser.add_argument('--train_batch_size', type = int ,default = 16)
parser.add_argument('--valid_batch_size', type = int , default = 32)
parser.add_argument('--num_workers', type = int , default = 32)
parser.add_argument('--debug',type = int,default = 0)
args = parser.parse_args()


OUTPUT_DIR = f'./'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
   
    
def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(CFG.seed)
device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')



def train_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader, total = 900)#len(data_loader))
    myiter = 0
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['targets'].to(device)
        #----------nocall_detect--------------
        '''
        with torch.no_grad():
            n_pred = nocall_det(inputs)
            n_pred = torch.softmax(n_pred['clipwise_output'],dim = 1)
            call_rate = n_pred[:,1].item().reshape(inputs.shape[0],1)
        targets = (0.1 + 0.9*call_rate) * targets
        '''
        #-------------------------------------
        with autocast(enabled=CFG.apex):
            outputs = model(inputs)
            loss = bceloss_fn(outputs, targets)
        
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(targets, outputs)
        tk0.set_postfix(loss=losses.avg)

        
        myiter += 1
        if myiter > 900:
            break
        
        
    return scores.avg, losses.avg


def train_mixup_cutmix_fn(model, data_loader, device, optimizer, scheduler):
    model.train()
    scaler = GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    scores = MetricMeter()
    tk0 = tqdm(data_loader,total = 900) #len(data_loader))
    myiter = 0
    for data in tk0:
        optimizer.zero_grad()
        inputs = data['image'].to(device)
        targets = data['targets'].to(device)

        #----------nocall_detect--------------
        '''
        with torch.no_grad():
            n_pred = nocall_det(inputs)
            n_pred = torch.softmax(n_pred['clipwise_output'], dim = 1)
            call_rate = n_pred[:,1].reshape(inputs.shape[0],1)
        targets = (0.1 + 0.9*call_rate) * targets
        '''
        #-------------------------------------

        if np.random.rand() < 0.5:
            inputs, new_targets = mixup(inputs, targets, 0.4)
            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = bce_mixup_criterion(outputs, new_targets) 
        else:
            inputs, new_targets = cutmix(inputs, targets, 0.4)
            with autocast(enabled=CFG.apex):
                outputs = model(inputs)
                loss = bce_cutmix_criterion(outputs, new_targets)
        
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        
        scheduler.step()
        losses.update(loss.item(), inputs.size(0))
        scores.update(new_targets[0], outputs)
        tk0.set_postfix(loss=losses.avg)
        
        
        myiter += 1
        if myiter > 900:
            break
        
    return scores.avg, losses.avg


def valid_fn(model, data_loader, device):
    model.eval()
    losses = AverageMeter()
    scores = MetricMeter() #f1_0.3 和 f1_0.5两种评价指标
    tk0 = tqdm(data_loader, total=len(data_loader))
    valid_preds = []
    with torch.no_grad():
        for data in tk0:
            inputs = data['image'].to(device)
            targets = data['targets'].to(device)
            outputs = model(inputs)
            loss = bceloss_fn(outputs, targets)    #奇了怪了....，模型输出应该是dict
            losses.update(loss.item(), inputs.size(0))
            scores.update(targets, outputs)
            tk0.set_postfix(loss=losses.avg)
    return scores.avg, losses.avg

def prepare_loader(df, fold):
    global index_dict,samplenum
    if CFG.rating is not None:
        df_train = df[df.rating >= CFG.rating].reset_index(drop = True)
    if fold is not None:
        df_train = df[df.kfold != fold].reset_index(drop=True)
        df_valid = df[df.kfold == fold].reset_index(drop=True)
    else:
        df_train = df.copy(deep = True)
        df_valid = df.copy(deep = True)

    #train_dataset = BirdCLEFDataset(df_train, target_sample_rate=CONFIG.sample_rate, max_time=CONFIG.max_time,mode = 'train')
    #valid_dataset = BirdCLEFDataset(df_valid, target_sample_rate=CONFIG.sample_rate, max_time=CONFIG.max_time,mode = 'val')
    #prepare weighted sampler
    
    
    weights = torch.zeros(CFG.num_classes)
    for index in range(CFG.num_classes):
        bird = CFG.target_columns[index]
        num = len(df_train[df_train['primary_label'] == bird])
        weights[index] = 1/num
        
        if bird in CFG.scored_birds:
            weights[index] = weights[index] * 1.5
    

    train_dataset = WaveformDataset(df_train , mode = 'train')
    valid_dataset = WaveformDataset(df_valid , mode = 'valid')
    
    #train for stage1
    #train_dataset = Stage1Dataset(df_train , mode = 'train')
    #valid_dataset = Stage1Dataset(df_valid , mode = 'valid')

    sample_weights = weights[train_dataset.get_classes_for_all()]
    sampler = WeightedRandomSampler(weights = sample_weights,num_samples = len(sample_weights),replacement = True)
    
    
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                              num_workers=args.num_workers, shuffle=False, pin_memory=True,sampler = sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, 
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

# In[90]:


def inference_fn(model, data_loader, device):
    model.eval()
    tk0 = tqdm(data_loader, total=len(data_loader))
    final_output = []
    final_target = []
    with torch.no_grad():
        for b_idx, data in enumerate(tk0):
            inputs = data['image'].to(device)
            targets = data['targets'].to(device).detach().cpu().numpy().tolist()
            output = model(inputs)
            output = output["clipwise_output"].cpu().detach().cpu().numpy().tolist()
            final_output.extend(output)
            final_target.extend(targets)
    return final_output, final_target

#读取已训练的模型并计算结果
def calc_cv(model_paths):
    df = pd.read_csv('train_folds.csv')
    y_true = []
    y_pred = []
    for fold, model_path in enumerate(model_paths):
        model = TimmSED(
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels)

        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        val_df = df[df.kfold == fold].reset_index(drop=True)
        dataset = WaveformDataset(df=val_df, mode='valid')
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=CFG.valid_bs, num_workers=0, pin_memory=True, shuffle=False
        )

        final_output, final_target = inference_fn(model, dataloader, device)
        y_pred.extend(final_output)
        y_true.extend(final_target)
        torch.cuda.empty_cache()

        f1_03 = metrics.f1_score(np.array(y_true) >= 0.5, np.array(y_pred) > 0.3, average="micro")
        print(f'micro f1_0.3 {f1_03}')

    f1_03 = metrics.f1_score(np.array(y_true) >= 0.5, np.array(y_pred) > 0.3, average="micro")
    f1_05 = metrics.f1_score(np.array(y_true) >= 0.5, np.array(y_pred) > 0.5, average="micro")

    print(f'overall micro f1_0.3 {f1_03}')
    print(f'overall micro f1_0.5 {f1_05}')
    return

def run_training(model, optimizer, scheduler, device, num_epochs,fold,finetune = False):
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    min_loss = 999
    best_score = -np.inf
    
    global samplenum,index_dict,train_dataloader,valid_dataloader
    df_ex = []
    for epoch in range(num_epochs):
        print('starting {} epoch...'.format(epoch + 1))
        
        start_time = time.time()
        
        # use sample num to bulid dataloader for this epoch
        '''
        sample_list = []
        for id in range(len(CFG.target_columns)):
            num = samplenum[id]
            index = index_dict[CFG.id2bird[id]]
            res = np.random.choice(index,num,replace = True)
            sample_list.append(res)
        subset = np.concatenate(sample_list)

        sampler = SubsetRandomSampler(subset)

        train_dataloader = DataLoader(train_datasets,batch_size=args.train_batch_size,
                                num_workers = args.num_workers,shuffle = False, pin_memory=True,sampler=sampler)
        '''
        if best_score >= 0.7 and not finetune:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.slr, weight_decay=CFG.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.ETA_MIN, T_max=500)
        
        #bce loss seems not need train_fn
        if epoch < args.epochs:
            train_avg, train_loss = train_mixup_cutmix_fn(model, train_dataloader, device, optimizer, scheduler)
        else: 
            train_avg, train_loss = train_fn(model, train_dataloader, device, optimizer, scheduler)

        valid_avg, valid_loss = valid_fn(model, valid_dataloader, device)
    
        elapsed = time.time() - start_time

        print(f'Epoch {epoch+1} - avg_train_loss: {train_loss:.5f}  avg_val_loss: {valid_loss:.5f}  time: {elapsed:.0f}s')
        print(f"Epoch {epoch+1} - train_f1_at_03:{train_avg['f1_at_03']:0.5f}  valid_f1_at_03:{valid_avg['f1_at_03']:0.5f}")
        print(f"Epoch {epoch+1} - train_f1_at_05:{train_avg['f1_at_05']:0.5f}  valid_f1_at_05:{valid_avg['f1_at_05']:0.5f}")
        #print(valid_avg['scored_dict'])
        valid_avg['scored_dict']['epoch'] = [epoch]

        df_ex.append(pd.DataFrame(valid_avg['scored_dict']))
        

        if valid_avg['f1_at_03'] > best_score:
            print(f">>>>>>>> Model Improved From {best_score} ----> {valid_avg['f1_at_03']}")
            print(f"other scores here... {valid_avg['f1_at_03']}, {valid_avg['f1_at_05']}")
            torch.save(model.state_dict(), f'fold-{fold}.bin')
            best_score = valid_avg['f1_at_03']
        if (epoch+1)%10 == 0:
            torch.save(model.state_dict(), f'justry/fold-{fold}-{epoch+1}.bin')
        
        #adjust samplenum with the train results
        '''
        for i in range(CFG.num_classes):
            bird = CFG.id2bird[i]
            pre_score = valid_avg['scored_dict'][bird]
            if pre_score is None:
                break
            if pre_score >= valid_avg['f1_at_03'] + 0.1:
                samplenum[i] = int(samplenum[i] * 0.8) + 1 
            if pre_score < valid_avg['f1_at_03'] - 0.1:
                enlarge_ratio = 10
                if bird in CFG.scored_birds:
                    enlarge_ratio = 15
                samplenum[i] = min(int(samplenum[i] * 1.25)+5,2000,init_num[i]*enlarge_ratio)
        '''
            

    ex_record = pd.concat(df_ex,axis = 0,ignore_index = True)
    ex_record.to_csv(CFG.EXP_ID+'.csv')
    
    

# In[91]:


if __name__ == '__main__':

#-----------------------------------------
#create a df with target birds
    scored_df_list = []
    for bird in CFG.scored_birds:
        sdf = train[train['primary_label'] == bird].reset_index(drop = True)
        scored_df_list.append(sdf)
    scored_df = pd.concat(scored_df_list,axis = 0,ignore_index=True)
    #print(len(scored_df))
    
#-----------------------------------------
    for fold in range(5):
        if fold not in CFG.folds:
            continue
        print("=" * 100)
        print(f"Fold {fold} Training")
        print("=" * 100)
        
        train_dataloader,valid_dataloader = prepare_loader(train,fold)
    
        model = TimmSED(
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels)

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=CFG.ETA_MIN, T_max=500)
        
        model = model.to(device)

        run_training(model,optimizer,scheduler,device = args.device,num_epochs = args.epochs,fold = fold)
        

    model_paths = [f'fold-{i}.bin' for i in CFG.folds]

    calc_cv(model_paths)






