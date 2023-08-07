from bdb import effective
import os
import gc
import cv2
import math
import copy
import time
import random

# For data manipulation
import numpy as np
import pandas as pd

from pathlib import Path
# Pytorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torch.cuda import amp

# Audio 
import torchaudio
from torchaudio.transforms import MelSpectrogram, Resample,AmplitudeToDB

# Utils
import joblib
from tqdm import tqdm
from collections import defaultdict

# Sklearn Imports
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# For Image Models
import timm

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")
from data_setting import CONFIG

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'

def gem_freq(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), 1)).pow(1.0 / p)


class GeMFreq(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = torch.nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem_freq(x, p=self.p, eps=self.eps)

class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, embedding_size, pretrained=True):
        super(BirdCLEFModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.model.global_pool = nn.Identity()
        self.pooling = GeM()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.fc = nn.Linear(embedding_size, CONFIG.num_class)

    def forward(self, images):
        features = self.model(images)
        pooled_features = self.pooling(features).flatten(1)
        embedding = self.embedding(pooled_features)
        output = self.fc(embedding)
        return output

class AttHead(nn.Module):
    def __init__(
        self,in_chans, p = 0.5,num_class = 397,train_period = 15.0
    ,infer_period = 5.0):
        super(AttHead,self).__init__()
        self.train_period = train_period
        self.infer_period = infer_period
        self.pooling = GeMFreq()
        
        self.dense_layers = nn.Sequential(
            nn.Dropout(p/2),
            nn.Linear(in_chans, 512),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.attention = nn.Conv1d(
            in_channels = 512,
            out_channels = num_class,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = True,
        )
        self.fix_scale = nn.Conv1d(
            in_channels = 512,
            out_channels = num_class,
            kernel_size = 1,
            stride = 1,
            padding = 0,
            bias = True,
        )
    
    def forward(self,feat):
        feat = self.pooling(feat).squeeze(-2).permute(0,2,1)
        feat = self.dense_layers(feat).permute(0,2,1)
        time_att = torch.tanh(self.attention(feat))
        #feat (batch_size,channels,time) 是时序上的一维向量
            #计算序列的开始和结束
        #time_att 由特征得出的类别分布向量
        assert self.train_period >= self.infer_period

        if self.training or self.train_period == self.infer_period or 1:
        
            clipwise_pred = torch.sum(
                torch.sigmoid(self.fix_scale(feat))*torch.softmax(time_att, dim=-1),
                dim = -1,
            )
            logits = torch.sum(
                self.fix_scale(feat)*torch.softmax(time_att,dim = -1),
                dim = -1,
            )
            return logits,clipwise_pred
        else:
            #framewise_pred_long  (batch_size,class,time)  逐帧预测结果
            #clipwise_pred_long   (batch_size,class)       音频整体预测结果
            framewise_pred_long = torch.sigmoid(self.fix_scale(feat))
            clipwise_pred_long = torch.sum(framewise_pred_long*torch.softmax(time_att,dim = -1),dim = -1)

            feat_time = feat.size(-1)#时间序列长度

            start = (
                feat_time/2 - feat_time*(self.infer_period/self.train_period)/2
            )  
            end = start + feat_time*(self.infer_period/self.train_period)
            
            start = int(start)
            end = int(end)

            feat = feat[:,:,start:end]
            att = torch.softmax(time_att[:,:,start:end], dim = -1)#注意力概率向量
            
            #framewise_pred 片段逐帧预测结果 
            #clipwise_pred  片段整体预测结果
            framewise_pred = torch.sigmoid(self.fix_scale(feat))
            clipwise_pred = torch.sum(framewise_pred*att, dim = -1)

            #未用sigmoid概率化的向量和
            logits = torch.sum(
                self.fix_scale(feat)*att,
                dim = -1,
            )
            time_att = time_att[:,:,start:end]
        
        return (
            logits,
            clipwise_pred,
            self.fix_scale(feat).permute(0,2,1),#batch_size,time,class
            time_att.permute(0,2,1),#batch_size,time,class
            clipwise_pred_long,
        )


#输入张量：[batch,?,?]
class NormalizeMelSpec(nn.Module):
    def __init__(self,eps = 1e-12):
        super().__init__()
        self.eps = eps
    
    def forward(self,x):

        mean = x.mean((1,2),keepdim = True)
        std = x.std((1,2),keepdim = True)
        x_std = (x-mean)/(std+self.eps)

        norm_min = x_std.min(-1)[0].min(-1)[0]
        norm_max = x_std.max(-1)[0].max(-1)[0]

        fix_ind = (norm_max - norm_min) > self.eps
        fix_ind = fix_ind * torch.ones_like((norm_max - norm_min))
        
        v = torch.zeros_like(x_std)
        
        #归一化后存在非零特征值(保留下来的是对应的batch)
        if fix_ind.sum():
            v_fix = x_std[fix_ind]
            norm_max_fix = norm_max[fix_ind,None,None]
            norm_min_fix = norm_min[fix_ind,None,None]
            v_fix = torch.max(
                torch.min(v_fix,norm_max_fix),
                norm_min_fix,
            )
            v_fix = (v_fix - norm_min_fix)/(norm_max_fix - norm_min_fix)
            v[fix_ind] = v_fix
        return v

class AttModel(nn.Module):
    def __init__(
        self,backbone = 'resnet34',p = 0.5,n_mels = 224,num_class = 152,train_period = 15.0,infer_period = 5.0,in_chans = 1,
    ):
        super().__init__()
        self.n_mels = n_mels
        #音频图片转化+归一化   
        self.logmelspec_extractor = nn.Sequential(
            MelSpectrogram(
                CONFIG.sample_rate,
                n_mels = n_mels,
                f_min = 20,
                n_fft = 2048,
                hop_length = 512,
                normalized=True,
            ),
            AmplitudeToDB(top_db = 80.0),
            NormalizeMelSpec(),
        )
        #backbone 采用 resnet
        '''
        self.backbone = timm.create_model(
            backbone,pretrained=False,in_chans = in_chans
        )
        '''
        self.backbone = timm.create_model(
            backbone,features_only = True,pretrained=False#,in_chans = in_chans
        )
        '''
        dense_input = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        '''
        encoder_channels = self.backbone.feature_info.channels()
        dense_input = encoder_channels[-1]
        self.s = dense_input
        self.head = AttHead(
            dense_input,
            p = p,
            num_class = num_class,
            train_period = train_period,
            infer_period = infer_period,
        )
    
    def forward(self,input):
        feats = self.backbone(input)
        return self.head(feats[-1])

class Model(nn.Module):
    def __init__(
        self,
        backbone = 'resnet34',
        p = 0.5,
        n_mels = 224,
        num_class = CONFIG.num_class,
        train_period = CONFIG.period,
        infer_period = 5.0,
        in_chans = 1,
    ):
        super().__init__()
        self.model = AttModel(backbone, p, n_mels, num_class, train_period, infer_period, in_chans)
