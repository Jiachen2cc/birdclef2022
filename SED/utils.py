import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import joblib
from data_setting import CFG
from torch.optim import lr_scheduler
import torch.nn as nn
from sklearn import metrics
import random
from data_setting import CFG

ROOT_DIR = '../bird_cleff/Birdcall_data'
TRAIN_DIR = '../bird_cleff/Birdcall_data/train_audio'
TEST_DIR = '../bird_cleff/Birdcall_data/test_soundscapes'


def slabel2list(s):
    for i in range(len(s)):
        x = s[i]
        llist = []
        if(len(x) == 2):
            pass
        else:
            x = x[1:-1]
            #print(x)
            x = x.split(', ')
            for label in x:
                llist.append(int(CFG.bird2id[label[1:-1]]))
        s[i] = llist
    return s        

def get_train_file_path(filename):
    return f"{TRAIN_DIR}/{filename}"

def get_meta_df(file_name = 'train_metadata.csv'):
    df = pd.read_csv(f"{ROOT_DIR}/{file_name}")
    print(df.head())
    df['file_path'] = df['filename'].apply(get_train_file_path)
    return df

def encoder_label(df):
    encoder = LabelEncoder()
    df['primary_label'] = encoder.fit_transform(df['primary_label'])

    with open('le.pkl','wb') as fp:
        joblib.dump(encoder, fp)
    
    return df

def create_folds(df):

    skf = StratifiedKFold(n_splits = CFG.n_fold,shuffle = True , random_state = CFG.seed)

    for fold, (_, val_) in enumerate(skf.split(X = df, y = df.primary_label)):
        df.loc[val_, 'kfold']  = fold

    return df

def preprocess_df(file_name = 'train_metadata.csv'):

    df = get_meta_df(file_name)
    df = encoder_label(df)
    df = create_folds(df)

    return df

def fetch_scheduler(optimizer):
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CFG.T_max, 
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CFG.T_max, 
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == None:
        return None
        
    return scheduler

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
        fix_ind = (fix_ind * torch.ones_like((norm_max - norm_min))).long()
        
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

class AverageMeter(object):
    #computes and stores the average and current value

    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n = 1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

# compute score for the model
class MetricMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.y_true = []
        self.y_pred = []
    
    def update(self, y_true, y_pred):
        self.y_true.extend(y_true.cpu().detach().numpy().tolist())
        self.y_pred.extend(y_pred['clipwise_output'].cpu().detach().numpy().tolist())
        
    @property
    def avg(self):
        #compute f1 score based on different prediction thresholds
        label = np.array(self.y_true) >= 0.5
        pred_03 = np.array(self.y_pred) > 0.3

        scored_dict = {}
        
        for id in range(CFG.num_classes):
            sub_label,sub_id = label[:,id],pred_03[:,id]
            if len(sub_label) == 0 or np.sum(sub_label) == 0:
                sub_f1 = None
            else:
                sub_f1 = metrics.f1_score(sub_label,sub_id)
            scored_dict[CFG.id2bird[id]] = [sub_f1]
        
        t_label,t_pred = label[:,CFG.scored_id],pred_03[:,CFG.scored_id]
        t_f1 = metrics.f1_score(t_label,t_pred,average = 'micro')
        scored_dict['total'] = [t_f1]
        
        self.f1_03 = metrics.f1_score(label,pred_03, average = 'micro')
        self.f1_05 = metrics.f1_score(label,np.array(self.y_pred) > 0.5, average = 'micro')
        
        return {
            'f1_at_03' : self.f1_03,
            'f1_at_05' : self.f1_05,
            'scored_dict':scored_dict
        }
