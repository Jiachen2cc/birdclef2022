import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import joblib
from data_setting import CONFIG
from torch.optim import lr_scheduler
import torch.nn as nn

ROOT_DIR = '../bird_cleff/Birdcall_data'
TRAIN_DIR = '../bird_cleff/Birdcall_data/train_audio'
TEST_DIR = '../bird_cleff/Birdcall_data/test_soundscapes'

def set_seed(seed = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)

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

    skf = StratifiedKFold(n_splits = CONFIG.n_fold)

    for fold, (_, val_) in enumerate(skf.split(X = df, y = df.primary_label)):
        df.loc[val_, 'kfold']  = fold

    return df

def preprocess_df(file_name = 'train_metadata.csv'):

    df = get_meta_df()
    df = encoder_label(df)
    df = create_folds(df)

    return df

def fetch_scheduler(optimizer):
    if CONFIG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=CONFIG.T_max, 
                                                   eta_min=CONFIG.min_lr)
    elif CONFIG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG.T_max, 
                                                             eta_min=CONFIG.min_lr)
    elif CONFIG.scheduler == None:
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

