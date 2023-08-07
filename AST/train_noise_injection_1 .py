import os
import gc
from tokenize import Double
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

import wandb

# other packages needed
from data_setting import CONFIG
from model import BirdCLEFModel
from loss import criterion
from utils import fetch_scheduler,preprocess_df,set_seed
import data_process as dp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type = int , default = 7,
                    help = 'the device for training')
parser.add_argument('--lr',type = float , default = 1e-4,
                    help = 'learning rate')
parser.add_argument('--weight_decay',type = float,default = 1e-6)
parser.add_argument('--epochs',type = int, default = 10,
                    help = 'epochs for training')
parser.add_argument('--train_batch_size', type = int ,default = 64)
parser.add_argument('--valid_batch_size', type = int , default = 32)
parser.add_argument('--num_workers', type = int , default = 2)
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

b_ = Fore.BLUE
sr_ = Style.RESET_ALL

import warnings
warnings.filterwarnings("ignore")

# For descriptive error messages
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


class BirdCLEFDataset(Dataset):
    def __init__(self, df, target_sample_rate, max_time, image_transforms=None,mode = 'train'):
        self.file_paths = df['file_path'].values   
        self.labels = df['primary_label'].values
        self.target_sample_rate = target_sample_rate
        num_samples = target_sample_rate * max_time
        self.num_samples = num_samples                 #待确认：num_samples的含义
        self.image_transforms = image_transforms
        self.mode = mode
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        filepath = self.file_paths[index]
        audio, sample_rate = torchaudio.load(filepath)
        audio = self.to_mono(audio)  #多声道求均值混合成单声道，再在时域上进行处理
        
        if sample_rate != self.target_sample_rate: #待确认：resample具体操作，猜测只是转化成目标采样频率即可
            resample = Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)
        
        if audio.shape[0] > self.num_samples: #应该是将audio转化为固定大小
            audio = self.crop_audio(audio)
            
        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)
            
        mel_spectogram = MelSpectrogram(sample_rate=self.target_sample_rate, 
                                        n_mels=CONFIG.n_mels, 
                                        n_fft=CONFIG.n_fft) #转化为梅尔频谱图
        mel = mel_spectogram(audio)
        label = torch.tensor(self.labels[index])
        
        # Convert to Image
        #image = torch.stack([mel, mel, mel]).numpy()
        image = mel.numpy()
        
        if self.mode == 'train':
            if random.random() < 0.7 and 'white_noise' in CONFIG.augs:
                image = dp.white_noise(image)
            if random.random() < 0.7 and 'pink_noise' in CONFIG.augs:
                image = dp.pink_noise(image)
            if random.random() < 0.7 and 'bandpass_noise' in CONFIG.augs:
                image = dp.bandpass_noise(image)
            if random.random() < 0.5 and 'upper' in CONFIG.augs:
                image = dp.low_pass_filter(image)
        
        image = torch.tensor(image)
        image = torch.stack([mel,mel,mel])
        

        # Normalize Image（简单粗暴----）
        max_val = torch.abs(image).max()
        image = image / max_val    
        
        return {
            "image": image, 
            "label": label
        }
            
    def pad_audio(self, audio):
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding) #奇怪的pad方式增加了
        return audio
        
    def crop_audio(self, audio):
        return audio[:self.num_samples]
        
    def to_mono(self, audio):
        return torch.mean(audio, axis=0)

def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        batch_size = images.size(0)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG.n_accumulate
            
        loss.backward()
        print('loss:{:.4f}'.format(loss))
        if (step + 1) % CONFIG.n_accumulate == 0:
            optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss

def valid_one_epoch(model, dataloader, device, epoch):
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    LABELS = []
    PREDS = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.long)
        
        batch_size = images.size(0)

        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        PREDS.append(preds.view(-1).cpu().detach().numpy())
        LABELS.append(labels.view(-1).cpu().detach().numpy())
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    LABELS = np.concatenate(LABELS)
    PREDS = np.concatenate(PREDS)
    val_f1 = f1_score(LABELS, PREDS, average='macro')
    gc.collect()
    
    return epoch_loss, val_f1

def prepare_loaders(df, fold):
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = BirdCLEFDataset(df_train, target_sample_rate=CONFIG.sample_rate, max_time=CONFIG.max_time,mode = 'train')
    valid_dataset = BirdCLEFDataset(df_valid, target_sample_rate=CONFIG.sample_rate, max_time=CONFIG.max_time,mode = 'val')

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, 
                              num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, 
                              num_workers=args.num_workers, shuffle=False, pin_memory=True)
    
    return train_loader, valid_loader

def run_training(model, optimizer, scheduler, device, num_epochs):
    # To automatically log gradients
    #wandb.watch(model, log_freq=100)
    
    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_epoch_f1 = 0
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device = device, epoch=epoch)
        torch.cuda.empty_cache()
        val_epoch_loss, val_epoch_f1 = valid_one_epoch(model, valid_loader, 
                                                       device = device, 
                                                       epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid F1'].append(val_epoch_f1)
        
        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        wandb.log({"Valid F1": val_epoch_f1})
        
        # deep copy the model
        if val_epoch_f1 >= best_epoch_f1:
            print(f"{b_}Validation F1 Improved ({best_epoch_f1} ---> {val_epoch_f1})")
            best_epoch_f1 = val_epoch_f1
            #run.summary["Best F1 Score"] = best_epoch_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "F1{:.4f}_epoch{:.0f}.bin".format(best_epoch_f1, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
            
        print()
    
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best F1: {:.4f}".format(best_epoch_f1))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

if __name__ == '__main__':
    
    set_seed()

    model = BirdCLEFModel(CONFIG.model_name,CONFIG.embedding_size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                       weight_decay=args.weight_decay)
    scheduler = fetch_scheduler(optimizer)
    
    df = preprocess_df()
    
    train_loader,valid_loader = prepare_loaders(df,fold = 0)
    
    run = wandb.init(project = CONFIG.competition,
                    job_type = 'Train',
                    name = 'test for noise injection',
                    tags = ['gem_pooling',CONFIG.model_name,'starter'],
                    anonymous = 'must')

    model, history = run_training(model, optimizer, scheduler,
                              device=device,
                              num_epochs=args.epochs)
    
    run.finish()
