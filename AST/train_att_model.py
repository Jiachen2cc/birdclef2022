import os
from pickle import LONG

from matplotlib import image
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
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
from sklearn.metrics import f1_score,label_ranking_average_precision_score

# For Image Models
import timm

# For colored terminal text
from colorama import Fore, Back, Style

import wandb

# other packages needed
from data_setting import CONFIG
from model import BirdCLEFModel,Model
from loss import criterion
from utils import fetch_scheduler,preprocess_df,set_seed,NormalizeMelSpec
import data_process as dp

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type = int , default = 7,
                    help = 'the device for training')
parser.add_argument('--lr',type = float , default = 1e-4,
                    help = 'learning rate')
parser.add_argument('--weight_decay',type = float,default = 1e-6)
parser.add_argument('--epochs',type = int, default = 30,
                    help = 'epochs for training')
parser.add_argument('--train_batch_size', type = int ,default = 32)
parser.add_argument('--valid_batch_size', type = int , default = 32)
parser.add_argument('--num_workers', type = int , default = 32)
args = parser.parse_args()

device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

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
        self.normalizer = NormalizeMelSpec()
        self.dive_coef = 100
        self.mel_spectrogram = MelSpectrogram(sample_rate=self.target_sample_rate, 
                                        n_mels=CONFIG.n_mels, 
                                        n_fft=CONFIG.n_fft) #转化为梅尔频谱图
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        # stragety : soft label and mixed spectrum
        if self.mode == 'train':
            image,label = self.augmentation_during_training(index)
        
        elif self.mode == 'val':
            image,label = self.preprocess_before_validation(index)
        

        # Normalize Image（简单粗暴----）
        #max_val = torch.abs(image).max()
        #image = image / max_val    
        
        return {
            "image": image, 
            "label": label
        }
            
    def pad_audio(self, audio):
        pad_length = self.num_samples - audio.shape[0]
        last_dim_padding = (0, pad_length)
        audio = F.pad(audio, last_dim_padding) #奇怪的pad方式增加了
        return audio
        
    def random_crop_audio(self, audio):
        start = random.randint(0,audio.shape[0] - self.num_samples - 1 )
        return audio[start : start + self.num_samples]
        
    def to_mono(self, audio):
        return torch.mean(audio, axis=0)
    
    def augmentation_during_training(self,index,num_for_mix = 2):
        mix_list = [index]
        for i in range(num_for_mix):
            mix_list.append(random.randint( 0, len(self.file_paths) -1 ))
        
        image = np.zeros((CONFIG.n_mels , CONFIG.len_check),dtype = np.float32)
        label = torch.zeros(CONFIG.num_class)

        for idx in mix_list:
            filepath = self.file_paths[idx]
            audio,sample_rate = torchaudio.load(filepath)
            audio = self.to_mono(audio)

            if sample_rate != self.target_sample_rate:
                resample = Resample(sample_rate, self.target_sample_rate)
                audio = resample(audio)
            
            if audio.shape[0] > self.num_samples:
                #存在冒险，裁剪可能减去包含有效声音的部分
                audio = self.random_crop_audio(audio)
            
            if audio.shape[0] < self.num_samples:
                audio = self.pad_audio(audio)
            mel = self.mel_spectrogram(audio).numpy()
            
            if random.random() < 0.7 and 'white_noise' in CONFIG.augs:
                mel = dp.white_noise(mel)
            if random.random() < 0.7 and 'pink_noise' in CONFIG.augs:
                mel = dp.pink_noise(mel)
            if random.random() < 0.7 and 'bandpass_noise' in CONFIG.augs:
                mel = dp.bandpass_noise(mel)
            if random.random() < 0.5 and 'upper' in CONFIG.augs:
                mel = dp.low_pass_filter(mel)
            
            image = image + mel*((random.random()*self.dive_coef + 1)/100)

            #label mix
            id = int(self.labels[idx])
            label[id] += 1.0


        image = torch.as_tensor(image)
        image = torch.stack([image,image,image])
        image = self.normalizer(image)
        
        return image , label
    
    def preprocess_before_validation(self,index):

        file_path = self.file_paths[index]
        audio , sample_rate = torchaudio.load(file_path)
        audio = self.to_mono(audio)

        if sample_rate != self.target_sample_rate:
            resample = Resample(sample_rate, self.target_sample_rate)
            audio = resample(audio)
        
        if audio.shape[0] > self.num_samples:
            audio = audio[:self.num_samples]
        if audio.shape[0] < self.num_samples:
            audio = self.pad_audio(audio)
        
        mel = self.mel_spectrogram(audio)
        image = torch.stack([mel,mel,mel])
        image = self.normalizer(image)

        #label = torch.tensor(self.labels[index])
        #one_hot label
        id = int(self.labels[index])
        label = torch.zeros(CONFIG.num_class)
        label[id] = 1.0

        return image,label

        
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.float)
        batch_size = images.size(0)
        '''
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss = loss / CONFIG.n_accumulate
            
        loss.backward()
        print('loss:{:.4f}'.format(loss))
        '''
        #print(images.shape)
        logits,_ = model(images)
        loss = criterion(logits, labels)
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

# can only be used when device == gpu
def train_one_epoch_with_amp(model, optimizer, scheduler,dataloader, device, epoch):
    model.train()
    
    dataset_size = 0
    running_loss = 0.0
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))

    scaler = torch.cuda.amp.GradScaler()
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.float)
        batch_size = images.size(0)
        
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / CONFIG.n_accumulate
        
        scaler.scale(loss).backward()
        print('loss:{:.4f}'.format(loss))
        if (step + 1) % CONFIG.n_accumulate == 0:
            #optimizer.step()
            # zero the parameter gradients
            #optimizer.zero_grad()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
                
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        
        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    
    return epoch_loss

def valid_one_epoch(model, dataloader, device, epoch):#
    model.eval()
    
    dataset_size = 0
    running_loss = 0.0
    
    LABELS = []
    PREDS = []
    LONG_PREDS = []
    
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:        
        images = data['image'].to(device, dtype=torch.float)
        labels = data['label'].to(device, dtype=torch.float)
        
        batch_size = images.size(0)
        
        with torch.no_grad():
            logits, clipwise_pred = model(images)
        #_, preds = torch.max(outputs, 1)
        loss = criterion(logits, labels)
        
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size
        
        epoch_loss = running_loss / dataset_size
        #print('pred_shape',clipwise_pred.shape)
        #print('label_shape',labels.shape)
        PREDS.append(clipwise_pred.cpu().detach().numpy())
        #LONG_PREDS.append(clipwise_pred_long.view(-1).cpu().detach().numpy())
        LABELS.append(labels.cpu().detach().numpy())
        
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])   
    
    LABELS = np.concatenate(LABELS,axis = 0)
    PREDS = np.concatenate(PREDS,axis = 0)
    #print('PRED_shape',PREDS.shape,'LABELS_shape',LABELS.shape)
    #LONG_PREDS = np.concatenate(LONG_PREDS)
    #val_f1 = f1_score(LABELS, PREDS, average='macro')
    val_f1 = label_ranking_average_precision_score(LABELS,PREDS)
    #val_f1_long = f1_score(LABELS,LONG_PREDS,average = 'macro')
    gc.collect()
    
    return epoch_loss, val_f1#,val_f1_long

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
    best_epoch_long_f1 = 0
    history = defaultdict(list)
    
    for epoch in range(1, num_epochs + 1): 
        gc.collect()
        print(device.type)
        '''
        if device.type == 'cpu':
            train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device = device, epoch = epoch)
        else:
            train_epoch_loss = train_one_epoch_with_amp(model, optimizer,scheduler,
                                           dataloader = train_loader,
                                           device = device, epoch = epoch)
        '''
        
        train_epoch_loss = train_one_epoch(model, optimizer, scheduler, 
                                           dataloader=train_loader, 
                                           device = device, epoch = epoch)
        
        torch.cuda.empty_cache()
        val_epoch_loss, val_epoch_f1 = valid_one_epoch(model, valid_loader, 
                                                       device = device, 
                                                       epoch=epoch)
    
        history['Train Loss'].append(train_epoch_loss)
        history['Valid Loss'].append(val_epoch_loss)
        history['Valid F1'].append(val_epoch_f1)
        #history['Valid F1_long'].append(val_epoch_long_f1)
        
        # Log the metrics
        ''''''
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})
        wandb.log({"Valid F1": val_epoch_f1})
        
        # deep copy the model
        if val_epoch_f1 >= best_epoch_f1:
            print(f"{b_}Validation F1 Improved ({best_epoch_f1} ---> {val_epoch_f1})")
            best_epoch_f1 = val_epoch_f1
            #run.summary["Best F1 Score"] = best_epoch_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "trained_att_model/F1{:.4f}_epoch{:.0f}.bin".format(best_epoch_f1, epoch)
            torch.save(model.state_dict(), PATH)
            # Save a model file from the current directory
            print(f"Model Saved{sr_}")
        '''
        if val_epoch_long_f1 >= best_epoch_long_f1:
            print(f"{b_}Validation long F1 Improved ({best_epoch_long_f1} ---> {val_epoch_long_f1})")
        '''
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

    #model = BirdCLEFModel(CONFIG.model_name,CONFIG.embedding_size)
    model = Model(backbone = CONFIG.model_name)
    model = model.model
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, 
                       weight_decay=args.weight_decay)
    scheduler = fetch_scheduler(optimizer)
    
    df = preprocess_df()
    
    train_loader,valid_loader = prepare_loaders(df,fold = 0)

    
    run = wandb.init(project = CONFIG.competition+'_att',
                    job_type = 'Train_with_att',
                    name = 'test for attention',
                    tags = ['gem_pooling','Att_model','baseline'],
                    anonymous = 'must')
    
    model, history = run_training(model, optimizer, scheduler,
                              device=device,
                              num_epochs=args.epochs)
    
    run.finish()
