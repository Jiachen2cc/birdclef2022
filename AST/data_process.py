from ast import fix_missing_locations
from json import load
from sys import path_hooks
import pandas as pd
#import plotly.graph_objects as go
import seaborn as sns
#import descartes 
#import geopandas as gpd
#from shapely.geometry import Point,Polygon
import matplotlib.pyplot as plt
#import IPython.display as ipd
import librosa
import librosa.display
import numpy as np
import random
from torch import fmax
from torchvision import transforms
import cv2 as cv
import os
from torch.utils.data import Dataset, DataLoader
import re
from torchaudio.transforms import MelSpectrogram,Resample
import torchaudio
import torch
import time



class CONFIG:
    def __init__(self):
        self.sr = 32000
        self.audio_root = 'Birdcall_data/train_audio'
        self.npy_dest = 'Birdcall_data/train_mel'
        self.img_dest = 'Birdcall_data/train_melimg'
        self.meta_path = 'Birdcall_data/train_metadata.csv'
        self.draw_path = 'Birdcall_data/draw_pic'
        self.num_mel = 224
        self.n_fft = 1024
        self.hop_length = 80
        self.level_noise = 0.05
        self.strains = 152

config = CONFIG()
#config.meta_data = pd.read_csv(config.meta_path)


#-------------------------------------
# explore data
def read_meta(meta_path = config.meta_path):
    meta_data = pd.read_csv(meta_path)
    meta_data = meta_data.drop(columns=['url','license','common_name','scientific_name'])
    #print(meta_data.head())

    # Number of different species
    #print(len(meta_data['primary_label'].value_counts()))


    # Number of training examples each species
    '''
    species = meta_data['primary_label'].value_counts()

    fig = go.Figure(data = [go.Bar(y= species.values,x = species.index)],
    layout = go.Layout(margin = go.layout.Margin(l=0,r=0,b=10,t=50)))

    fig.update_layout(title = 'Number of training samples per species')
    fig.show()
    '''

    # Background sounds(not reliable)
    print(meta_data['secondary_labels'].value_counts())
'''   
def visual_location(meta_path = config.meta_path):
    world_map = gpd.read_file('world_shape/world_shapefile.shp')
    crs = {'init':"epsg:4326"}

    train =pd.read_csv(meta_path)

    #choose some species
    species_list = ['houspa','mallar3','moudov','houfin']
    
    data = train[train['primary_label'].isin(species_list)]
    data['latitude'] = data['latitude'].astype(float)
    data['longitude'] = data['longitude'].astype(float)

    geometry = [Point(xy) for xy in zip(data['longitude'],data['latitude'])]

    geo_df = gpd.GeoDataFrame(data,crs = crs,geometry = geometry)

    species_id = geo_df['primary_label'].value_counts().reset_index()
    species_id.insert(0,'ID',range(0,0+len(species_id)))
    species_id.columns = ['ID','primary_label','count']

    geo_df = pd.merge(geo_df,species_id,how='left',on='primary_label')

    fig,ax = plt.subplots(figsize = (16,10))
    world_map.plot(ax = ax, alpha = 0.4, color = 'grey')

    palette = iter(sns.hls_palette(len(species_id)))
    for i in range(len(species_list)):
        geo_df[geo_df['ID'] == i].plot(ax = ax,
                                       markersize = 20,
                                       color = next(palette),
                                       marker='o',
                                       label = species_id['primary_label'].values[i])
    ax.legend()
    plt.show()
'''
def audio_explore(path = 'Birdcall_data/train_audio/afrsil1/XC125458.ogg'):
    #ipd.Audio(path)
    sig,rate = librosa.load(path,sr = None)
    #print('signal shape:', sig.shape)
    #print('signal rate:', rate)
    '''
    plt.figure(figsize=(15,5))
    librosa.display.waveshow(sig,sr = rate)
    plt.show()
    '''
    spec = librosa.feature.melspectrogram(y = sig,sr = rate,n_mels = 128,fmin = 0, fmax = 16000,n_fft = 3200,hop_length = 80)
    print(spec.shape)
#-------------------------------------

#-------------------------------------
#loda data and label
def load_data(ori = config.img_dest):
    img_data,bird_list = [],[]
    species = os.listdir(ori)
    print('number of different species:{}'.format(len(species)))
    for x in species:
        bird_list.append(x)
        imgs_path = os.path.join(ori,x)
        imgs = os.listdir(imgs_path)
        for name in imgs:
            img_data.append(os.path.join(imgs_path,name))
    #print(img_data)
    #print(len(bird_list))
    print(bird_list)
    return img_data,bird_list       
#get supervised information from meta data
def get_supinfo(path):
    seg = path.split('\\')
    strain,filename = seg[1],seg[2]
    filename = filename.split('.')[0]
    filename = strain+'/'+filename+'.ogg'

    df1 = config.meta_data[config.meta_data['filename'] == filename]
    df1 = df1.drop(columns=['url','license','common_name','scientific_name'])
    return df1
#reshape the data(change shape[1])
def reshape_mel(mel,len_1):
    if mel.shape[1]>len_1: 
        start = random.randint(0, mel.shape[1] - len_1 - 1)
        mel = mel[:, start : start + random.randint(len_1-14, len_1)]
    else:
        len_zero = random.randint(0, len_1-mel.shape[1])
        mel = np.concatenate((np.zeros((config.num_mel,len_zero)),mel), axis=1)

    mel = np.concatenate((mel,np.zeros((config.num_mel,len_1-mel.shape[1]))), axis=1)
    
    return mel
#-------------------------------------
#visualize & save(incomplete)
def draw_img(img):
    #assume img tensor [3,row,col]
    img = img.permute(1,2,0).numpy()
    img = img - img.min()
    img = img/(img.max()+1e-12)
    return img

#-------------------------------------
#-------------------------------------
#generate noise

def white_noise(img):
    noise = (np.random.sample(img.shape)+9)*img.mean()*config.level_noise*(np.random.sample()+0.3)
    img = img+noise
    return img

def pink_noise(img):
    row = random.randint(1,config.num_mel)
    noise1 = np.array([np.concatenate((1-np.arange(row)/row,np.zeros(config.num_mel-row)))]).T
    noise2 = (np.random.sample(img.shape)+9)*2*img.mean()*config.level_noise*(np.random.sample()+0.3)
    img = img*noise1+noise2

    return img

def bandpass_noise(img):
    a = random.randint(0,config.num_mel//2)
    b = random.randint(a+20,config.num_mel)
    
    noise = (np.random.sample((b-a,img.shape[1]))+9)*0.1*img.mean()*config.level_noise*(np.random.sample()+0.3)
    img[a:b,:] = img[a:b,:] + noise

    return img

def low_pass_filter(img):
    img = img - img.min()
    r = random.randint(config.num_mel//2,config.num_mel)
    x = random.random()/2
    noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(config.num_mel-r)-x+1))]).T
    img = img*noise
    img = img/img.max()

    return img


#-------------------------------------

#augmentation method section
#pipeline gray2color(optimal)-norm_pix-gamma_transform(optimal)
#返回值为0~1的浮点图
def random_gamma_transform(img,power = 1.5, c= 0.7):
    gamma_img = img**(random.random()*power+c)
    return gamma_img

def norm_pix(img,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]):
    trans = transforms.Compose([transforms.Normalize(mean=mean,std=std)])
    img = trans(img/255)
    return img

def gray2color(x,len_check):
    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize([config.num_mel,len_check]),transforms.ToTensor()])
    x = np.stack([x,x,x],axis = -1)
    v = (255*x).astype(np.uint8)
    v = trans(v)
    return v

def mono2gray(x,eps = 1e-12,mean = None,std = None):

    mean = mean or x.mean()
    std = std or x.std()
    x = (x - mean)/(std + eps)

    _min,_max = x.min(),x.max()

    if (_max - _min) > eps:
        v = np.clip(x, _min, _max)
        v = 255*(v - _min)/(_max - _min)
        v = v.astype(np.uint8)
    else:
        v = np.zeros_like(x, dtype = np.uint8)
    
    return v

#音频转化为梅尔频谱图
mel_spectrogram = MelSpectrogram(sample_rate = config.sr, n_mels = config.num_mel , n_fft = config.n_fft)

def preprocess_data(ori = config.audio_root,dest = config.npy_dest):
    species = os.listdir(ori)

    if not os.path.exists(dest):
        os.mkdir(dest)

    for x in species:
        audios_path = os.path.join(ori,x)
        npy_path = os.path.join(dest,x)

        #判断图片文件夹是否存在，若不存在则创建
        if not os.path.exists(npy_path):
            os.mkdir(npy_path)

        audios = os.listdir(audios_path)
        for name in audios:
            print(os.path.join(audios_path,name))
            #audio,_ = librosa.load(os.path.join(audios_path,name),sr = self.sr)
            start = time.time()
            audio,sample_rate = torchaudio.load(os.path.join(audios_path,name))
            audio = torch.mean(audio,axis = 0)

            if sample_rate != config.sr:
                resample = Resample(sample_rate,config.sr)
                audio = resample(audio)
            
            mel = mel_spectrogram(audio).numpy()
            print('time used during change audio into mel takes {:.4f} seconds'.format(time.time() - start))
            #img = self.ogg2img(audio)
                
                #设置图片名
            i_name = name.split('.')[0]+'.npy'
            #cv.imwrite(os.path.join(npy_path,i_name),img)
            np.save(os.path.join(npy_path,i_name),mel)

            start = time.time()
            mel = np.load(os.path.join(npy_path,i_name))
            print('time used during load a numpy.ndarry takes {:.4f} seconds'.format(time.time() - start))

            break

        break
            
    print('finish converting!')


def preprocess_meta(path = config.meta_path):
    meta = pd.read_csv(path)
    meta['secondary_labels'] = meta['secondary_labels'].map(lambda x:','.join(re.split(r'\W+',x)[1:-1]))
    meta.to_csv(path)
    '''
    f = meta['secondary_labels']
    f = f.iloc[1]
    s = re.split(r'\W+',f)[1:-1]
    u = ','.join(s)
    print(u.split(','))
    '''
    
        


#read_meta()
#visual_location()
#audio_explore()
#audio2mel()

if __name__ == '__main__':
    #preprocess_meta()
    #load_data()
    preprocess_data()
    