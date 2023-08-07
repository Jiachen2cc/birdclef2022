from fileinput import filename
import librosa
import numpy as np
import albumentations as A
import random
from torch.utils.data import Dataset

from data_setting import CFG,AudioParams
from audio_augmentation import *
import pandas as pd
import soundfile as sf
from utils import slabel2list
# In[83]:

#target_list = ['crehon', 'ercfra', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh']
target_list = ['barpet','ercfra','elepai','crehon','hawgoo','hawhaw','hawpet1','maupar','puaioh']
#target_list1 = ['crehon', 'hawgoo', 'hawhaw', 'hawpet1', 'maupar', 'puaioh']
target_list2 = ['akiapo','aniani','barpet','ercfra','elepai','hawcre','hawama','jabwar','houfin','warwhe1']
src_path = '../bird_cleff/Birdcall_data/train_audio/'
target_path1 = '../target_bird/train_audio/'
target_path2 = '../target_data/train_audio/' 
target_path = '../target_data/train_audio/'
no_call_noise_path = '../freefield/rich_metadata.csv'
noise_path = '../freefield/'
long_seg_path = '../long_segment/train_audio/'

#多个transform的封装函数
class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, y: np.ndarray, sr):
        for trns in self.transforms:
            y = trns(y, sr)
        return y

#单个transform的包装函数



class OneOf(Compose):
    # https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py
    def __init__(self, transforms, p=0.5):
        super().__init__(transforms)
        self.p = p
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]

    def __call__(self, y: np.ndarray, sr):
        data = y
        if self.transforms_ps and (random.random() < self.p):
            #?随机生成一个随机数生成器，决定采用变换的概率
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms, p=self.transforms_ps)
            data = t(y, sr)
        return data


def compute_melspec(y, params):
    """
    Computes a mel-spectrogram and puts it at decibel scale
    Arguments:
        y {np array} -- signal
        params {AudioParams} -- Parameters to use for the spectrogram. Expected to have the attributes sr, n_mels, f_min, f_max
    Returns:
        np array -- Mel-spectrogram
    """
    melspec = librosa.feature.melspectrogram(
        y=y, sr=params.sr, n_mels=params.n_mels, fmin=params.fmin, fmax=params.fmax,
    )

    #melspec = librosa.power_to_db(melspec).astype(np.float32)
    return melspec


def crop_or_pad(y, length, sr, train=True, probs=None):
    """
    Crops an array to a chosen length
    Arguments:
        y {1D np array} -- Array to crop
        length {int} -- Length of the crop
        sr {int} -- Sampling rate
    Keyword Arguments:
        train {bool} -- Whether we are at train time. If so, crop randomly, else return the beginning of y (default: {True})
        probs {None or numpy array} -- Probabilities to use to chose where to crop (default: {None})
    Returns:
        1D np array -- Cropped array
    """
    if len(y) < length:
        #可以在pad处也通过随机插入扩充样本
        if not train:
            start = 0
        else:
            start = np.random.randint(length - len(y))
        y1 = np.zeros(length)
        y1[start:start+len(y)] = y
        return y1.astype(np.float32)
        #y = np.concatenate([y, np.zeros(length - len(y))])
    elif len(y) >length :
        if not train:
            start = 0
        elif probs is None:
            start = np.random.randint(len(y) - length)
        else:
            start = (
                    np.random.choice(np.arange(len(probs)), p=probs) + np.random.random()
            )
            start = int(sr * (start))

        y = y[start: start + length]
        return y.astype(np.float32)
    else:
        return y.astype(np.float32)


def mono_to_color(X, eps=1e-6, mean=None, std=None):
    #将单通道矩阵转化为彩色频谱图
    #可能主要是用于可视化
    """
    Converts a one channel array to a 3 channel one in [0, 255]
    Arguments:
        X {numpy array [H x W]} -- 2D array to convert
    Keyword Arguments:
        eps {float} -- To avoid dividing by 0 (default: {1e-6})
        mean {None or np array} -- Mean for normalization (default: {None})
        std {None or np array} -- Std for normalization (default: {None})
    Returns:
        numpy array [3 x H x W] -- RGB numpy array
    """
    X = np.stack([X, X, X], axis=-1)

    # Standardize
    mean = mean or X.mean()
    std = std or X.std()
    X = (X - mean) / (std + eps)

    # Normalize to [0, 255]
    _min, _max = X.min(), X.max()

    if (_max - _min) > eps:
        V = np.clip(X, _min, _max)
        V = 255 * (V - _min) / (_max - _min)
        V = V.astype(np.uint8)
    else:
        V = np.zeros_like(X, dtype=np.uint8)
    return V


mean = (0.485, 0.456, 0.406) # RGB
std = (0.229, 0.224, 0.225) # RGB

# A:albumentations
albu_transforms = {
    'train' : A.Compose([
            A.HorizontalFlip(p=0.5),
            #A.augmentations.transforms.JpegCompression(p=0.5),
            #A.augmentations.transforms.ImageCompression(p=0.5, compression_type=A.augmentations.transforms.ImageCompression.ImageCompressionType.WEBP),
            A.OneOf([
                A.Cutout(max_h_size=5, max_w_size=16),
                A.augmentations.CoarseDropout(max_holes=4),
            ], p=0.5),
            A.Normalize(mean, std),
    ]),
    'valid' : A.Compose([
            A.Normalize(mean, std),
    ]),
}

#在数据集上执行相关变换
class WaveformDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 mode='train'):
        self.df = df
        noise_df = pd.read_csv(no_call_noise_path)
        noise_df = noise_df[noise_df['primary_label'] == 'nocall'].reset_index(drop = True)
        self.noise_df = noise_df
        self.mode = mode
        self.s_labels = slabel2list(df['secondary_labels'].values)
        #self.a_labels = slabel2list(df['add_label'].values)
        #self.r_labels = slabel2list(df['remove_label'].values)
        
        if mode == 'train':
            self.wave_transforms = Compose(
                [
                    OneOf(
                        [
                            NoiseInjection(p=1, max_noise_level=0.04),
                            GaussianNoise(p=1, min_snr=5, max_snr=20),
                            PinkNoise(p=1, min_snr=5, max_snr=20),
                            BrownNoise(p=1,min_snr=5, max_snr=20),
                        ],
                        p=0.2,
                    ),
                    #ExternalNoise(noise_df = self.noise_df,p=1,min_snr=5,max_snr=20),
                    RandomVolume(p=0.2, limit=4),
                    TimeShift(sr = AudioParams.sr),
                    Normalize(p=1),
                ]
            )
        else:
            self.wave_transforms = Compose(
                [
                    Normalize(p=1),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        SR = 32000
        sample = self.df.loc[idx, :]
        
        #wav_path = sample["file_path"]
        p_label = sample["primary_label"]
        stype = sample['type']
        #wav_path = src_path+sample['filename']
        
        #if p_label not in target_list:
        #    wav_path = src_path + sample['filename']
        #else:
        if stype == 'enlarge':
            wav_path = target_path+sample['filename']
        else:
            wav_path = src_path+sample['filename']
        
        '''
        if p_label in target_list1:
            wav_path = target_path1+sample['filename']
        elif p_label in target_list2:
            wav_path = target_path2+sample['filename']
        else:
            wav_path = src_path+sample['filename']
        '''
        
        s_labels = self.s_labels[idx]
        
        y = Audio_to_Array(wav_path)


        # SEC = int(len(y)/2/SR)
        # if SEC > 0:
        #     start = np.random.randint(SEC)
        #     end = start+AudioParams.duration
        
        #音频裁剪+数据增强
        if len(y) > 0:
            
            if len(y) > AudioParams.duration*SR:
                y = y[:AudioParams.duration * SR]
            
            #y = y[:AudioParams.duration*SR]

            if self.wave_transforms:
                y = self.wave_transforms(y, sr=SR)
        
        
        #长度对齐(较长的片段直接裁剪，较短的片段)
        y = np.concatenate([y, y, y])[:AudioParams.duration * AudioParams.sr] 
        y = crop_or_pad(y, AudioParams.duration * AudioParams.sr, sr=AudioParams.sr, train=True, probs=None)
        #-----------------------
        
        #音频转化到图片
        image = compute_melspec(y, AudioParams)

        if self.mode == 'train':
               image = random_power(image , power = 3 , c = 0.5)
        
        #get noise mel(long seg part no need noise mel)
        global noise_path
        
        if self.mode == 'train' and random.random() < 0.5:
            idy = np.random.randint(len(self.noise_df))
            n_path = noise_path + self.noise_df.loc[idy,:].filepath
            noise_mel = get_noise_mel(n_path)

            if noise_mel.shape[1] > CFG.len_check:
                start = np.random.randint(noise_mel.shape[1] - CFG.len_check)
                noise_mel = noise_mel[:,start:start+CFG.len_check]

            image = image + noise_mel
        
        #power2db + norm

        image = librosa.power_to_db(image)
        image = (image+80)/80

        
        
        if self.mode == 'train':
            if random.random() < 0.9:
                image = bandpass_noise(image)
            if random.random() < 0.5:
                image = lower_uper(image)
            image = random_power(image, power = 2 , c = 0.7)
        
        image = mono_to_color(image)
        image = image.astype(np.uint8)
        
        image = albu_transforms[self.mode](image=image)['image']
        image = image.T
        
        targets = np.zeros(len(CFG.target_columns), dtype=float)
        #get origin label
        targets[CFG.bird2id[p_label]] = 1
        for ebird_code in s_labels:
            targets[ebird_code] = 0.6
        
        #fixing_label
        '''
        for ebird_code in a_label:
            targets[ebird_code] = 0.6
        for ebird_code in r_label:
            targets[ebird_code] = 0
        '''
        

        return {
            "image": image,
            "targets": targets,
        }
    
    def get_classes_for_all(self):
        cla = list(self.df['primary_label'].values)
        res = [CFG.bird2id[bird] for bird in cla]
        return res



def Audio_to_Array(path):
    SR = AudioParams.sr
    USE_SEC = 60
    y, sr = sf.read(path, always_2d=True)
    y = np.mean(y, 1) # there is (X, 2) array
    #if len(y) > SR:
    #     y = y[SR:-SR]

    if len(y) > SR * USE_SEC:
        y = y[:SR * USE_SEC]

    return y


def get_noise_mel(path):

    SR = AudioParams.sr
    wav,sr = librosa.load(noise_path+path,sr = SR)
    mel = compute_melspec(wav,AudioParams)

    mel = np.concatenate((np.zeros((CFG.n_mels,CFG.len_check)),mel),axis = 1)
    mel = np.concatenate((mel,np.zeros((CFG.n_mels,CFG.len_check))),axis = 1)

    start = np.random.randint(mel.shape[1] - CFG.len_check)

    mel = mel[:,start:start + CFG.len_check]
    mel = random_power(mel)

    return mel

def external_noise(path , y , min_snr = 5, max_snr = 20):

    SR = AudioParams.sr
    noise,sr = librosa.load(noise_path+path, sr = SR)
    
    snr = np.random.uniform(min_snr,max_snr)
    a_signal = np.sqrt(y ** 2).max()
    an_noise = a_signal / (10 ** (snr/20))

    #clip noise
    start = np.random.randint(len(noise) - AudioParams.duration * AudioParams.sr)
    noise = noise[start:start + AudioParams.duration * AudioParams.sr]

    a_noise = np.sqrt(noise ** 2).max()
    augmented = (y + noise * 1/a_noise * an_noise).astype(y.dtype)

    return augmented







