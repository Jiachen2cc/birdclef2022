import pandas as pd
import torch
from utils import slabel2list
from data_setting import AudioParams,CFG
import soundfile as sf
from mydataset import Audio_to_Array,Compose
from audio_augmentation import Normalize
import copy
import numpy as np
from torch.utils.data import Dataset,DataLoader
import librosa
import albumentations as A
from model import TimmSED
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device',type = int, default = 0)
parser.add_argument('--batch_size',type = int,default = 32)
parser.add_argument('--num_workers',type = int,default = 32)
parser.add_argument('--debug',type = int ,default = 0)

args = parser.parse_args()

# we try to use signal model to discover missing label and refuse wrong label at a time
# to make sure the operation is predictable,we discover new label with high threshold and refuse wrong label with low threshold



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

    melspec = librosa.power_to_db(melspec).astype(np.float32)
    melspec = (melspec+80)/80
    return melspec

def mono_to_color(X, eps=1e-6, mean=None, std=None):
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
#find files without second_label

albu_transforms = {
    'train' : A.Compose([
            #A.HorizontalFlip(p=0.5),
            #A.augmentations.transforms.JpegCompression(p=0.5),
            #A.augmentations.transforms.ImageCompression(p=0.5, compression_type=A.augmentations.transforms.ImageCompression.ImageCompressionType.WEBP),
            #A.OneOf([
            #    A.Cutout(max_h_size=5, max_w_size=16),
            #    A.augmentations.CoarseDropout(max_holes=4),
            #], p=0.5),
            A.Normalize(mean, std),
    ]),
    'valid' : A.Compose([
            A.Normalize(mean, std),
    ]),
}

meta_path = '../bird_cleff/Birdcall_data/train_enlargedata_v8.csv'
#meta_path = '../optimize_training_data/train_enlargedata_v8.csv'
meta = pd.read_csv(meta_path)
#no_sec = meta[meta['secondary_labels'] == '[]']
if args.debug:
    meta = meta[:1000]

#clip to 30s
SR = 32000
duration = 30
src_path = '../bird_cleff/Birdcall_data/train_audio/'
filename = meta['filename'].values


def get_clip_list(filepath):

    y = Audio_to_Array(filepath)
    clip_list = []
    #divide into 30s segment
    s = int(len(y)//SR)
    limit = len(y)
    start = 0
    for start in range(0,s,30):
        if SR*(start + 30) > limit:
            break
        end = start + 30
        target_clip = copy.deepcopy(y[start*SR:end*SR])
        clip_list.append(target_clip)
    else:
        start = start + 30
    if start < s:
        target_clip = copy.deepcopy(y[start*SR:])
        target_clip = np.concatenate([target_clip,np.zeros([(duration*SR - len(target_clip))])])
        clip_list.append(target_clip)
    
    return clip_list

class Denoisedataset(Dataset):

    def __init__(self,df):
        self.df = df
        self.wave_transform = Compose(
            [
                Normalize(p=1),
            ]
        )
        self.target_path = '../target_data/train_audio/'
        self.src_path = '../bird_cleff/Birdcall_data/train_audio/'

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        SR = 32000
        sample = self.df.loc[idx, :]

        p_label = sample['primary_label']
        s_labels = slabel2list([sample['secondary_labels']])[0]
        src = sample['type']

        if src == 'enlarge':
            wav_path = self.target_path + sample['filename']
        else:
            wav_path = self.src_path + sample['filename']
        
        y = Audio_to_Array(wav_path)
        
        if len(y) > 0:
            
            if len(y) > AudioParams.duration*SR:
                start = np.random.randint(len(y) - AudioParams.duration*SR)
                y = y[start:start + AudioParams.duration * SR]
            
            #y = y[:AudioParams.duration*SR]


            if self.wave_transform:
                y = self.wave_transform(y, sr=SR)

        y = np.concatenate([y, y, y])[:AudioParams.duration * AudioParams.sr] 
        y = crop_or_pad(y, AudioParams.duration * AudioParams.sr, sr = AudioParams.sr, train=True, probs=None)
        
        image = compute_melspec(y, AudioParams)
        image = mono_to_color(image)
        image = image.astype(np.uint8)

        image = albu_transforms['valid'](image = image)['image']
        image = image.T

        return {
            'image':image#,
            #'p_label':p_label,
            #'s_labels':s_labels
        }
    
class Clipdataset(Dataset):

    def __init__(self,clip_list):
        self.clip_list = clip_list
        self.wave_transform = Compose(
            [
                Normalize(p=1),
            ]
        )

    def __len__(self):
        return len(self.clip_list)
    
    def __getitem__(self, index):

        y = np.nan_to_num(self.clip_list[index])
        #y = self.wave_transform(y,sr = SR)

        image = compute_melspec(y,AudioParams)
        image = mono_to_color(image)
        image = image.astype(np.uint8)

        image = albu_transforms['valid'](image = image)['image']
        image = image.T

        return {'image':image}


if __name__ == '__main__':
    
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    # load high performance model
    model_paths = ['denoise_model/fold-0-do-074.bin','angry-fold/fold-0-an.bin']

    models = []
    for p in model_paths:
        model = TimmSED(
            base_model_name=CFG.base_model_name,
            pretrained=CFG.pretrained,
            num_classes=CFG.num_classes,
            in_channels=CFG.in_channels)
    
        model.to(device)
        model.load_state_dict(torch.load(p,map_location = {'cuda:3':'cuda:0','cuda:2':'cuda:0','cuda:4':'cuda:0','cuda:7':
                                                      'cuda:0','cuda:1':'cuda:0'}))
    model.eval()
    models.append(model)
    
        
    predict_dict = {}

    predict_dict['filename'] = filename
    for bird in CFG.target_columns:
        predict_dict[bird] = []


    print('use {}sec clip'.format(AudioParams.duration))
    dataset = Denoisedataset(df = meta)
    loader = DataLoader(dataset , batch_size = args.batch_size, shuffle = False ,num_workers = args.num_workers)
    for data in tqdm(loader):
        image = data['image'].to(device)
    
        with torch.no_grad():
            probas = []
            for model in models:
                with torch.cuda.amp.autocast():
                    output = model(image)
            probas.append(output['clipwise_output'].detach().cpu().numpy())#.reshape(-1))
            
            probas = np.array(probas)
    
        probas = probas**2
        probas = probas.mean(0)
        probas = probas**(1/2)

        for i in range(probas.shape[0]):
            proba = probas[i,:]
            for j in range(CFG.num_classes):
                bird = CFG.id2bird[j]
                predict_dict[bird].append(proba[j])

    newdf = pd.DataFrame(predict_dict)
    newdf.to_csv('final_test/fixed_label.csv')


    
    




    




    




 


    




