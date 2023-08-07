import numpy as np
import colorednoise as cn
import librosa
import random
from data_setting import CFG,AudioParams

# wrap audio transform methods
class AudioTransform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y: np.ndarray, sr):
        if self.always_apply:
            return self.apply(y, sr=sr)
        else:
            if np.random.rand() < self.p:
                return self.apply(y, sr=sr)
            else:
                return y

    def apply(self, y: np.ndarray, **params):
        raise NotImplementedError

# minmax normalize
class Normalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        max_vol = np.abs(y).max()
        y_vol = y * 1 / max_vol
        return np.asfortranarray(y_vol)

# normalize towards guassian distribution
class NewNormalize(AudioTransform):
    def __init__(self, always_apply=False, p=1):
        super().__init__(always_apply, p)

    def apply(self, y: np.ndarray, **params):
        y_mm = y - y.mean()
        return y_mm / y_mm.abs().max()

# noise injection methods
class NoiseInjection(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_noise_level=0.5):
        super().__init__(always_apply, p)

        self.noise_level = (0.0, max_noise_level)

    def apply(self, y: np.ndarray, **params):
        noise_level = np.random.uniform(*self.noise_level)
        noise = np.random.randn(len(y))
        augmented = (y + noise * noise_level).astype(y.dtype)
        return augmented


class GaussianNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        white_noise = np.random.randn(len(y))
        a_white = np.sqrt(white_noise ** 2).max()
        augmented = (y + white_noise * 1 / a_white * a_noise).astype(y.dtype)
        return augmented


class PinkNoise(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, min_snr=5, max_snr=20):
        super().__init__(always_apply, p)

        self.min_snr = min_snr
        self.max_snr = max_snr

    def apply(self, y: np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y ** 2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        pink_noise = cn.powerlaw_psd_gaussian(1, len(y))
        a_pink = np.sqrt(pink_noise ** 2).max()
        augmented = (y + pink_noise * 1 / a_pink * a_noise).astype(y.dtype)
        return augmented


class BrownNoise(AudioTransform):
    def __init__(self, always_apply = False, p=0.5, min_snr = 5, max_snr = 20):
        super().__init__(always_apply,p)

        self.min_snr = min_snr
        self.max_snr = max_snr
    
    def apply(self, y:np.ndarray, **params):
        snr = np.random.uniform(self.min_snr, self.max_snr)
        a_signal = np.sqrt(y**2).max()
        a_noise = a_signal / (10 ** (snr / 20))

        brown_noise = cn.powerlaw_psd_gaussian(2, len(y))
        a_brown = np.sqrt(brown_noise ** 2).max()
        augmented = (y + brown_noise * 1 / a_brown * a_noise).astype(y.dtype)

        return augmented

# load noisy data from freefield dataset and add them to 
class ExternalNoise(AudioTransform):
    def __init__(self, noise_df, always_apply = False, p = 0.5, min_snr = 5, max_snr = 20):
        super().__init__(always_apply,p)
        self.df = noise_df
        self.noise_path = '../freefield/'

        self.min_snr = min_snr
        self.max_snr = max_snr
    
    def apply(self, y:np.ndarray, **params):
        #snr = np.random.uniform(self.min_snr, self.max_snr)
        #a_signal = np.sqrt(y**2).max()
        #a_noise = a_signal / (10 ** (snr / 20))

        idx = np.random.randint(len(self.df))
        noise_path =self.noise_path + self.df.loc[idx,:].filepath
        
        # try 10 seconds * 16000 sample_rate = 160000 samples
        external_noise,_ = librosa.load(noise_path, sr = 32000)
        external_noise = external_noise[:len(y)]
        #a_external = np.sqrt(external_noise ** 2).max()
        #augmented = (y + external_noise * 1 / a_external * a_noise).astype(y.dtype)
        augmented = (y + external_noise).astype(y.dtype)

        return augmented
        






    
# famous audio augmentation skills 
# pitchshift timestretch timeshift
class PitchShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_range=5):
        super().__init__(always_apply, p)
        self.max_range = max_range

    def apply(self, y: np.ndarray, sr, **params):
        n_steps = np.random.randint(-self.max_range, self.max_range)
        augmented = librosa.effects.pitch_shift(y, sr = sr, n_steps = n_steps)
        return augmented

class TimeStretch(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_rate=1.2):
        super().__init__(always_apply, p)
        self.max_rate = max_rate

    def apply(self, y: np.ndarray, **params):
        rate = np.random.uniform(0, self.max_rate)
        augmented = librosa.effects.time_stretch(y, rate = rate)
        return augmented

class TimeShift(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, max_shift_second=2, sr=32000, padding_mode="replace"):
        super().__init__(always_apply, p)
    
        assert padding_mode in ["replace", "zero"], "`padding_mode` must be either 'replace' or 'zero'"
        self.max_shift_second = max_shift_second
        self.sr = sr
        self.padding_mode = padding_mode

    def apply(self, y: np.ndarray, **params):
        shift = np.random.randint(-self.sr * self.max_shift_second, self.sr * self.max_shift_second)
        augmented = np.roll(y, shift)
        if self.padding_mode == "zero":
            if shift > 0:
                augmented[:shift] = 0
            else:
                augmented[shift:] = 0
        return augmented

def _db2float(db: float, amplitude=True):
    if amplitude:
        return 10 ** (db / 20)
    else:
        return 10 ** (db / 10)

# volume convert augmenttation
def volume_down(y: np.ndarray, db: float):
    """
    Low level API for decreasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to decrease
    Returns
    -------
    applied: numpy.ndarray
        audio with decreased volume
    """
    applied = y * _db2float(-db)
    return applied


def volume_up(y: np.ndarray, db: float):
    """
    Low level API for increasing the volume
    Parameters
    ----------
    y: numpy.ndarray
        stereo / monaural input audio
    db: float
        how much decibel to increase
    Returns
    -------
    applied: numpy.ndarray
        audio with increased volume
    """
    applied = y * _db2float(db)
    return applied


class RandomVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        if db >= 0:
            return volume_up(y, db)
        else:
            return volume_down(y, db)


class CosineVolume(AudioTransform):
    def __init__(self, always_apply=False, p=0.5, limit=10):
        super().__init__(always_apply, p)
        self.limit = limit

    def apply(self, y: np.ndarray, **params):
        db = np.random.uniform(-self.limit, self.limit)
        cosine = np.cos(np.arange(len(y)) / len(y) * np.pi * 2)
        dbs = _db2float(cosine * db)
        return y * dbs

def random_power(images, power = 1.5, c= 0.7):
    
    images = images - images.min()
    images = images/(images.max() + 1e-12)
    images = images**(random.random()*power + c)

    return images

def bandpass_noise(images , level_noise = 0.05):    
    # image shape [224,313]
    a = random.randint(0, CFG.n_mels // 2)
    b = random.randint(a + 20 , CFG.n_mels)

    images[a:b, :] = images[a:b,:] + (np.random.sample((b-a,CFG.len_check)).astype(np.float32)+9) * 0.05 * images.mean() * level_noise  * (np.random.sample() + 0.3)

    return images

def lower_uper(images):
    images = images - images.min()
    r = random.randint(CFG.n_mels//2 , CFG.n_mels)
    x = random.random()/2
    pink_noise = np.array([np.concatenate((1-np.arange(r)*x/r,np.zeros(CFG.n_mels-r)-x+1))]).T
    images = images*pink_noise
    images = images/(images.max() + 1e-12)

    return images
