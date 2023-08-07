# birdclef2022
bird sound detection
for Kaggle competition https://www.kaggle.com/c/birdclef-2022
rank 80/807

# solution 

input data: 
original audio: https://www.kaggle.com/competitions/birdclef-2022/data <br/>
external audio: freefield1010 <br/>

model: sound event detection(SED) (Also try Audio transformer model, but can not get better performance)
backbone: time-efficient net b0
loss: BCE loss + focal loss

data augmentation:        <br/>
noise injection:          Gaussian noise, pink noise, brown noise, external noise(noise from freefield1010)
volume scaler:            random transform the original volume
mix with other audio:     cut mix, linear mix
others:                   time stretch, pitch shift, 



