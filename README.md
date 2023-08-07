# BirdCLEF2022
bird sound detection <br/>
for Kaggle competition https://www.kaggle.com/c/birdclef-2022 <br/>
rank 80/807 <br/>

# Solution 

## Data: <br/>
original audio: https://www.kaggle.com/competitions/birdclef-2022/data <br/>
external audio: freefield1010 <br/>
split the given long audio into smaller clips and convert them to Mel spectrum for model training <br/>

## Model: 
sound event detection(SED) (Also try Audio transformer model, but can not get better performance) <br/>
backbone: time-efficient net b0 <br/>
loss: BCE loss + focal loss

## Data augmentation:        <br/>
noise injection:          Gaussian noise, pink noise, brown noise, external noise(noise from freefield1010) <br/>
volume scaler:            random transform the original volume <br/>
mix with other audio:     cut mix, linear mix <br/>
others:                   time stretch, pitch shift, <br/>

## Data optimization:            <br/>
Firstly, train a base model on large audio clips since their labels are closer to the given labels. Then use them to correct the first and secondary labels of smaller audio clips. <br/>
Finally, train the final model on optimized datasets.







