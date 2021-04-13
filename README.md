# Eliza Vocal

An experimental project inspired by [ELIZA](https://en.wikipedia.org/wiki/ELIZA)

Works with Python 3.8 in French Language.

Many AI fields are used for this project : 

- Voice Recognition 
- Natural Language Understanding (NLU)
- Face Emotion Recognition (FER)
- Speech Emotion Recognition (SER)

NLU, FER and SER are combined to give the more adapted answer to the user.

## Modules used

- dlib
- kivy
- librosa
- openCV
- numpy
- pandas
- pyAudio
- snips_nlu
- vosk

## Install

You just need to install all Python modules in requirements.txt.

```
pip3 install -r requirements.txt
python3 main.py
```

Models are already created and loaded in the data folder.

## TO DO

- [ ] Add English for voice recognition and understanding
- [ ] Improve Accuracy
- [ ] Add better conversation
- [ ] Find a way to use less CPU (around 45% of my processor, which is a Intel Core i7-1065G7)