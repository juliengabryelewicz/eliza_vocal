import librosa
import numpy as np

def extract_feature(data, **kwargs):
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    mfcc = kwargs.get("mfcc")
    audio_data = np.frombuffer(data, dtype=np.int16).astype('float32')
    sample_rate = 16000
    if chroma or contrast:
        stft = np.abs(librosa.stft(audio_data))
    result = np.array([])
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(audio_data, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    return result