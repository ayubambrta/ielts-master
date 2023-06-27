import io
import pydub
import librosa
import numpy as np
from urllib.request import urlopen

def feature_extraction(file_name):
    wav = io.BytesIO()

    with urlopen(file_name) as r:
        r.seek = lambda *args: None  # allow pydub to call seek(0)
        pydub.AudioSegment.from_file(r).export(wav, "wav")

    wav.seek(0)
    X , sample_rate = librosa.load(wav)

    if X.ndim > 1:
        X = X[:,0]
    X = X.T
            
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=30).T, axis=0)
    rmse = np.mean(librosa.feature.rms(y=X).T, axis=0)
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0)
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0)
    number_of_features = 3 + 30
    datasetchecks = np.empty((0,number_of_features))
    extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])

    return datasetchecks, extracted_features

# Define get value
def fluency_calculation(model, datasetchecks, extracted_features):
    datasetcheck = np.vstack([datasetchecks, extracted_features])
    datasetcheckss = np.expand_dims(datasetcheck, axis=1)
    pred = model.predict(datasetcheckss)
    classFluency = np.argmax(pred, axis = 1).tolist()
    if classFluency[0] == 0:
        return 3
    elif classFluency[0] == 1:
        return 4
    elif classFluency[0] == 2:
        return 5
    elif classFluency[0] == 3:
        return 6.5
    elif classFluency[0] == 4:
        return 8
    elif classFluency[0] == 5:
        return 9
    else:
        return 0
