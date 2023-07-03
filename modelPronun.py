from transformers import pipeline

audioUrl = "https://res.cloudinary.com/dntqqcuci/video/upload/v1687939594/Citeureup_2_cgvbar.wav"

# Pronunciation Load Model
pronunciation_model = "hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation"
classifier = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")

# Pronunciation Prediction
predict = classifier(audioUrl)
pronunciationBand = list(predict[0].values())

def prediction():
    if pronunciationBand[1] == "proficient":
        return 9
    elif pronunciationBand[1] == "advanced":
        return 7
    elif pronunciationBand[1] == "intermediate":
        return 5
    elif pronunciationBand[1] == "beginer":
        return 3
    else:
        return 0
    
print(prediction())
    
