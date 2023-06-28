from transformers import pipeline

# Predict
pronunciation_model = "hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation"
classifier = pipeline("audio-classification", model="hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation")