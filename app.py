import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import whisper
from pathlib import Path
from dotenv import load_dotenv
from keras.models import load_model
from flask import Flask, request, jsonify
from transformers import pipeline
from happytransformer import HappyTextToText

from uploadFile import *
from modelDownload import *
from modelFluency import *
from modelGrammar import *
from modelLexical import *
# from modelPronun import *

load_dotenv()

app = Flask(__name__)

# Fluency Model Path
pathFluency = os.environ.get("FLUENCY_MODEL_NAME")
fluency_file = Path(pathFluency)
if not fluency_file.is_file():
    # file exists check
    downloadModel(os.environ.get("FLUENCY_MODEL_STORAGE"), dest_folder="./models/")
    print("Fluency Model Already Downloaded")
else:
    print("Fluency Model File Is Exists")

# Fluency Load Model
loaded_model_fluency = load_model(pathFluency)
print("Load Fluency Model Finished")

# Lexical Model Path
lexical_model_h5_dir = os.environ.get("LEXICAL_MODEL_NAME")
lexical_model_file = Path(lexical_model_h5_dir)
if not lexical_model_file.is_file():
    # file exists check
    downloadModel(os.environ.get("LEXICAL_MODEL_STORAGE"), dest_folder="./models/")           # WAITING UPDATE
    print("Lexical Model Already Downloaded")
else:
    print("Lexical Model File Is Exists")

# Lexical Tokenizer Path
lexical_tokenizer_dir = os.environ.get("LEXICAL_TOKENIZER_NAME")
lexical_token_file = Path(lexical_tokenizer_dir)
if not lexical_token_file.is_file():
    # file exists check
    downloadModel(os.environ.get("LEXICAL_TOKENIZER_STORAGE"), dest_folder="./models/")   # WAITING UPDATE
    print("Lexical Pickle Already Downloaded")
else:
    print("Lexical Pickle Is Exists")

# Lexical Load Model and Tokenizer
model, tokenizer, asr_model = load_model_and_tokenizer(lexical_model_h5_dir, lexical_tokenizer_dir)
print("Load Lexical Model Finished")

# Grammar Load Whisper
grammar_whisper_model = whisper.load_model("base.en")
print("Grammar Load Whisper Finished")

# Grammar Load Model from HappyTransformer
hafid_happy_t5 = HappyTextToText("T5", "hafidikhsan/IELTS-GEC-T5-C4_200M-125k")
print("Load Grammar Model Finished")

# Pronun Load Model
pronunciation_model = "hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation"

@app.route("/")
def home():
    return "Hello, This is IELTS API Docker Master v.2.3 - (Fluency, Grammar, Lexical Update)"

@app.route("/upload", methods=["POST"])
def upload():
    print("This is Upload IELTS API Docker Master v.2.3 - (Fluency, Grammar, Lexical Update)")

    file_to_upload = request.files['file']
    uploadLink = uploadFile(file_to_upload)
    audioUrl = (uploadLink['secure_url'])

    # Fluency Prediction
    datasetcheck, extracted_features = feature_extraction(audioUrl)
    fluencyBand = fluency_calculation(loaded_model_fluency, datasetcheck, extracted_features)

    # Grammar Prediction
    evaluation = GrammarEval()
    band_eval = grammar(hafid_happy_t5, grammar_whisper_model, audioUrl, evaluation)
    grammarBand = to_level(band_eval)
    # asr = to_level(band_eval)

    # Lexical Prediction
    lexicalBand = lexical_calculation(audioUrl, model, tokenizer, asr_model)

    # Prediction Sample
    classifier = pipeline("audio-classification", model = pronunciation_model)
    pronunciationBand = classifier(audioUrl)

    # Delete Audio After Prediction
    cloudinary.uploader.destroy(uploadLink['public_id'], resource_type="video")

    return jsonify({"Fluency Band": fluencyBand,
                    "Grammar Band": grammarBand,
                    "Lexical Band": lexicalBand,
                    "Pronunciation Band": pronunciationBand})

# Deploy debug must delete
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))