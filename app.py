import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import whisper
import datetime
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

load_dotenv()

app = Flask(__name__)

# Fluency Model Path
print("Start Fluency Model Check")
MC_fluency_start = datetime.datetime.now()

pathFluency = os.environ.get("FLUENCY_MODEL_NAME")
fluency_file = Path(pathFluency)
if not fluency_file.is_file():
    # file exists check
    downloadModel(os.environ.get("FLUENCY_MODEL_STORAGE"), dest_folder="./models/")
    print("Fluency Model Already Downloaded")
else:
    print("Fluency Model File Is Exists")

MC_fluency_end = datetime.datetime.now()
print("Fluency Model Check Finished")
MC_fluency = (MC_fluency_end - MC_fluency_start).total_seconds()

# Fluency Load Model
print("Start Fluency Load Model")
LM_fluency_start = datetime.datetime.now()

loaded_model_fluency = load_model(pathFluency)

LM_fluency_end = datetime.datetime.now()
print("Fluency Load Model Finished")
LM_fluency = (LM_fluency_end - LM_fluency_start).total_seconds()

# Lexical Model Path
print("Start Lexical Model Check")
MC_lexical_start = datetime.datetime.now()

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

MC_lexical_end = datetime.datetime.now()
print("Lexical Model Check Finished")
MC_lexical = (MC_lexical_end - MC_lexical_start).total_seconds()

# Lexical Load Model and Tokenizer
print("Start Lexical Load Model")
LM_lexical_start = datetime.datetime.now()

model, tokenizer, asr_model = load_model_and_tokenizer(lexical_model_h5_dir, lexical_tokenizer_dir)

LM_lexical_end = datetime.datetime.now()
print("Lexical Load Model Finished")
LM_lexical = (LM_lexical_end - LM_lexical_start).total_seconds()

# Grammar Load Model Whisper & Happy Transformer
print("Start Grammar Load Model")
LM_grammar_start = datetime.datetime.now()

grammar_whisper_model = whisper.load_model("base.en")
hafid_happy_t5 = HappyTextToText("T5", "hafidikhsan/IELTS-GEC-T5-C4_200M-125k")

LM_grammar_end = datetime.datetime.now()
print("Grammar Load Model Finished")
LM_grammar = (LM_grammar_end - LM_grammar_start).total_seconds()

# Pronun Load Model
print("Start Pronunciation Load Model")
LM_pronun_start = datetime.datetime.now()

pronunciation_model = "hafidikhsan/Wav2vec2-large-robust-Pronounciation-Evaluation"
classifier = pipeline("audio-classification", model=pronunciation_model)

LM_pronun_end = datetime.datetime.now()
print("Pronunciation Load Model Finished")
LM_pronun = (LM_pronun_end - LM_pronun_start).total_seconds()

model_process = (MC_fluency + LM_fluency + MC_lexical + LM_lexical + LM_grammar + LM_pronun)

print("Fluency Check Model Process  :", MC_fluency, "s")
print("Fluency Load Model Process   :", LM_fluency, "s")
print("Lexical Check Model Process  :", MC_lexical, "s")
print("Lexical Load Model Process   :", LM_lexical, "s")
print("Grammar Load Model Process   :", LM_grammar, "s")
print("Pronun Load Model Process    :", LM_pronun, "s")
print("All Model Process          :", model_process, "s")

@app.route("/")
def home():
    return "Hello, This is IELTS API Docker Master v.2.4 - (Complete Model Update)"

@app.route("/upload", methods=["POST"])
def upload():
    
    print("Start Upload File")
    upload_start = datetime.datetime.now()
    file_to_upload = request.files['file']
    uploadLink = uploadFile(file_to_upload)
    audioUrl = (uploadLink['secure_url'])
    upload_end = datetime.datetime.now()
    print("Upload File Finished - Sent Url Link")

    upload_process = (upload_end - upload_start).total_seconds()

    # Fluency Prediction
    print("Start Fluency Prediction")
    fluency_start = datetime.datetime.now()
    datasetcheck, extracted_features = feature_extraction(audioUrl)
    fluencyBand = fluency_calculation(loaded_model_fluency, datasetcheck, extracted_features)
    fluency_end = datetime.datetime.now()
    print("Fluency Prediction Finished")

    fluency_process = (fluency_end - fluency_start).total_seconds()

    # Grammar Prediction
    print("Start Grammar Prediction")
    grammar_start = datetime.datetime.now()
    evaluation = GrammarEval()
    band_eval = grammar(hafid_happy_t5, grammar_whisper_model, audioUrl, evaluation)
    grammarBand, asr = to_level(band_eval)
    grammar_end = datetime.datetime.now()
    print("Finishing Grammar Prediction Finished")

    grammar_process = (grammar_end - grammar_start).total_seconds()

    # Lexical Prediction
    print("Start Lexical Prediction")
    lexical_start = datetime.datetime.now()
    lexicalBand = lexical_calculation(audioUrl, model, tokenizer, asr_model)
    lexical_end = datetime.datetime.now()
    print("Finishing Lexical Prediction Finished")    

    lexical_process = (lexical_end - lexical_start).total_seconds()
    
    # Prediction Sample
    print("Start Pronunciation Prediction")
    pronun_start = datetime.datetime.now()
    predict = classifier(audioUrl)
    pronunciationBand = list(predict[0].values())
    if pronunciationBand[1] == "proficient":
        pronunciationBand = 9
    elif pronunciationBand[1] == "advanced":
        pronunciationBand = 7
    elif pronunciationBand[1] == "intermediate":
        pronunciationBand = 5
    elif pronunciationBand[1] == "beginer":
        pronunciationBand = 3
    else:
        pronunciationBand = 0
    pronun_end = datetime.datetime.now()
    print("Finishing Pronunciation Prediction Finished") 

    pronun_process = (pronun_end - pronun_start).total_seconds()

    # Delete Audio After Prediction
    print("Start Delete File")
    delete_start = datetime.datetime.now()
    cloudinary.uploader.destroy(uploadLink['public_id'], resource_type="video")
    delete_end = datetime.datetime.now()
    print("Delete File Finished")

    delete_process = (delete_end - delete_start).total_seconds()

    predict_process = (upload_process + fluency_process + grammar_process + lexical_process + pronun_process + delete_process)

    print("Upload Process   :", upload_process, "s")
    print("Fluency Process  :", fluency_process, "s")
    print("Grammar Process  :", grammar_process, "s")
    print("Lexical Process  :", lexical_process, "s")
    print("Pronun Process   :", pronun_process, "s")
    print("Delete Process   :", delete_process, "s")
    print("All Process      :", predict_process, "s")

    return jsonify({"Fluency Band": fluencyBand,
                    "Grammar Band": grammarBand,
                    "Lexical Band": lexicalBand,
                    "Pronunciation Band": pronunciationBand,
                    "Transcript": asr})

# Deploy debug must delete
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))