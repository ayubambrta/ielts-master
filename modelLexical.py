# Import library
import pickle
import whisper
import numpy as np
import tensorflow as tf
from keras.optimizers import SGD
from transformers import TFDistilBertModel, DistilBertConfig

# Define params
params = {
    "MAX_LENGTH": 512,
    "RANDOM_STATE": 42,
    "KERNEL_REGULARIZERS": 0.01,
    "LAYER_DROPOUT": 0.5,
    "DENSE_ACTIVATION": "tanh",
    "DENSE_BIAS": "zeros",
    "OUTPUT_ACTIVATION": "softmax",
    "LEARNING_RATE": 1e-4,
    }

# Define encode function
def encode_batch(tokenizer, texts, batch_size=16, max_length=512):
    input_ids = []
    attention_mask = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer.batch_encode_plus(batch,
                                             max_length=max_length,
                                             padding="max_length",
                                             truncation=True,
                                             return_attention_mask=True,
                                             return_token_type_ids=False
                                             )
        input_ids.extend(inputs["input_ids"])
        attention_mask.extend(inputs["attention_mask"])
        
    return tf.convert_to_tensor(input_ids), tf.convert_to_tensor(attention_mask)

# Define ASR function
def speech_to_text(model_base, audio_path):
    result = model_base.transcribe(audio_path)
    
    return result["text"]

# Define neural network function
def build_model_from_load(transformer, max_length=params["MAX_LENGTH"]):
    
    # Define weight
    weight_initializer = tf.keras.initializers.GlorotNormal(seed=params["RANDOM_STATE"]) 

    # Define regularizer
    l2_initializer = tf.keras.regularizers.l2(l=params["KERNEL_REGULARIZERS"]) 
    
    # Define input
    input_ids_layer = tf.keras.layers.Input(shape=(max_length,), 
                                            name="input_ids", 
                                            dtype="int32",
                                           )
    input_attention_layer = tf.keras.layers.Input(shape=(max_length,), 
                                                  name="input_attention", 
                                                  dtype="int32",
                                                 )
    # The output of the DistilBERT model is a tuple with the element at index 0
    # hidden-state output representation of the last model layer
    # with a tf.Tensor of size (batch_size, sequence_length, hidden_size=768).
    last_hidden_state = transformer.distilbert([input_ids_layer, input_attention_layer])[0]
    
    # We will use DistilBERT output for the [CLS] token located at index 0.
    # So we will do token splicing [CLS] which gives a 2D data.
    cls_token = last_hidden_state[:, 0, :]
    
    D1 = tf.keras.layers.Dropout(params["LAYER_DROPOUT"],
                                 seed=params["RANDOM_STATE"],
                                 name="Dropout_1",
                                )(cls_token)

    Dense1 = tf.keras.layers.Dense(128,
                                   activation=params["DENSE_ACTIVATION"],
                                   kernel_initializer=weight_initializer,
                                   kernel_regularizer=l2_initializer,
                                   bias_initializer=params["DENSE_BIAS"],
                                   name="Dense_1",
                                  )(D1)
    
    D2 = tf.keras.layers.Dropout(params["LAYER_DROPOUT"],
                                 seed=params["RANDOM_STATE"],
                                 name="Dropout_2",
                                )(Dense1)
                                
    output = tf.keras.layers.Dense(6, 
                                   activation=params["OUTPUT_ACTIVATION"],
                                   kernel_regularizer=l2_initializer,
                                   kernel_initializer=weight_initializer,
                                   bias_initializer=params["DENSE_BIAS"],
                                   name="Output",
                                  )(D2)
    
    # Define model
    model = tf.keras.Model([input_ids_layer, input_attention_layer], output, name="MODEL_NAME")
    
    # Compile model
    model.compile(loss="categorical_crossentropy",
                  optimizer=SGD(learning_rate=params["LEARNING_RATE"], momentum=0.003),
                  metrics=["accuracy"],
                 )
    
    return model

# Define load model and tokenizer
def load_model_and_tokenizer(model_path, token_path):
    # Load model
    # Distil bert config
    config = DistilBertConfig(dropout=0.5, 
                              attention_dropout=0.5, 
                              output_hidden_states=True,
                            )

    # Init DistilBERT model
    distilBERT_load = TFDistilBertModel.from_pretrained("distilbert-base-uncased", config=config)

    # Load weight
    model_load = build_model_from_load(distilBERT_load)
    model_load.load_weights(model_path)

    # Load tokenizer
    with open(token_path, "rb") as handle:
        load_tokenizer = pickle.load(handle)

    # Initialize ASR
    asr_model = whisper.load_model("base.en")

    return model_load, load_tokenizer, asr_model

# Define get lexical value
def lexical_calculation(wav, model, tokenizer, asr_model):
    # Speect to text
    asr = speech_to_text(asr_model, wav)

    # Data test to list
    asr_list = [asr]

    # Tokenizer
    token_ids, token_attention = encode_batch(tokenizer, asr_list)

    # print(token_ids.shape)

    # Predict
    pred = model.predict([token_ids, token_attention])
    classes = np.argmax(pred, axis=1)

    # To band convert
    if classes[0]==0:
        return 3
    elif classes[0]==1:
        return 4
    elif classes[0]==2:
        return 5
    elif classes[0]==3:
        return 6.5
    elif classes[0]==4:
        return 8
    else:
        return 9
