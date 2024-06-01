import re
import sys
import string
import numpy as np
from urllib.parse import urlparse
from urllib.parse import urlparse, parse_qs
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class_names = ['Benign', 'Defacement', 'Phishing', 'Malware']

async def predict(url: str, model_name: str):
    print("predict func "+url +" using " + model_name)
    sys.stdout.flush()
    
    model_path = f"/app/models/{model_name}"
    print("model name: " + model_name)
    sys.stdout.flush()
        
    model = load_model(model_path)
    print("load model")
    sys.stdout.flush()

    # Get the max_sequence_length from the model
    max_sequence_length = model.layers[0].input_shape[1]
    print("max_sequence_length: ", max_sequence_length)
    sys.stdout.flush()

    # Tokenize and preprocess the input
    tokens = tokenize_url(url)
    print("tokens: ", tokens)
    sys.stdout.flush()
    
    cleaned_tokens = clean_and_normalize(tokens)
    print("cleaned_tokens: ", cleaned_tokens)
    sys.stdout.flush()

    # Use the existing tokenizer if available; otherwise, create a new one
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(cleaned_tokens)

    sequences = tokenizer.texts_to_sequences(cleaned_tokens)
    print("sequences: ", sequences)
    sys.stdout.flush()

    X_padded = pad_sequences([sequences], maxlen=max_sequence_length)
    print("X_padded: ", X_padded)
    print("X_padded dtype: ", X_padded.dtype)
    sys.stdout.flush()

    # Await the prediction result
    prediction = model.predict(X_padded)
    print("prediction: ", prediction)
    print("prediction dtype: ", prediction.dtype)
    sys.stdout.flush()
    
    predicted_class = np.argmax(prediction)
    print("predicted class: ", predicted_class)
    sys.stdout.flush()
    
    class_name = class_names[predicted_class]
    print("class_name: " + class_name)
    sys.stdout.flush()
    
    probability = np.max(prediction) * 100
    print("probability: ", probability)
    sys.stdout.flush()
    
    return {"status": "success", "predicted_class": int(predicted_class), "accuracy": probability}

def tokenize_url(url):
    parsed_url = urlparse(url)
    scheme = parsed_url.scheme
    domain = parsed_url.netloc
    path = parsed_url.path
    scheme_tokens = re.findall(r'\w+', scheme) if scheme else []
    domain_tokens = re.findall(r'\w+', domain) if domain else []
    path_tokens = re.findall(r'\w+', path) if path else []
    
    query_params = parse_qs(parsed_url.query)
    query_tokens = []
    for key, values in query_params.items():
        query_tokens.extend(re.findall(r'\w+', key))
        for value in values:
            query_tokens.extend(re.findall(r'\w+', value))
    tokens = scheme_tokens + domain_tokens + path_tokens + query_tokens
    return tokens

def clean_and_normalize(tokens):
    cleaned_tokens = []
    for token in tokens:
        token = token.lower()
        token = ''.join(char for char in token if char not in string.punctuation)
        cleaned_tokens.append(token)
    return cleaned_tokens
