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
    print_info(f"predict {url} using {model_name}")
    
    model_path = f"/app/models/{model_name}"
    #print_info("model name: " + model_name)
        
    model = load_model(model_path)
    #print_info(f"load model")

    # Get the max_sequence_length from the model
    max_sequence_length = 148
    print_info(f"max_sequence_length: {max_sequence_length}")
    
    cleaned_tokens = clean_and_normalize(tokenize_url(url))
    print_info(f"cleaned_tokens: {cleaned_tokens}")
    
    tokens = await read_tokens()

    # Use the existing tokenizer if available; otherwise, create a new one
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = Tokenizer()
        
        tokenizer.fit_on_texts(tokens)
        print('USER INFO:    Found %s unique tokens.' % len(tokenizer.word_index))
        
        tokenizer.fit_on_texts(cleaned_tokens)
        print('USER INFO:    Found %s unique tokens.' % len(tokenizer.word_index))

    sequences = tokenizer.texts_to_sequences(cleaned_tokens)
    print_info(f"sequences: {sequences}")

    concatenated_sequence = [item for sublist in sequences for item in sublist]
    #print([concatenated_sequence]) 

    X_padded = pad_sequences([concatenated_sequence], maxlen=max_sequence_length)
    # print("X_padded: ", X_padded)
    #print("X_padded dtype: ", X_padded.dtype)
    #sys.stdout.flush()
    del tokenizer

    # Await the prediction result
    prediction = model.predict(X_padded)
    print_info(f"prediction: {prediction}")
    # print("prediction dtype: ", prediction.dtype)
    # sys.stdout.flush()
    
    predicted_class = np.argmax(prediction)
    # print("predicted class: ", predicted_class)
    # sys.stdout.flush()
    
    class_name = class_names[predicted_class]
    print_info(f"class_name: {class_name}")
    
    probability = np.max(prediction) * 100
    print_info(f"probability: {probability}")
    
    return {"status": "success", "predicted_class": class_name, "accuracy": probability}

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

async def read_tokens():
    word_index = {}
    with open('src/tokens.txt', 'r') as file:
        for line in file:
            word, index = line.strip().split(': ')
            word_index[word] = int(index)
            
    return word_index

def print_info(msg):
    print("USER INFO:    " + msg)