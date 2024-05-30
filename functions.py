import re
import string
import numpy as np
from urllib.parse import urlparse
from urllib.parse import urlparse, parse_qs
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

async def predict(url: str, model_name: str):
    print("predict func"+url +"using" + model_name)
    
    model_path = f"/app/models/{model_name}"
    print("model name: " + model_name)
        
    model = tf.keras.models.load_model(model_path)

    # Get the max_sequence_length from the model
    max_sequence_length = model.layers[0].input_shape[1]

    # Tokenize and preprocess the input
    tokens = tokenize_url(url)  # Make sure tokenize_url is defined and returns a list of tokens
    cleaned_tokens = clean_and_normalize(tokens)  # Make sure clean_and_normalize is defined

    # Use the existing tokenizer if available; otherwise, create a new one
    if 'tokenizer' not in globals():
        global tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(cleaned_tokens)

    sequences = tokenizer.texts_to_sequences(cleaned_tokens)
    X_padded = pad_sequences(sequences, maxlen=max_sequence_length)

    # Await the prediction result
    prediction = model.predict(np.array([X_padded[0]]))
    print("prediction: " + prediction)

    return prediction

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
