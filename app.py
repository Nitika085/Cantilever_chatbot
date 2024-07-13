from flask import Flask, request, jsonify,render_template

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
import nltk
import numpy as np
import json
import re

nltk.download('punkt')

app = Flask(__name__)

# Load the model
chatbot_model_path = 'chatbot_model.h5'
chatbot_model = load_model(chatbot_model_path)

# Load the tokenizer
with open('tokenizer.json') as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Sample data
conversations = [
    "Hello, how can I help you?",
    "Hi, I need some assistance.",
    "Sure, what do you need help with?",
    "I am looking for a new phone.",
    "We have a wide range of phones. What is your budget?",
    "My budget is around $300."
]

# Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Tokenize and vectorize the sample data
sequences = tokenizer.texts_to_sequences(conversations)
data = pad_sequences(sequences, maxlen=10, padding='post')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')

    # Preprocess user input
    user_input = preprocess_text(user_input)
    sequence = tokenizer.texts_to_sequences([user_input])
    tokenized_input = pad_sequences(sequence, maxlen=10, padding='post')

    # Predict response
    prediction = chatbot_model.predict(tokenized_input)
    response_index = np.argmax(prediction)
    
    # For simplicity, we just return one of the sample responses
    response = conversations[response_index % len(conversations)]

    return render_template('index.html',response=response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    







