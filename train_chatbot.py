import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import json
import re

# Sample training data
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

# Tokenize and vectorize the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(conversations)
sequences = tokenizer.texts_to_sequences(conversations)
word_index = tokenizer.word_index

# Save the tokenizer
with open('tokenizer.json', 'w') as f:
    json.dump(tokenizer.to_json(), f)

# Pad sequences
max_len = 10  # Adjust based on your data
data = pad_sequences(sequences, maxlen=max_len, padding='post')

# Create labels (dummy labels for this example)
labels = np.zeros((len(data), len(word_index) + 1))  # Number of classes is the vocab size
for i in range(len(labels)):
    labels[i][i % (len(word_index) + 1)] = 1  # Dummy labels for example purposes

# Define the model
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=100, input_length=max_len))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(len(word_index) + 1, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(data, labels, epochs=10, batch_size=1)

# Save the model
model.save('chatbot_model.h5')
