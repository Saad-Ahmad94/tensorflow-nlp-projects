import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

from keras.datasets import imdb
import tensorflow as tf 
import keras
from keras.utils.data_utils import pad_sequences

VOCAB_SiZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

#Load the dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words= VOCAB_SiZE)

#Preprocessing to pad the input and test data to get same length 
train_data = pad_sequences(train_data, MAXLEN)
test_data = pad_sequences(test_data, MAXLEN)

#Create the model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SiZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation='sigmoid')
], name= "LSTM Network")


#Train the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

#Model Evaluation
result = model.evaluate(test_data, test_labels)
print(result)


#Making predictions

word_index = imdb.get_word_index()

def encode_text(text):
    tokens = tf.keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return pad_sequences([tokens], MAXLEN)[0]

text = "that was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)


def predict(text):
        encoded_text = encode_text(text)
        pred = np.zeros((1,250))
        pred[0] = encode_text
        result = model.predict(pred)
        print(result[0])

postive_review = "That movie was so awesome! I really loved it and would watch it again because it was amazingly great"
predict(postive_review)

negative_reviw = "that movie sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_reviw)