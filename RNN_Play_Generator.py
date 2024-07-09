import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

import keras 
import tensorflow as tf

#Load the data
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#Read the file contents

#Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#length of text is the number of characters in it
print('Length of text: {} characters'.format(len(text)))


#Encoding

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}