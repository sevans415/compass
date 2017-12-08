#!/Users/spencerevans/tensorflow/bin/python

import treeUtil 
import cPickle

import numpy as np
import pandas as pd

import rnnClass

from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import train_test_split
import itertools
from random import *



# load IBC Data
data_source = '../Downloads/ibc3.xlsx'
data = pd.read_excel(data_source)
print data_source


# prepare the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["words"].values)
vocab_size = len(tokenizer.word_index) + 1

# break into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(data['words'].values, data['label'].values, test_size=0.2)

# tokenize all of the words, int encodes each of them
encoded_texts = tokenizer.texts_to_sequences(data["words"].values)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# iterate through all sentences and find the max length
# we will pad all other sentences to be of that length
# this allows to take in a sentence of any length
max_sentence_length = 0
for item in data['words'].values:
    length = len(item.split(" "))
    if length > max_sentence_length:
        max_sentence_length = length
padded_texts = sequence.pad_sequences(encoded_texts, maxlen=max_sentence_length, padding='post')

# load in the GloVe vectors and store them as a dict mapping word to their vector
embeddings_index = dict()
f = open('../Downloads/Glove.6B/glove.6B.100d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


# create a weight matrix for words in training docs
# create a matrix that stores the GloVevector for each word in the data we are considering
# we will pass this into our model
embedding_matrix = np.zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[index, :] = embedding_vector


# create the keras GloVe Vector word embedding that we pass directly into the model
embedding = Embedding(vocab_size, 100, input_length=max_sentence_length, trainable=False)
embedding.build((None,))
embedding.set_weights([embedding_matrix])



# fix random seed for reproducibility
np.random.seed(7)

# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)



# initialize the params and param names we want to examine
params = [
    [1],                                            # epochs
    [8, 16, 32, 64, 128],                                       # batch_size
    [0.1, 0.3],                                                 # validation_split
    ['mean_squared_error', 'binary_crossentropy'],              # loss
    [0.1, 0.2, 0.4],                                            # dropout
    [0, 1],                                                     # opt_flag
    [0.001, 0.01, 0.1],                                         # opt_lr
    [0, 1],                                                     # cnn_flag
    [4, 8, 16, 32, 64],                                         # kernel_size
    [1]                                                         # lstm_flag
]

param_names = ["epochs", "batch_size", "validation_split", "loss", "dropout", 
                "opt_flag", "opt_lr", "cnn_flag", "kernel_size", "lstm_flag"]


# file name to identify the excel output later
file_name = "crossVal on " + data_source

# run the randomized grid search
rnnClass.randomizedGridSearch(params, param_names, 20, X_train, y_train, 
                        X_test, y_test, embedding, file_name)








