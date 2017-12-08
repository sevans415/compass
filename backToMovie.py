#!/Users/spencerevans/tensorflow/bin/python

import treeUtil 
import cPickle



# LSTM with dropout for sequence classification in the IMDB dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# load IBC Data

import glob
import os

pos_file_list = glob.glob(os.path.join(os.getcwd(), "../Downloads/Train/pos", "*.txt"))
neg_file_list = glob.glob(os.path.join(os.getcwd(), "../Downloads/Train/neg", "*.txt"))
test_file_list = glob.glob(os.path.join(os.getcwd(), "../Downloads/Test", "*.txt"))

pos = []
neg = []
test = []
for file_path in pos_file_list:
    with open(file_path) as f_input:
        pos.append(f_input.read())
        
for file_path in neg_file_list:
    with open(file_path) as f_input:
        neg.append(f_input.read())

for file_path in test_file_list:
    with open(file_path) as f_input:
        test.append(f_input.read())


n1 = len(pos)
words1 = np.full(n1, None)
label1 = np.full(n1, 0)
for i, review in enumerate(pos):
    words1[i] = review

n2 = len(neg)
words2 = np.full(n2, None)
label2 = np.full(n2, 1)
for i, review in enumerate(neg):
    words2[i] = review
    
n3 = len(test)
testWords = np.full(n3, None)
for i, review in enumerate(test):
    testWords[i] = review
    
pos_df = pd.DataFrame(data = {'words':words1, 'label':label1})
neg_df = pd.DataFrame(data = {'words':words2, 'label':label2})
frames = [pos_df, neg_df]
data = pd.concat(frames)


tokenizer = Tokenizer()
tokenizer.fit_on_texts(data["words"].values)
vocab_size = len(tokenizer.word_index) + 1

X_train = data['words']
y_train = data['label']
X_test = testWords

# X_train, X_test, y_train, y_test = train_test_split(data['words'], data['label'], test_size=0.3, random_state=8)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)


top_words = 5000

# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# load the whole embedding into memory
# GLOVE SHITTT
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
embedding_matrix = np.zeros((vocab_size, 100))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector



# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
embedding = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=max_review_length, trainable=False)
model.add(embedding)
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())
model.fit(X_train, y_train, epochs=20, batch_size=64)
# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
preds = model.predict(X_test)
output_preds = pd.DataFrame(data = preds)
output_preds.to_csv("kaggle_preds.csv")

print("Accuracy: %.2f%%" % (scores[1]*100))
print("with CNN")




