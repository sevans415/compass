#!/Users/spencerevans/tensorflow/bin/python

import treeUtil 
import cPickle



# LSTM with dropout for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer


# load IBC Data

[lib, con, neutral] = cPickle.load(open('fall2017/cs182/compass/ibcData.pkl', 'rb'))

lib_list = [tree.get_words() for tree in lib]
con_list = [tree.get_words() for tree in con]
neutral_list = [tree.get_words() for free in neutral]
all_text = lib_list + neutral_list + con_list

labels = [0] * len(lib_list) + [1] * len(neutral_list) + [2] * len(con_list)

# prepare the tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_text)
vocab_size = len(tokenizer.word_index) + 1

# int encode the texts
encoded_texts = tokenizer.texts_to_sequences(all_text)
print encoded_texts[0]

# break into training/testing sets
train_sz = 0.7

lib_train = tokenizer.texts_to_sequences(lib_list[:int(train_sz * len(lib_list))])
con_train = tokenizer.texts_to_sequences(con_list[:int(train_sz * len(con_list))])
neutral_train = tokenizer.texts_to_sequences(neutral_list[:int(train_sz * len(neutral_list))])
lib_test = tokenizer.texts_to_sequences(lib_list[int(train_sz * len(lib_list)):])
con_test = tokenizer.texts_to_sequences(con_list[int(train_sz * len(con_list)):])
neutral_test = tokenizer.texts_to_sequences(neutral_list[int(train_sz * len(neutral_list)):])

X_train = lib_train + neutral_train + con_train
X_test = lib_test + neutral_test + con_test
y_train = [0] * len(lib_train) + [1] * len(neutral_train) + [2] * len(con_train)
y_test = [0] * len(lib_test) + [1] * len(neutral_test) + [2] * len(con_test)


# pad documents to a max of 90 words
max_sentence_length = 90
# padded_texts = sequence.pad_sequences(encoded_texts, maxlen=max_sentence_length, padding='post')
# print padded_texts[0]

# load the whole embedding into memory
# GLOVE SHITTT
# embeddings_index = dict()
# f = open('../Downloads/Glove.6B/glove.6B.50d.txt')
# for line in f:
#     values = line.split()
#     word = values[0]
#     coefs = asarray(values[1:], dtype='float32')
#     embeddings_index[word] = coefs
# f.close()
# print('Loaded %s word vectors.' % len(embeddings_index))




# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest

# truncate and pad input sequences
X_train = sequence.pad_sequences(X_train, maxlen=max_sentence_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_sentence_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(vocab_size, embedding_vecor_length, input_length=max_sentence_length))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
