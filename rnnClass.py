#!/Users/spencerevans/tensorflow/bin/python

 
import numpy as np
import pandas as pd

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split

import itertools
from random import *

import signal
import sys
import time

df = pd.DataFrame()
randUniqueFileInt = randint(1, 1000)



# this function allows us to interupt the randomizedGridSearch without losing the data
# that has already been compiled.
# it intercepts the ctrl + c Keyboard Interupt command and deals with it here
def signal_handler(signal, frame):
    print '\nYou pressed Ctrl+C!'

    # save the current data gathered and print it to terminal
    file_path = "results/Interupted" + str(randUniqueFileInt) + ".xlsx"
    print "Saving dataframe to excel file as " + file_path + "and printing"
    df.to_excel(file_path)
    print df

    # give the user option to continue or exit, re-prompts until valid response
    valid_answer = False
    while not valid_answer:
        text = raw_input("Do you want to exit? [y/n]\n")
        if text == "y":
            print "Exiting"
            sys.exit(0)
        elif text == "n":
            valid_answer = True
            print "Continuing"




# allows us to test a large parameter space by sampling randomly
def randomizedGridSearch(params, param_names, iters, X_train, y_train, 
                        X_test, y_test, embedding, file_name):
    # sanity check to make sure params line up
    assert(len(param_names) == len(params))
    print "Current params: ", zip(param_names, params)

    column_names = param_names + ["score"]
    for name in column_names:
        df[name] = None
    # create all possible permutations of the parameters
    param_permutations = list(itertools.product(*[param for param in params]))

    # variables to implement the random element of the randomized grid search
    ctr = 0
    listLength = len(param_permutations)
    proportion = listLength / iters

    # initializes the signal handler to catch any keyboard interupts 
    signal.signal(signal.SIGINT, signal_handler)

    # iterate through all possible parameter permutations
    for i, p in enumerate(param_permutations):  
        # implements the random part of the grid search, roughly this ensures that we 
        # have a max of "iters" iterations       
        if proportion <= 1 or randint(1, proportion) == 1:
            # build, train, test model
            model = RNN(embedding, epochs=p[0], batch_size=p[1], loss=p[3], dropout=p[4],
                         opt_flag=p[5], opt_lr=p[6], cnn_flag=p[7], kernel_size=p[8], lstm_flag=p[9])
            model.fit(X_train, y_train, validation_split=p[2])
            score = model.evaluate(X_test, y_test)
            
            # store the list of params used in this iteration and the score acheived
            data = list(p)
            data.append(score)
            df.loc[ctr] = data
            ctr += 1

            # print params used, score, and how far through the random grid search we are
            print zip(param_names, p)
            print "Score: ", score, "\n"
            print "\nPROGRESS: ", round(float(ctr) / iters, 2) * 100, "%", " done"

    # store the completed dataframe as an excel file in the results folder
    # randUniqueFileInt to avoid overwriting other files accidentally
    file_path = "results/tuning" + file_name + str(randUniqueFileInt) + ".xlsx"
    df.to_excel(file_path)
    print "Done!\n\n"
    print df


# RNN class with fit, predict, evaluate, score, get_params methods
class RNN():
    def __init__(self, embedding, epochs=10, batch_size=32, neurons=100, 
                loss='binary_crossentropy', dropout=0.2, opt_flag=True,
                 opt_lr=0.001, cnn_flag=True, lstm_flag=True, kernel_size=3, ):
        self.embedding = embedding
        self.epochs = epochs 
        self.batch_size = batch_size 
        self.neurons = neurons 
        self.loss = loss 
        self.dropout = dropout 
        self.opt_flag = opt_flag 
        self.opt_lr = opt_lr 
        self.cnn_flag = cnn_flag
        self.lstm_flag = lstm_flag
        self.kernel_size = kernel_size



    def fit(self, X_train, y_train, validation_split=None, class_weight=None):
        signal.signal(signal.SIGINT, signal_handler)
        self.model = Sequential()

        # add GloVe embedding
        self.model.add(self.embedding)

        # add CNN layer based on cnn_flag
        if self.cnn_flag:
            self.model.add(Conv1D(filters=32, kernel_size=self.kernel_size, padding='same', activation='relu'))
            self.model.add(MaxPooling1D(pool_size=2))

        # add LSTM and Dense layer
        if self.lstm_flag:
            self.model.add(LSTM(self.neurons, dropout=self.dropout, recurrent_dropout=self.dropout))
        else:
            self.model.add(GRU(self.neurons, activation='tanh', dropout=self.dropout, recurrent_dropout=self.dropout))

        self.model.add(Dense(1, activation='sigmoid'))

        # set optimizer based on flags
        optimizer = None
        if self.opt_flag == "adam":
            optimizer = keras.optimizers.Adam(lr=self.opt_lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        else:
            optimizer = keras.optimizers.RMSprop(lr=self.opt_lr, rho=0.9, epsilon=1e-08, decay=0.0)
        
        # compile and fit model
        self.model.compile(loss=self.loss, optimizer=optimizer, metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, 
                        validation_split=validation_split, class_weight=class_weight)


    def predict(self, X):
        return self._model.predict(X)

    def get_params(self, deep=True):
        return {'n_classes': self.n_classes, 'batch_size': self.batch_size, 'valid_set': self.valid_set,
                "layers": self.layers, "hidden_units": self.hidden_units,
                'input_dropout': self.input_dropout, 'hidden_dropout': self.hidden_dropout,
                'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay}

    def evaluate(self, X, y, batch_size=None, verbose=1, sample_weight=None):
        return self.model.evaluate(x=X, y=y, batch_size=batch_size, verbose=verbose, 
                                    sample_weight=sample_weight)


    def scoreAccuracy(self, X, y, batch_size=None, verbose=1, sample_weight=None):
        score = self.evaluate(X, y, batch_size, verbose, sample_weight)
        return score[1] * 100








