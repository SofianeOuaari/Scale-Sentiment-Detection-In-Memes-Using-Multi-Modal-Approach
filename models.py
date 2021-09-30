import numpy as np 
import pandas as pd 
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Input,Conv2D,Conv3D,MaxPooling2D,MaxPooling3D,Embedding,LSTM





class Unimodal():


    def __init__(self):
        self.MAX_NB_WORDS=1000
        self.EMBEDDING_DIM =300


    def word_embedding_lstm(self,input_arr,number_lstm_layers,number_dense_layers,number_labels):
        model=Sequential()
        model.add(Embedding(self.MAX_NB_WORDS,self.EMBEDDING_DIM,input_length=input_arr.shape[1]))

        number_lstm_layers=max(1,number_lstm_layers)
        for i in range(number_lstm_layers-1):
            model.add(LSTM(100,recurrent_dropout=0.2,dropout=0.2,return_sequences=True))
        
        model.add(LSTM(100))
        for _ in range(number_dense_layers):
            model.add(Dense(64,"relu"))
        model.add(Dense(number_labels,"softmax"))

        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        return model


