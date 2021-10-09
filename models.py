import numpy as np 
import pandas as pd 
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Input,Conv2D,Conv3D,MaxPooling2D,MaxPooling3D,Embedding,CuDNNLSTM,Dropout,LSTM
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19





class Unimodal():


        


    def word_embedding_lstm(self,input_arr,max_nb_words,embedding_dim,number_lstm_layers,number_dense_layers,number_labels):
        model=Sequential()
        model.add(Embedding(max_nb_words,embedding_dim,input_length=input_arr.shape[1]))

        number_lstm_layers=max(1,number_lstm_layers)
        for i in range(number_lstm_layers-1):
            model.add(LSTM(100,recurrent_dropout=0.2,dropout=0.2,return_sequences=True))
            #model.add(CuDNNLSTM(100,return_sequences=True))
        
        model.add(LSTM(100))
        for _ in range(number_dense_layers):
            model.add(Dense(64,"relu"))
            model.add(Dropout(0.2))
        model.add(Dense(number_labels,"softmax"))

        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        return model


    def image_cnn_model(self,input_arr,number_dense_layers,number_cnn_layers,number_of_filters,filter_size,pool_size,number_labels):

        model=Sequential()

        for i in range(max(1,number_cnn_layers)):
            if i==0:
                model.add(Conv2D(number_of_filters,filter_size,activation="relu",input_shape=input_arr.shape[1:]))
                model.add(MaxPooling2D(pool_size))
            else:
                model.add(Conv2D(number_of_filters,filter_size,activation="relu"))
                model.add(MaxPooling2D(pool_size))

        model.add(Flatten())

        for i in range(max(1,number_dense_layers)):
            model.add(Dense(64,"relu"))
        
        model.add(Dense(number_labels,"softmax"))


        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        return model
    

    def image_resnet_model(self,input_arr,number_dense_layers,number_labels):

        base_model=ResNet50(include_top=False,input_shape=input_arr.shape[1:])
        x=base_model.output
        x=MaxPooling2D((2,2))(x)
        x=Flatten()(x)

        for _ in range(number_dense_layers):
            x=Dense(32,"relu")(x)
        
        predictions=Dense(number_labels,"softmax")(x)

        #x.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        return model
    
    def image_vgg16_model(self,input_arr,number_dense_layers,number_labels):

        base_model=VGG16(include_top=False,input_shape=input_arr.shape[1:])
        x=base_model.output
        x=MaxPooling2D((2,2))(x)
        x=Flatten()(x)

        for _ in range(number_dense_layers):
            x=Dense(32,"relu")(x)
        
        predictions=Dense(number_labels,"softmax")(x)

        #x.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        return model
    
    def image_vgg19_model(self,input_arr,number_dense_layers,number_labels):

        base_model=VGG19(include_top=False,input_shape=input_arr.shape[1:])
        x=base_model.output
        x=MaxPooling2D((2,2))(x)
        x=Flatten()(x)

        for _ in range(number_dense_layers):
            x=Dense(32,"relu")(x)
        
        predictions=Dense(number_labels,"softmax")(x)

        #x.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        model = Model(inputs = base_model.input, outputs = predictions)
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

        return model


class Multimodal():
    pass
