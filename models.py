import numpy as np 
import pandas as pd 
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Input,Conv2D,Conv3D,MaxPooling2D,MaxPooling3D,Embedding,CuDNNLSTM,Dropout,LSTM,concatenate
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model





class Unimodal():


        


    def word_embedding_lstm(self,input_arr,max_nb_words,embedding_dim,number_lstm_layers,number_dense_layers,number_labels,is_gpu_available):
        model=Sequential()
        model.add(Embedding(max_nb_words,embedding_dim,input_length=input_arr.shape[1]))

        number_lstm_layers=max(1,number_lstm_layers)
        for i in range(number_lstm_layers-1):
            if not is_gpu_available:
                model.add(LSTM(100,recurrent_dropout=0.2,dropout=0.2,return_sequences=True))
            else:
                model.add(CuDNNLSTM(100,return_sequences=True))
            #model.add(CuDNNLSTM(100,return_sequences=True))
        if not is_gpu_available:
            model.add(LSTM(100))
        else:
            model.add(CuDNNLSTM(100))
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
    def word_embedding_lstm_cnn(self,input_arr_text,input_arr_img,max_nb_words,embedding_dim,number_lstm_layers,number_dense_layers_text,number_dense_layers_cnn,number_cnn_layers,number_of_filters,filter_size,pool_size,number_labels,is_gpu_available):

        input_text=Input(shape=(None,))

        x_text=Embedding(max_nb_words,embedding_dim)(input_text)
        number_lstm_layers=max(1,number_lstm_layers)
        for i in range(number_lstm_layers-1):
            if not is_gpu_available:
                x_text=LSTM(100,recurrent_dropout=0.2,dropout=0.2,return_sequences=True)(x_text)
            else:
                x_text=CuDNNLSTM(100,return_sequences=True)(x_text)
            #model.add(CuDNNLSTM(100,return_sequences=True))
        if not is_gpu_available:
            x_text=LSTM(100)(x_text)
        else:
            x_text=CuDNNLSTM(100)(x_text)
        '''for _ in range(number_dense_layers_text):
            model.add(Dense(64,"relu"))
            model.add(Dropout(0.2))'''
        
        input_cnn=Input(shape=input_arr_img.shape[1:])

        for i in range(max(1,number_cnn_layers)):
            if i==0:
                x_img=Conv2D(number_of_filters,filter_size,activation="relu")(input_cnn)
                x_img=MaxPooling2D(pool_size)(x_img)
            else:
                x_img=Conv2D(number_of_filters,filter_size,activation="relu")(x_img)
                x_img=MaxPooling2D(pool_size)(x_img)

        flatten=Flatten()(x_img)

        for i in range(max(1,number_dense_layers_cnn)):
            if i==0:
                x_img=Dense(64,"relu")(flatten)
            else:
                x_img=Dense(64,"relu")(x_img)

        concatenated=concatenate([x_text,x_img])

        output=Dense(number_labels,"softmax")(concatenated)

        model=Model([input_text,input_cnn],output)
        
        print(model.summary())
        #plot_model(model, to_file='multiple_inputs.png')

        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        return model
    

    def cnn_multiple_color_spaces(self,img_color_spaces,number_dense_layers_cnn,number_cnn_layers,number_of_filters,filter_size,pool_size,number_labels):
        color_inputs=[]
        concatenated=[]
        for color_space in img_color_spaces:
            input_cnn=Input(shape=color_space.shape[1:])
            color_inputs.append(input_cnn)
            for i in range(max(1,number_cnn_layers)):
                if i==0:
                    x_img=Conv2D(number_of_filters,filter_size,activation="relu")(input_cnn)
                    x_img=MaxPooling2D(pool_size)(x_img)
                else:
                    x_img=Conv2D(number_of_filters,filter_size,activation="relu")(x_img)
                    x_img=MaxPooling2D(pool_size)(x_img)

            flatten=Flatten()(x_img)

            for i in range(max(1,number_dense_layers_cnn)):
                if i==0:
                    x_img=Dense(64,"relu")(flatten)
                else:
                    x_img=Dense(64,"relu")(x_img)
            
            concatenated.append(x_img)

        concat=concatenate(concatenated)
        output=Dense(number_labels,"softmax")(concat)

        model=Model(color_inputs,output)
        
        model.compile(loss="sparse_categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        return model




        #pass 



    #pass