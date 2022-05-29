from calendar import c
import numpy as np 
import pandas as pd 
import time
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import Dense,Flatten,Input,Conv2D,UpSampling2D,Conv3D,MaxPooling2D,MaxPooling3D,BatchNormalization,Activation,Embedding,CuDNNLSTM,Dropout,LSTM,concatenate,add,Reshape,multiply,average
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from tensorflow.keras.utils import plot_model
from utils import get_glove_embedding_glove,get_glove_embedding_fasttext





class Unimodal():


        


    def word_embedding_lstm(self,input_arr,max_nb_words,embedding_dim,number_lstm_layers,number_dense_layers,number_labels,is_gpu_available):
        model=Sequential()
        model.add(Embedding(max_nb_words,embedding_dim,input_length=input_arr.shape[1]))

        number_lstm_layers=max(1,number_lstm_layers)
        for i in range(number_lstm_layers-1):
            if not is_gpu_available:
                model.add(LSTM(15,recurrent_dropout=0.2,dropout=0.2,return_sequences=True))
            else:
                model.add(CuDNNLSTM(100,return_sequences=True))
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
        if not is_gpu_available:
            x_text=LSTM(100)(x_text)
        else:
            x_text=CuDNNLSTM(100)(x_text)
        
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

    def text_image_autoencoder(self,tokenizer,input_arr_text,input_arr_img,max_nb_words,embedding_dim,number_lstm_layers,number_dense_layers_text,number_dense_layers_cnn,number_cnn_layers,number_of_filters,filter_size,pool_size,number_labels,is_gpu_available):
        ####### Input Variables ######

        emb=get_glove_embedding_glove(100,tokenizer,input_arr_text.shape[1])
        X_embedded = emb.predict(input_arr_text)

        
        ###### Text Encoding ######
        
        
        input_i = Input(shape=(input_arr_text.shape[1],100))
        encoded_h1 = Dense(512, activation='tanh')(input_i)
        encoded_h2 = Dense(512, activation='tanh')(encoded_h1)
        encoded_h3 = Dense(256, activation='tanh')(encoded_h2)
        encoded_h4 = Dense(256, activation='tanh')(encoded_h3)
        encoded_h5 = Dense(128, activation='tanh')(encoded_h4)
        latent = Dense(64, activation='tanh')(encoded_h5)
        text_latent_dim=K.int_shape(latent)[1:]
        latent=Flatten()(latent)


        ###### Image Encoding ######
        input_arr_img=input_arr_img.astype("float32")/255

        input_img = Input(shape=input_arr_img.shape[1:])
        
        x = Conv2D(64, (3, 3), padding='same')(input_img)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(16, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        enc_h,enc_w,enc_channel=K.int_shape(encoded)[1:]
        encoded=Flatten()(encoded)

        bimodal_latent=concatenate([latent,encoded])

        ### Text Decoding
        decoder=Dense(text_latent_dim[0]*text_latent_dim[1])(bimodal_latent)
        decoder=Reshape((text_latent_dim[0],text_latent_dim[1]))(decoder)
        decoder_h1 = Dense(128, activation='tanh')(decoder)
        decoder_h2 = Dense(256, activation='tanh')(decoder_h1)
        decoder_h3 = Dense(256, activation='tanh')(decoder_h2)
        decoder_h4 = Dense(512, activation='tanh')(decoder_h3)
        decoder_h5 = Dense(512, activation='tanh')(decoder_h4)

        output = Dense(100, activation='tanh')(decoder_h5)

        ### Image Decoding

        encoded=Dense(enc_h*enc_w*enc_channel,activation="relu")(bimodal_latent)
        
        encoded=Reshape((enc_h,enc_w,enc_channel))(encoded)
        x = Conv2D(16, (3, 3), padding='same')(encoded)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(3, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        decoded = Activation('sigmoid')(x)

        bimodal_autoencoder=Model([input_i,input_img],[output,decoded])
        bimodal_autoencoder.compile("adadelta","mse")

        print(bimodal_autoencoder.summary())

        es = EarlyStopping(monitor='val_loss',patience=5)
        bimodal_autoencoder.fit([X_embedded,input_arr_img],[X_embedded,input_arr_img],callbacks=[es],epochs=25)

        #return autoencoder_img
        bimodal_encoder = Model([input_i,input_img], bimodal_latent)
        return bimodal_autoencoder,bimodal_encoder
    
    def inference_bimodal_encoder(self,tokenizer,input_arr_text,input_arr_img,bimodal_encoder):
        emb=get_glove_embedding_glove(100,tokenizer,input_arr_text.shape[1])
        X_embedded = emb.predict(input_arr_text)

        features=bimodal_encoder.predict([X_embedded,input_arr_img])
        print(features)

        return features

         



    def text_image_residual_network(self,tokenizer,input_arr_text,input_arr_img,labels,input_arr_text_val,input_arr_img_val,labels_val,max_nb_words,embedding_dim,number_lstm_layers,number_dense_layers_text,number_dense_layers_cnn,number_cnn_layers,number_of_filters,filter_size,pool_size,number_labels,is_gpu_available,model_name):

        emb=get_glove_embedding_glove(100,tokenizer,input_arr_text.shape[1])
        X_embedded = emb.predict(input_arr_text)
        X_embedded_val=emb.predict(input_arr_text_val)

        input_i = Input(shape=(input_arr_text.shape[1],100))

        x_text=LSTM(64,return_sequences=True)(input_i)
        x_text=LSTM(64)(x_text)
        x_text=Dense(256,activation="tanh")(x_text)

        input_arr_img=input_arr_img.astype("float32")/255

        input_img = Input(shape=input_arr_img.shape[1:])

        x_img = Conv2D(64, (3, 3))(input_img)
        x_img = BatchNormalization()(x_img)
        x_img = Activation('relu')(x_img)
        x_img = MaxPooling2D((2, 2))(x_img)
        x_img = Conv2D(32, (3, 3))(x_img)
        x_img = BatchNormalization()(x_img)
        x_img = Activation('relu')(x_img)
        x_img = MaxPooling2D((2, 2))(x_img)
        x_img = Conv2D(16, (3, 3))(x_img)
        x_img = BatchNormalization()(x_img)
        x_img = Activation('relu')(x_img)
        x_img=Flatten()(x_img)
        x_img=Dense(256,activation="tanh")(x_img)

        block_multip=multiply([x_text,x_img])

        x_block_1=add([x_text,block_multip])

        x_text=Dense(256,"tanh")(x_block_1)

        x_img=Dense(256,"tanh")(x_img)
        x_img=Dense(256,"tanh")(x_img)

        block_multip=multiply([x_text,x_img])

        x_block_2=add([x_block_1,block_multip])

        x_text=Dense(256,"tanh")(x_block_2)

        x_img=Dense(256,"tanh")(x_img)
        x_img=Dense(256,"tanh")(x_img)

        block_multip=multiply([x_text,x_img])
        x_block_3=add([x_block_2,block_multip])

        output=Dense(number_labels,"softmax")(x_block_3)


        residual_network=Model([input_i,input_img],output)
        residual_network.compile(optimizer="adadelta",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

        print(residual_network.summary())
        #assert False 
        es = EarlyStopping(monitor='val_loss',patience=5)
        csv_logger = CSVLogger(f'logs_metrics/residual_network_log_{time.time()}.csv', append=True, separator=',')
        cp=ModelCheckpoint(filepath = model_name)
        block_1 = Model([input_i,input_img], x_block_1)
        block_2 = Model([input_i,input_img], x_block_2)
        block_3 = Model([input_i,input_img], x_block_3)

        residual_network.fit([X_embedded,input_arr_img],labels,epochs=25,callbacks=[es,csv_logger,cp],validation_data=([X_embedded_val,input_arr_img_val],labels_val))

        return block_1,block_2,block_3



    def lstm_multi_embedding(self,tokenizer,input_arr_text,labels,input_arr_text_val,input_arr_img_val,labels_val,max_nb_words,number_labels,is_gpu_available,model_name="multiembedding_network_3.h5"):

        emb_glove=get_glove_embedding_glove(300,tokenizer,input_arr_text.shape[1])
        emb_fasttext=get_glove_embedding_fasttext(tokenizer,input_arr_text.shape[1])


        X_embedded_glove = emb_glove.predict(input_arr_text)
        X_embedded_ft=emb_fasttext.predict(input_arr_text) 

        X_embedded_glove_val = emb_glove.predict(input_arr_text_val)
        X_embedded_ft_val=emb_fasttext.predict(input_arr_text_val) 

        input_glove = Input(shape=(input_arr_text.shape[1],300))
        x_glove=LSTM(256,return_sequences=True)(input_glove)
        x_glove=LSTM(256)(x_glove) 

        
        input_ft = Input(shape=(input_arr_text.shape[1],300))
        x_ft=LSTM(256,return_sequences=True)(input_ft)
        x_ft=LSTM(256)(x_ft)
        
        x_avg=average([x_glove,x_ft])

        x=Dense(512)(x_avg)
        x=Dense(256)(x)

        output=Dense(number_labels,"softmax")(x)

        multi_embedding=Model([input_glove,input_ft],output)
        multi_embedding.compile(optimizer="adadelta",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

        print(multi_embedding.summary())

        es = EarlyStopping(monitor='val_loss',patience=5)
        csv_logger = CSVLogger(f'logs_metrics/multiembedding_network_log_{time.time()}.csv', append=True, separator=',')
        cp=ModelCheckpoint(filepath = model_name)

        multi_embedding.fit([X_embedded_glove,X_embedded_ft],labels,epochs=100,callbacks=[es,csv_logger,cp],validation_data=([X_embedded_glove_val,X_embedded_ft_val],labels_val))


        x_avg_emb = Model([input_glove,input_ft],x_avg)

        return x_avg_emb
        

        




