from keras_preprocessing.sequence import pad_sequences
import numpy as np
from preprocess import return_rgb_images_text,preprocess_text,label_encoder,return_all_color_spaces,get_tokenized_padded_text
from models import Unimodal,Multimodal,get_glove_embedding_glove,get_glove_embedding_fasttext
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from keras.models import load_model
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
from pages.ssmml import sentiment_prediction_multi_emb, sentiment_prediction_residual
from utils import change_to_three_sentiment_labels,augment_dataset,plot_color_spaces,save_label_encoder,load_keras_model,turn_numpy_df
import matplotlib.pyplot as plt
from pycaret.classification import *
import time










if __name__=="__main__":

    img_arr,img_name,texts,y=return_rgb_images_text("memotion_dataset_7k/train.csv",224,224)
    img_arr_valid,img_name_valid,texts_valid,y_valid=return_rgb_images_text("memotion_dataset_7k/valid.csv",224,224)
    img_arr_test,img_name_test,texts_test,y_test=return_rgb_images_text("memotion_dataset_7k/test.csv",224,224)


    y=change_to_three_sentiment_labels(y)
    y_valid=change_to_three_sentiment_labels(y_valid)
    y_test=change_to_three_sentiment_labels(y_test)
    #img_arr_aug,img_name_aug,texts_aug,y_aug=augment_dataset(img_arr,img_name,texts,y)
    #img_arr,img_name,texts,y=augment_dataset(img_arr,img_name,texts,y)
    encoded_label,l_e=label_encoder(y)
    
    

    tokenized_texts,tokenizer=preprocess_text(texts,10000,25,"text_tokenizer.pickle")
    tokenized_val_text=get_tokenized_padded_text(tokenizer,texts_valid,25)
    tokenized_test_text=get_tokenized_padded_text(tokenizer,texts_test,25)

    unimo=Unimodal()
    multi_modal=Multimodal()
    vocab_size=len(tokenizer.word_index)+1
    encoded_label,l_e=label_encoder(y)
    encoded_label_val=l_e.transform(y_valid)
    encoded_label_test=l_e.transform(y_test)

    #print(unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,5,5,3,is_gpu_available=True).summary())
    
    
    # LSTM Unimodal
    w_e_lstm=unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,1,2,3,is_gpu_available=False)
    w_e_lstm.fit(tokenized_texts,encoded_label,batch_size=32,epochs=15,validation_data=(tokenized_val_text,encoded_label_val))


    ## CNN & Resnet Unimodal
    cnn_model=unimo.image_cnn_model(img_arr,2,2,5,(3,3),(2,2),number_labels=3)

    cnn_model_resnet=unimo.image_resnet_model(img_arr,2,3)

    cnn_model_resnet.fit(img_arr,encoded_label,batch_size=8,epochs=15,validation_data=(img_arr_valid,y_valid))

    # Multimodal Custom CNN+LSTM
    multi_modal_lstm_cnn=multi_modal.word_embedding_lstm_cnn(texts,img_arr,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False)
    print(tokenized_texts.shape)
    encoded_label=encoded_label.reshape(-1,1)
    print(encoded_label.shape)
    print(img_arr.shape)

    print(tokenized_val_text.shape)
    print(img_arr_valid.shape)

    #assert False
    
    multi_modal_lstm_cnn.fit([np.array(tokenized_texts),np.array(img_arr)],encoded_label,batch_size=32,epochs=15,validation_data=([tokenized_val_text,img_arr_valid],encoded_label_val))
    #img_arr=img_arr.astype("float32")/255
    
    ### Multi-color channel 
    multi_modal=Multimodal()
    print("preparing all color spaces")
    hsv_img,lab_img,grey_img,ylcrcb_img=return_all_color_spaces(img_arr)
    plot_color_spaces(hsv_img[5],lab_img[5],grey_img[5],ylcrcb_img[5])
    hsv_img_val,lab_img_val,grey_img_val,ylcrcb_img_val=return_all_color_spaces(img_arr_valid)
    print("returning all Color Spaces")
    #grey_img=np.reshape(grey_img,(-1,224,224,1))
    img_color_spaces=[img_arr,hsv_img,lab_img,grey_img,ylcrcb_img]
    multi_channel_color_space_cnn=multi_modal.cnn_multiple_color_spaces(img_color_spaces,number_dense_layers_cnn=2,number_cnn_layers=5,number_of_filters=32,filter_size=(3,3),pool_size=(2,2),number_labels=5)
    es = EarlyStopping(monitor='val_accuracy',patience=5)
    csv_logger = CSVLogger(f'logs_metrics/multi_color_spaces_log_{time.time()}.csv', append=True, separator=',')
    multi_channel_color_space_cnn.fit([img_arr,hsv_img,lab_img,grey_img,ylcrcb_img],encoded_label,batch_size=8,epochs=15,validation_data=([img_arr_valid,hsv_img_val,lab_img_val,grey_img_val,ylcrcb_img_val],encoded_label_val),callbacks=[csv_logger,es])


    ### Saving Features from Bimodal AutoEncoders ###  

    bimodal_autoencoder,bimodal_latent=multi_modal.text_image_autoencoder(tokenizer,np.array(tokenized_texts),img_arr,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False)
    bimodal_latent.save("bimodal_latent_3.h5")
    bimodal_encoder=load_keras_model("bimodal_latent_3.h5")
    X_train_fea=multi_modal.inference_bimodal_encoder(tokenizer,np.array(tokenized_texts),img_arr,bimodal_encoder)
    X_val_fea=multi_modal.inference_bimodal_encoder(tokenizer,np.array(tokenized_val_text),img_arr_valid,bimodal_encoder)
    X_test_fea=multi_modal.inference_bimodal_encoder(tokenizer,np.array(tokenized_test_text),img_arr_test,bimodal_encoder)
    scaler=MinMaxScaler()
    
    X_train_fea=scaler.fit_transform(X_train_fea)
    X_val_fea=scaler.transform(X_val_fea)
    X_test_fea=scaler.transform(X_test_fea)

    turn_numpy_df(X_train_fea,encoded_label,"train_latent_autoencoder_3.csv")
    turn_numpy_df(X_val_fea,encoded_label_val,"val_latent_autoencoder_3.csv")
    turn_numpy_df(X_test_fea,encoded_label_test,"test_latent_autoencoder_3.csv")




    ### Residual Multimodal Network 


    block_1,block_2,block_3=multi_modal.text_image_residual_network(tokenizer,np.array(tokenized_texts),img_arr,encoded_label,np.array(tokenized_val_text),img_arr_valid,encoded_label_val,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False,model_name="residual_network_3.h5")
    block_1.save("block_1_aug.h5")
    block_2.save("block_2_aug.h5")
    block_3.save("block_3_aug.h5")



    block_1=load_keras_model("block_1_1.h5")
    block_2=load_keras_model("block_2_1.h5")
    block_3=load_keras_model("block_3_1.h5")

    glove=get_glove_embedding_glove(100,tokenizer,np.array(tokenized_test_text).shape[1])
    X_emb_glove=glove.predict(np.array(tokenized_texts))
    X_emb_glove_val=glove.predict(np.array(tokenized_val_text))
    X_emb_glove_test=glove.predict(np.array(tokenized_test_text))

    fea_train_1=block_1.predict([X_emb_glove,img_arr])
    fea_val_1=block_1.predict([X_emb_glove_val,img_arr_valid])
    fea_test_1=block_1.predict([X_emb_glove_test,img_arr_test])
    scaler=MinMaxScaler()
    fea_train_1=scaler.fit_transform(fea_train_1)
    fea_val_1=scaler.transform(fea_val_1)
    fea_test_1=scaler.transform(fea_test_1)

    turn_numpy_df(fea_train_1,encoded_label,"residual_block_1_1_train.csv")
    turn_numpy_df(fea_val_1,encoded_label_val,"residual_block_1_1_val.csv")
    turn_numpy_df(fea_test_1,encoded_label_test,"residual_block_1_1_test.csv")

    fea_train_2=block_2.predict([X_emb_glove,img_arr])
    fea_val_2=block_2.predict([X_emb_glove_val,img_arr_valid])
    fea_test_2=block_2.predict([X_emb_glove_test,img_arr_test])
    scaler=MinMaxScaler()
    fea_train_2=scaler.fit_transform(fea_train_2)
    fea_val_2=scaler.transform(fea_val_2)
    fea_test_2=scaler.transform(fea_test_2)

    turn_numpy_df(fea_train_2,encoded_label,"residual_block_2_1_train.csv")
    turn_numpy_df(fea_val_2,encoded_label_val,"residual_block_2_1_val.csv")
    turn_numpy_df(fea_test_2,encoded_label_test,"residual_block_2_1_test.csv")

    fea_train_3=block_3.predict([X_emb_glove,img_arr])
    fea_val_3=block_3.predict([X_emb_glove_val,img_arr_valid])
    fea_test_3=block_3.predict([X_emb_glove_test,img_arr_test])
    scaler=MinMaxScaler()
    fea_train_3=scaler.fit_transform(fea_train_3)
    fea_val_3=scaler.transform(fea_val_3)
    fea_test_3=scaler.transform(fea_test_3)

    turn_numpy_df(fea_train_3,encoded_label,"residual_block_3_1_train.csv")
    turn_numpy_df(fea_val_3,encoded_label_val,"residual_block_3_1_val.csv")
    turn_numpy_df(fea_test_3,encoded_label_test,"residual_block_3_1_test.csv")


    ### MultiEmbedding 


    multi_modal.lstm_multi_embedding(tokenizer,np.array(tokenized_texts),encoded_label,vocab_size,3,False)



