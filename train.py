from keras_preprocessing.sequence import pad_sequences
import numpy as np
from preprocess import return_rgb_images_text,preprocess_text,label_encoder,return_all_color_spaces,get_tokenized_padded_text
from models import Unimodal,Multimodal
from keras.callbacks import CSVLogger,EarlyStopping
from utils import change_to_three_sentiment_labels,augment_dataset
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
    
    

    '''w_e_lstm=unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,1,2,3,is_gpu_available=False)
    w_e_lstm.fit(tokenized_texts,encoded_label,batch_size=32,epochs=15,validation_data=(tokenized_val_text,encoded_label_val))'''

    
    '''multi_modal_lstm_cnn=multi_modal.word_embedding_lstm_cnn(texts,img_arr,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False)
    print(tokenized_texts.shape)
    encoded_label=encoded_label.reshape(-1,1)
    print(encoded_label.shape)
    print(img_arr.shape)

    print(tokenized_val_text.shape)
    print(img_arr_valid.shape)

    #assert False
    
    multi_modal_lstm_cnn.fit([np.array(tokenized_texts),np.array(img_arr)],encoded_label,batch_size=32,epochs=15,validation_data=([tokenized_val_text,img_arr_valid],encoded_label_val))'''
    #img_arr=img_arr.astype("float32")/255
    '''cnn_model=unimo.image_cnn_model(img_arr,2,2,5,(3,3),(2,2),number_labels=3)
    cnn_model=unimo.image_resnet_model(img_arr,2,3)

    cnn_model.fit(img_arr,encoded_label,batch_size=8,epochs=15,validation_data=(img_arr_valid,y_valid))'''


    ### Multi-color channel 
    '''multi_modal=Multimodal()
    
    hsv_img,lab_img,grey_img,ylcrcb_img=return_all_color_spaces(img_arr)
    grey_img=np.reshape(grey_img,(-1,224,224,1))
    img_color_spaces=[img_arr,hsv_img,lab_img,grey_img,ylcrcb_img]
    multi_channel_color_space_cnn=multi_modal.cnn_multiple_color_spaces(img_color_spaces,number_dense_layers_cnn=2,number_cnn_layers=5,number_of_filters=32,filter_size=(3,3),pool_size=(2,2),number_labels=3)
    es = EarlyStopping(monitor='val_accuracy',patience=5)
    csv_logger = CSVLogger(f'logs_metrics/multi_color_spaces_log_{time.time()}.csv', append=True, separator=',')
    multi_channel_color_space_cnn.fit([img_arr,hsv_img,lab_img,grey_img,ylcrcb_img],encoded_label,batch_size=8,epochs=15,validation_split=0.25,callbacks=[csv_logger,es])'''


    ### AutoEncoders
    multi_modal=Multimodal()
    print(texts.shape)

    text_autoencoder,img_autoencoder=multi_modal.text_image_autoencoder(tokenizer,np.array(tokenized_texts),img_arr,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False)




