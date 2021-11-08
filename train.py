import numpy as np
from preprocess import return_rgb_images_text,preprocess_text,label_encoder,return_all_color_spaces
from models import Unimodal,Multimodal
from keras.callbacks import CSVLogger,EarlyStopping










if __name__=="__main__":

    img_arr,img_name,texts,y=return_rgb_images_text(224,224)
    encoded_label,l_e=label_encoder(y)
    ''''tokenized_texts,tokenizer=preprocess_text(texts,500,25,"text_tokenizer.pickle")

    unimo=Unimodal()
    multi_modal=Multimodal()
    vocab_size=len(tokenizer.word_index)+1
    encoded_label,l_e=label_encoder(y)

    print(unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,5,5,3,is_gpu_available=True).summary())
    
    

    w_e_lstm=unimo.word_embedding_lstm(tokenized_texts,vocab_size,300,1,5,5,is_gpu_available=True)

    w_e_lstm.fit(tokenized_texts,encoded_label,batch_size=32,epochs=15,validation_split=0.25)'''

    
    '''multi_modal_lstm_cnn=multi_modal.word_embedding_lstm_cnn(texts,img_arr,vocab_size,300,1,1,1,3,5,(3,3),(2,2),5,True)
    print(tokenized_texts.shape)
    encoded_label=encoded_label.reshape(-1,1)
    print(encoded_label.shape)
    print(img_arr.shape)
    multi_modal_lstm_cnn.fit([np.array(tokenized_texts),np.array(img_arr)],encoded_label,batch_size=32,epochs=15,validation_split=0.25)'''
    '''cnn_model=unimo.image_cnn_model(img_arr,2,2,5,(3,3),(2,2),number_labels=5)

    cnn_model.fit(img_arr,encoded_label,batch_size=8,epochs=15,validation_split=0.25)'''


    ### Multi-color channel 
    multi_modal=Multimodal()
    
    hsv_img,lab_img,grey_img,ylcrcb_img=return_all_color_spaces(img_arr)
    rey_img=np.reshape(grey_img,(-1,224,224,1))
    img_color_spaces=[img_arr,hsv_img,lab_img,grey_img,ylcrcb_img]
    multi_channel_color_space_cnn=multi_modal.cnn_multiple_color_spaces(img_color_spaces,number_dense_layers_cnn=2,number_cnn_layers=5,number_of_filters=32,filter_size=(3,3),pool_size=(2,2),number_labels=5)
    es = EarlyStopping(monitor='val_accuracy',patience=5)
    csv_logger = CSVLogger('multi_color_spaces_log.csv', append=True, separator=',')
    multi_channel_color_space_cnn.fit([img_arr,hsv_img,lab_img,grey_img,ylcrcb_img],encoded_label,batch_size=8,epochs=15,validation_split=0.25,callbacks=[csv_logger,es])


