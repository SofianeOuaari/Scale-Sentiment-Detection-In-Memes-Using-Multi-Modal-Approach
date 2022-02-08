from keras_preprocessing.sequence import pad_sequences
import numpy as np
from preprocess import return_rgb_images_text,preprocess_text,label_encoder,return_all_color_spaces,get_tokenized_padded_text
from models import Unimodal,Multimodal
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from keras.callbacks import CSVLogger,EarlyStopping
from utils import change_to_three_sentiment_labels,augment_dataset,plot_color_spaces
import matplotlib.pyplot as plt
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


    ### AutoEncoders
    #  

    bimodal_autoencoder,bimodal_latent=multi_modal.text_image_autoencoder(tokenizer,np.array(tokenized_texts),img_arr,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False)
    bimodal_latent.save("bimodal_latent.h5")
    bimodal_encoder=load_model("bimodal_latent.h5")
    X_train_fea=multi_modal.inference_bimodal_encoder(tokenizer,np.array(tokenized_texts),img_arr,bimodal_encoder)
    X_val_fea=multi_modal.inference_bimodal_encoder(tokenizer,np.array(tokenized_val_text),img_arr_valid,bimodal_encoder)
    X_test_fea=multi_modal.inference_bimodal_encoder(tokenizer,np.array(tokenized_test_text),img_arr_test,bimodal_encoder)

    #svm_model=LinearSVC(C=500)
    #svm_model=KNeighborsClassifier(7,n_jobs=-1)
    '''p_svm_50=Pipeline([("smote",SMOTE()),("svm_50",LinearSVC(C=50))])
    p_svm_100=Pipeline([("smote",SMOTE()),("svm_100",LinearSVC(C=100))])
    p_svm_500=Pipeline([("smote",SMOTE()),("svm_500",LinearSVC(C=500))])'''

    #svm_model=VotingClassifier([('svm_50',LinearSVC(C=50)),('svm_100',LinearSVC(C=100)),('svm_500',LinearSVC(C=500)),('knn3',KNeighborsClassifier(3,n_jobs=-1)),('knn5',KNeighborsClassifier(5,n_jobs=-1))])
    #svm_model=RandomForestClassifier(n_estimators=50,n_jobs=-1)
    #svm_model=XGBClassifier(n_estimators=500,n_jobs=-1)
    '''scaler=MinMaxScaler()
    svm_model=LinearSVC(C=500,random_state=0)'''
    #svm_model=GradientBoostingClassifier(n_estimators=1000)
    #svm_model=OneVsRestClassifier(LinearSVC(C=50,random_state=0),n_jobs=-1)
    
    #X_train_fea=scaler.fit_transform(X_train_fea)
    '''X_train_fea_aug,y_aug=SMOTE(n_jobs=-1).fit_resample(X_train_fea,y)
    print(len(X_train_fea_aug))
    svm_model.fit(X_train_fea_aug,y_aug)'''
    '''svm_model.fit(X_train_fea,y)

    y_pred_val=svm_model.predict(scaler.transform(X_val_fea))
    y_pred_test=svm_model.predict(scaler.transform(X_test_fea))
    print(classification_report(y_valid,y_pred_val))
    print(classification_report(y_test,y_pred_test))'''




    ### Residual Multimodal Network 


    multi_modal.text_image_residual_network(tokenizer,np.array(tokenized_texts),img_arr,encoded_label,vocab_size,300,1,1,1,3,5,(3,3),(2,2),3,False)

    ### MultiEmbedding 


    multi_modal.lstm_multi_embedding(tokenizer,np.array(tokenized_texts),encoded_label,vocab_size,3,False)



