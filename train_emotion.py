from keras_preprocessing.sequence import pad_sequences
import numpy as np
from preprocess import return_rgb_images_text_humour,preprocess_text,label_encode_binary_humours,get_tokenized_padded_text
from models import Unimodal,Multimodal,get_glove_embedding_glove
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier,VotingClassifier,ExtraTreesClassifier
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from keras.models import load_model
from keras.callbacks import CSVLogger,EarlyStopping,ModelCheckpoint
from utils import change_binary_classes,load_keras_model,turn_numpy_df
import matplotlib.pyplot as plt
from pycaret.classification import *
import time










if __name__=="__main__":

    img_arr,img_name,texts,y=return_rgb_images_text_humour("memotion_dataset_7k/train.csv",224,224)
    img_arr_valid,img_name_valid,texts_valid,y_valid=return_rgb_images_text_humour("memotion_dataset_7k/valid.csv",224,224)
    img_arr_test,img_name_test,texts_test,y_test=return_rgb_images_text_humour("memotion_dataset_7k/test.csv",224,224)

    
    y=change_binary_classes(y)
    y_valid=change_binary_classes(y_valid)
    y_test=change_binary_classes(y_test)
    print(y.value_counts())
    print(y_valid.value_counts())
    print(y_test.value_counts())


    y,y_valid,y_test=label_encode_binary_humours(y,y_valid,y_test)
    print(y_valid)


    

    tokenized_texts,tokenizer=preprocess_text(texts,10000,25,"text_tokenizer.pickle")
    tokenized_val_text=get_tokenized_padded_text(tokenizer,texts_valid,25)
    tokenized_test_text=get_tokenized_padded_text(tokenizer,texts_test,25)

    unimo=Unimodal()
    multi_modal=Multimodal()
    vocab_size=len(tokenizer.word_index)+1
    multi_modal=Multimodal()

    

    all_humours=["humour","motivational","offensive","sarcasm"]

    for H in all_humours:
        block_1,block_2,block_3=multi_modal.text_image_residual_network(tokenizer,np.array(tokenized_texts),img_arr,y[H],np.array(tokenized_val_text),img_arr_valid,y_valid[H],vocab_size,300,1,1,1,3,5,(3,3),(2,2),2,False,model_name=f"residual_network_3_{H}.h5")
        block_1.save(f"block_1_1_{H}.h5")
        block_2.save(f"block_2_1_{H}.h5")
        block_3.save(f"block_3_1_{H}.h5")



        block_1=load_keras_model(f"block_1_1_{H}.h5")
        block_2=load_keras_model(f"block_2_1_{H}.h5")
        block_3=load_keras_model(f"block_3_1_{H}.h5")

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

        turn_numpy_df(fea_train_1,y[H],f"residual_block_1_{H}_train.csv")
        turn_numpy_df(fea_val_1,y_valid[H],f"residual_block_1_{H}_val.csv")
        turn_numpy_df(fea_test_1,y_test[H],f"residual_block_1_{H}_test.csv")

        fea_train_2=block_2.predict([X_emb_glove,img_arr])
        fea_val_2=block_2.predict([X_emb_glove_val,img_arr_valid])
        fea_test_2=block_2.predict([X_emb_glove_test,img_arr_test])
        scaler=MinMaxScaler()
        fea_train_2=scaler.fit_transform(fea_train_2)
        fea_val_2=scaler.transform(fea_val_2)
        fea_test_2=scaler.transform(fea_test_2)

        turn_numpy_df(fea_train_2,y[H],f"residual_block_2_{H}_train.csv")
        turn_numpy_df(fea_val_2,y_valid[H],f"residual_block_2_{H}_val.csv")
        turn_numpy_df(fea_test_2,y_test[H],f"residual_block_2_{H}_test.csv")

        fea_train_3=block_3.predict([X_emb_glove,img_arr])
        fea_val_3=block_3.predict([X_emb_glove_val,img_arr_valid])
        fea_test_3=block_3.predict([X_emb_glove_test,img_arr_test])
        scaler=MinMaxScaler()
        fea_train_3=scaler.fit_transform(fea_train_3)
        fea_val_3=scaler.transform(fea_val_3)
        fea_test_3=scaler.transform(fea_test_3)

        turn_numpy_df(fea_train_3,y[H],f"residual_block_3_{H}_train.csv")
        turn_numpy_df(fea_val_3,y_valid[H],f"residual_block_3_{H}_val.csv")
        turn_numpy_df(fea_test_3,y_test[H],f"residual_block_3_{H}_test.csv")

