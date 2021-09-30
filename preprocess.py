from models import Unimodal
from utils import load_keras_tokenizer
import os
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def min_max_h_w():
    h_min=0
    w_min=0
    h_max=0
    w_max=0

    img_path="memotion_dataset_7k/images"
    for i in os.listdir(img_path):
        try:
            img_file_name=os.path.join(img_path,i)
            img_arr=cv2.imread(img_file_name)
            if h_min==0:
                h_min=img_arr.shape[0]
            else:
                h_min=min(h_min,img_arr)
            if w_min==0:
                w_min=img_arr.shape[1]
            else:
                w_min=min(w_min,img_arr.shape[1])
        except Exception as e: 
            pass
    
    return h_min,w_min
def return_rgb_images(h_size,w_size): 

    img_df=[]
    img_name=[]
    df=pd.read_csv("memotion_dataset_7k/labels.csv")

    img_path="memotion_dataset_7k/images/"
    for i in df.image_name:
        try:
            img_file_name=os.path.join(img_path,i)
            img_name.append(i)
            
            img_arr=cv2.imread(img_file_name)
            img_arr=cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
            img_arr=np.resize(img_arr,(h_size,w_size))
            img_df.append(img_arr)
            print(img_file_name)
        except Exception as e: 
            pass
    
    return np.array(img_df),np.array(img_name)

def save_np_array(file_name,np_array):
    with open(file_name,"wb") as f:
        np.save(f,np_array)

def load_np_array(file_name):
    with open(file_name,"rb") as f: 
        np_array=np.load(f)
    return np_array
def create_save_keras_tokenizer(train,max_nb_words,file_tokenizer_name):

    tokenizer=Tokenizer(num_words=max_nb_words,lower=True)
    tokenizer.fit_on_texts(train)
    with open(file_tokenizer_name, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_tokenized_padded_text(tokenizer,texts,max_length):
    

    sequences=tokenizer.texts_to_sequences(texts)
    sequences=pad_sequences(sequences,maxlen=max_length)

    return sequences


if __name__=="__main__":

    df=pd.read_csv("memotion_dataset_7k/labels.csv")
    df.text_corrected=df.text_corrected.apply(lambda x:str(x))
    tokenizer=load_keras_tokenizer("keras_text_tokenizer.pickle")
    print(tokenizer)
    tokenized=get_tokenized_padded_text(tokenizer,df.text_corrected,15)

    unimo=Unimodal()

    print(unimo.word_embedding_lstm(tokenized,5,5,3).summary())
