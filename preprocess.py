from models import Unimodal
from utils import load_keras_tokenizer,load_np_array
import os
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



def mean_h_w():
    h_min=0
    w_min=0
    h_max=0
    w_max=0
    h=[]
    w=[]

    img_path="memotion_dataset_7k/images"
    for i in os.listdir(img_path):
        try:
            img_file_name=os.path.join(img_path,i)
            img_arr=cv2.imread(img_file_name)
            h.append(img_arr.shape[0])
            w.append(img_arr.shape[1])
            '''if h_max==0:
                h_max=img_arr.shape[0]
            else:
                h_max=max(h_max,img_arr.shape[0])
            if w_max==0:
                w_max=img_arr.shape[1]
            else:
                w_max=max(w_max,img_arr.shape[1])'''
        except Exception as e: 
            pass
    
    return np.array(h).mean(),np.array(w).mean(),np.array(h).std(),np.array(w).std()



def label_encoder(y):

    l_e=LabelEncoder()

    y_encoded=l_e.fit_transform(y)

    return y_encoded,l_e

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, (width,height), interpolation = inter)

    # return the resized image
    return resized

def return_rgb_images_text(h_size,w_size): 

    img_df=[]
    img_name=[]
    texts=[]
    df=pd.read_csv("memotion_dataset_7k/labels.csv")
    y=[]
    img_path="memotion_dataset_7k/images/"
    for i in range(len(df.image_name)):

    #for i in range(1000):
        try:
            
            img_file_name=os.path.join(img_path,df.image_name.iloc[i])
            
            
            img_arr=cv2.imread(img_file_name)
            img_arr=cv2.cvtColor(img_arr,cv2.COLOR_BGR2RGB)
            img_arr=image_resize(img_arr,width=w_size,height=h_size)
            print(img_arr.shape)
            img_df.append(img_arr)
            img_name.append(df.image_name.iloc[i])
            texts.append(df.text_corrected.iloc[i])
            y.append(df.overall_sentiment.iloc[i])
            print(img_file_name)
        except Exception as e: 
            print(e)
            #assert False
            pass
    
    return np.array(img_df),np.array(img_name),np.array(texts),np.array(y)

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
    
    return tokenizer


def get_tokenized_padded_text(tokenizer,texts,max_length):
    

    sequences=tokenizer.texts_to_sequences(texts)
    sequences=pad_sequences(sequences,maxlen=max_length)

    return sequences
def preprocess_text(text,max_nb_words,max_length,file_tokenizer_name):

    #tokenizer=load_keras_tokenizer("keras_text_tokenizer.pickle")

    str_transform=lambda x:str(x)
    print(text.shape)
    #text=str_transform(text)

    print(text.shape)

    tokenizer=create_save_keras_tokenizer(text,max_nb_words,file_tokenizer_name)

    tokenized=get_tokenized_padded_text(tokenizer,text,max_length)
    print(tokenized.shape)

    return tokenized,tokenizer


if __name__=="__main__":

    '''print(tf.test.is_gpu_available())
    print(tf.config.list_physical_devices('GPU'))'''
    #assert False

    #print(min_max_h_w())
    #assert(False)

    '''h_size,w_size,h_std,w_std=mean_h_w()
    print(h_size)
    print(w_size)
    print(h_std)
    print(w_std)'''

    '''img_arr,img_name,texts,y=return_rgb_images_text(300,300)
    print(img_arr.shape)'''

    #save_np_array("img_df_label.npy",y)

    '''df=pd.read_csv("memotion_dataset_7k/labels.csv")
    df.text_corrected=df.text_corrected.apply(lambda x:str(x))
    tokenizer=load_keras_tokenizer("keras_text_tokenizer.pickle")
    print(len(tokenizer.word_index))
    vocab_size=len(tokenizer.word_index)+1
    print(tf.test.is_gpu_available())
    #assert False
    tokenized=get_tokenized_padded_text(tokenizer,df.text_corrected,15)

    unimo=Unimodal()

    print(unimo.word_embedding_lstm(tokenized,vocab_size,300,5,5,3).summary())
    
    encoded_label=LabelEncoder().fit_transform(df.overall_sentiment)
    w_e_lstm=unimo.word_embedding_lstm(tokenized,vocab_size,300,5,5,5)'''

    '''unimo=Unimodal()

    #labels=load_np_array('img_df_label.npy')

    encoded_label=LabelEncoder().fit_transform(y)
    
    print(encoded_label)'''
    #w_e_lstm.fit(tokenized,encoded_label,batch_size=32,epochs=15)

    '''np_load_old = np.load

    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    img_arr=load_np_array("img_df.npy")'''
    #print(img_arr.shape)

    '''plt.imshow(img_arr[0])
    plt.show()
    #print(np.reshape(img_arr,(len(img_arr),500,735)))
    #cnn_model=unimo.image_cnn_model(img_arr,2,2,5,(3,3),(2,2),number_labels=5)
    resnet_model=unimo.image_resnet_model(img_arr,3,5)
    vgg16_model=unimo.image_vgg16_model(img_arr,1,5)
    vgg19_model=unimo.image_vgg19_model(img_arr,1,5)'''
    #print(cnn_model.summary())

    #cnn_model.fit(img_arr,encoded_label,batch_size=32,epochs=15,validation_split=0.25)
    #resnet_model.fit(img_arr,encoded_label,batch_size=4,epochs=15,validation_split=0.25)
    #vgg16_model.fit(img_arr,encoded_label,batch_size=4,epochs=15,validation_split=0.25)
    #vgg19_model.fit(img_arr,encoded_label,batch_size=2,epochs=15,validation_split=0.25)