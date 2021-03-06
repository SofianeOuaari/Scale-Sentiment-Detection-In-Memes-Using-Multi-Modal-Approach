import os,codecs
import pickle
import joblib
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential,load_model
from keras.layers import Embedding
from keras.initializers import Constant
import nltk
import nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.flow.sequential as naf
import snowballstemmer
from stop_words import get_stop_words




def load_np_array(file_name):
    with open(file_name,"rb") as f: 
        np_array=np.load(f)
    return np_array

def load_keras_tokenizer(file_name):
    with open(file_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return tokenizer

def load_keras_model(file_name):
    model=load_model(file_name)

    return model


def save_label_encoder(encoder,filename):
    joblib.dump(encoder, filename)

def load_label_encoder(filename):
    encoder=joblib.load(filename)

    return encoder
def save_label_encoder(encoder,filename):
    joblib.dump(encoder, filename)

def turn_numpy_df(num_arr,y,filename):

    df=pd.DataFrame(num_arr)
    print(df.head())
    print(df.columns)
    df["target"]=y

    df.to_csv(filename,index=False)
    
def clean_text(string):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    stemmer = snowballstemmer.stemmer('english');

    no_punct = ""
    for char in string:
        if char not in punctuations:
            no_punct = no_punct + char
    res = ''.join([i for i in no_punct if not i.isdigit()])
    res=res.strip('\n')
    res=res.strip('\t')
    r = re.compile(r'[\n\r\t]')
    res = r.sub(" ", res)
    res=res.lower()
    tokens=res.split(" ")
    stop_words = get_stop_words('en')
    cleaned=""
    for token in tokens: 
        if token not in stop_words:
            #cleaned+=stemmer.stemWord(token)+" "
            cleaned+=token+" "
            
    return cleaned

def change_to_three_sentiment_labels(y):
    return pd.Series(y).replace({"very_negative":"negative","very_positive":"positive"})

def change_binary_classes(y_humour): 
    y_humour.humour=y_humour.humour.replace({"not_funny":"not_humorous","funny":"humorous","very_funny":"humorous","hilarious":"humorous"})
    y_humour.offensive=y_humour.offensive.replace({"slight":"offensive","very_offensive":"offensive","hateful_offensive":"offensive"})
    y_humour.sarcasm=y_humour.sarcasm.replace({"general":"sarcastic","twisted_meaning":"sarcastic","very_twisted":"sarcastic"})
    return y_humour
def split_train_valid_test():
    df=pd.read_csv("memotion_dataset_7k/labels.csv")

    df_train,df_test=train_test_split(df,test_size=0.25,stratify=df.overall_sentiment)
    df_test,df_valid=train_test_split(df_test,test_size=0.35,stratify=df_test.overall_sentiment)
    df_train.to_csv("memotion_dataset_7k/train.csv",index=True)
    df_valid.to_csv("memotion_dataset_7k/valid.csv",index=True)
    df_test.to_csv("memotion_dataset_7k/test.csv",index=True)

def get_glove_embedding_glove(dim,tokenizer,input_length):
    embeddings_index = {}
    word_index = tokenizer.word_index
    f = open(os.path.join("gloves", f'glove.6B.{dim}d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model_emb = Sequential()
    embedding_layer = Embedding(len(word_index) + 1,dim,weights=[embedding_matrix],input_length=input_length,trainable=False)

    model_emb.add(embedding_layer)
    model_emb.compile('rmsprop', 'mse')
    return model_emb

def get_glove_embedding_fasttext(tokenizer,input_length):
    embeddings_index = {}
    dim=300
    word_index = tokenizer.word_index
    f = codecs.open(os.path.join("fasttext",'wiki.simple.vec'), errors = 'ignore',encoding="utf8")
    for line in f:
        try:
            values = line.split()
            word = values[0]
            #print(values[1:])
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except Exception as e:
            print(e)
            pass
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))
    embedding_matrix = np.zeros((len(word_index) + 1, dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model_emb = Sequential()
    embedding_layer = Embedding(len(word_index) + 1,dim,weights=[embedding_matrix],input_length=input_length,trainable=False)

    model_emb.add(embedding_layer)
    model_emb.compile('rmsprop', 'mse')
    return model_emb
    
def plot_color_spaces(hsv_img,lab_img,grey_img,ylcrcb_img):
    print(plt.cm.cmap_d.keys())
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    hsv = plt.cm.get_cmap('hsv')
    ax1.imshow(hsv_img,cmap=hsv)
    
    ax1.set_title("HSV")
    ax2.imshow(lab_img)
    ax2.set_title("LAB")
    ax3.imshow(grey_img,cmap=plt.cm.gray)
    ax3.set_title("GRAY")
    ax4.imshow(ylcrcb_img)
    ax4.set_title("YlCrCb")

    plt.show()

def augment_text(text):
    TOPK=20 #default=100
    ACT = 'insert' #"substitute"
    '''nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')'''

    aug_syn = naw.SynonymAug(aug_src='wordnet',aug_min=5,aug_max=15)
    #aug_bert = naw.ContextualWordEmbsAug(
    #model_path='distilbert-base-uncased',action=ACT, top_k=TOPK)
    '''aug_w2v = naw.WordEmbsAug(
    model_type='glove', model_path='/content/glove.6B.300d.txt',
    action="substitute")'''
    #aug=naf.Sequential([aug_syn,aug_bert])
    #texts=aug.augment(text,n=10)
    texts=aug_syn.augment(clean_text(text),n=3)

    print(texts)
    return texts

def augment_dataset(img_arr,img_name,texts,y,augment_text_only=False):
    img_arr_aug,img_name_aug,texts_aug,y_aug=[],[],[],[]
    val,counts=np.unique(y,return_counts=True)
    d_label={}

    for v,c in zip(val,counts):
        d_label[v]=c


    

    for i_arr,i_name,text,label in zip(img_arr,img_name,texts,y):
        (h, w) = i_arr.shape[:2]
        (cX, cY) = (w // 2, h // 2)
        
        #if not augment_text_only:
        img_arr_aug.append(i_arr)
        img_name_aug.append(i_name)
        texts_aug.append(text)
        y_aug.append(label)
        if d_label[label]<max(d_label.values()):
            generated_texts=augment_text(text)

            for t in generated_texts: 
                texts_aug.append(t)
                img_name_aug.append(i_name)
                y_aug.append(label)

                if not augment_text_only:
                    angle=np.random.randint(15,270)
            
                    M = cv2.getRotationMatrix2D((cX, cY),angle, 1.0)
                    rotated = cv2.warpAffine(i_arr, M, (w, h))
                    img_arr_aug.append(rotated)

                    
                    print(rotated.shape)
                    '''plt.imshow(rotated)
                    plt.show()'''
                else:
                    img_arr_aug.append(i_arr)


    return np.array(img_arr_aug),np.array(img_name_aug),np.array(texts_aug),np.array(y_aug)



if __name__=="__main__":
    #augment_text("Hello How are you my friends, today I will present an important lecture about dynamic compounds")
    split_train_valid_test()