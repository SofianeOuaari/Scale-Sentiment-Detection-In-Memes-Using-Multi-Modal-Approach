import os
import pickle
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
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
    stop_words = get_stop_words('hu')
    cleaned=""
    for token in tokens: 
        if token not in stop_words:
            #cleaned+=stemmer.stemWord(token)+" "
            cleaned+=token+" "
            
    return cleaned

def change_to_three_sentiment_labels(y):
    return pd.Series(y).replace({"very_negative":"negative","very_positive":"positive"})
def split_train_test():
    df=pd.read_csv("memotion_dataset_7k/labels.csv")

    df_train,df_test=train_test_split(df,test_size=0.25)
    df_train.to_csv("memotion_dataset_7k/train.csv",index=True)
    df_test.to_csv("memotion_dataset_7k/test.csv",index=True)

def get_glove_embedding(dim,tokenizer,input_length):
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
    


def augment_text(text):
    TOPK=20 #default=100
    ACT = 'insert' #"substitute"
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

    aug_syn = naw.SynonymAug(aug_src='wordnet',aug_max=3)
    #aug_bert = naw.ContextualWordEmbsAug(
    #model_path='distilbert-base-uncased',action=ACT, top_k=TOPK)
    '''aug_w2v = naw.WordEmbsAug(
    model_type='glove', model_path='/content/glove.6B.300d.txt',
    action="substitute")'''
    #aug=naf.Sequential([aug_syn,aug_bert])
    #texts=aug.augment(text,n=10)
    texts=aug_syn.augment(clean_text(text),n=5)

    print(texts)



if __name__=="__main__":
    #augment_text("Hello How are you my friends, today I will present an important lecture about dynamic compounds")
    split_train_test()