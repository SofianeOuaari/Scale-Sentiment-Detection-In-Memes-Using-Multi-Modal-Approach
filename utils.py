import os
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import nltk
import nlpaug
import nlpaug.augmenter.word as naw
import nlpaug.flow.sequential as naf 




def load_np_array(file_name):
    with open(file_name,"rb") as f: 
        np_array=np.load(f)
    return np_array

def load_keras_tokenizer(file_name):
    with open(file_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return tokenizer


def clean_text(text):
    pass 


def augment_text(text):
    TOPK=20 #default=100
    ACT = 'insert' #"substitute"
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    aug_syn = naw.SynonymAug(aug_src='wordnet',aug_max=3)
    #aug_bert = naw.ContextualWordEmbsAug(
    #model_path='distilbert-base-uncased',action=ACT, top_k=TOPK)
    '''aug_w2v = naw.WordEmbsAug(
    model_type='glove', model_path='/content/glove.6B.300d.txt',
    action="substitute")'''
    #aug=naf.Sequential([aug_syn,aug_bert])
    #texts=aug.augment(text,n=10)
    texts=aug_syn.augment(text,n=10)

    print(texts)



if __name__=="__main__":
    augment_text("Hello How are you my friends, today I will present an important lecture about dynamic compounds")