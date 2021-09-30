import os
import pickle
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import cv2
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences




def load_np_array(file_name):
    with open(file_name,"rb") as f: 
        np_array=np.load(f)
    return np_array

def load_keras_tokenizer(file_name):
    with open(file_name, 'rb') as handle:
        tokenizer = pickle.load(handle)
    
    return tokenizer