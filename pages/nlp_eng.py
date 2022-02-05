import streamlit as st
import base64
import time
import random
import numpy as np
import pandas as pd
import nlpaug
from nlpaug.util.file.download import DownloadUtil
import nlpaug.augmenter.word as naw
import nlpaug.flow.sequential as naf
from googletrans import Translator


class FileDownloader(object):
    
	
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
		new_filename = "{}_{}_.{}".format(self.filename,time.strftime("%Y%m%d-%H%M%S"),self.file_ext)
		href = f'<a href="data:file/{self.file_ext};base64,{b64}" download="{new_filename}">Download File</a>'
		st.markdown(href,unsafe_allow_html=True)

def augment_synonym(df,min,max,n_aug):
    aug_syn = naw.SynonymAug(aug_src='wordnet',aug_min=int(min),aug_max=int(max))

    for i in range(len(df)):
        t=df.text.iloc[i]
        l=df.label.iloc[i]
        texts=aug_syn.augment(str(t),n=int(n_aug))
        d_aug={"text":texts,"label":np.full(len(texts),int(l),dtype=int)}
        d_aug=pd.DataFrame(d_aug)
        print(d_aug)
        df=pd.concat([df,d_aug])

    return df.reset_index(drop=True)

def augment_with_translation(df,src_lang,n_aug):

    languages=['en','fr','it','es','de','hu','cs','ar']
    translator=Translator()
    for i in range(len(df)):
        t=df.text.iloc[i]
        l=df.label.iloc[i]
        texts=[]

        for _ in range(int(n_aug)-1):
            dst_lang=src_lang
            while dst_lang==src_lang:
                dst_lang=random.choice(languages)
            translation=translator.translate(t,src=src_lang,dest=dst_lang)
            print(translation.text)
            translation_back=translator.translate(translation.text,src=dst_lang,dst=src_lang)
            texts.append(translation_back.text)
            d_aug={"text":texts,"label":np.full(len(texts),int(l),dtype=int)}
            d_aug=pd.DataFrame(d_aug)
            print(d_aug)
            df=pd.concat([df,d_aug])
    return df.reset_index(drop=True)

def augment_with_contextual_embedding(df,emb,action,src_lang,n_aug):
    
    for i in range(len(df)):
        t=df.text.iloc[i]
        l=df.label.iloc[i]
        texts=[]

        for _ in range(int(n_aug)-1):
            if emb=="Word2vec":
                pass
            elif emb=="Glove":
                aug_glove = naw.WordEmbsAug(model_type='glove', model_path='gloves/glove.6B.300d.txt',action=action)
                texts=aug_glove.augment(str(t),n=int(n_aug))
            elif emb=="Fasttext":
                aug_fasttext = naw.WordEmbsAug(model_type='fasttext', model_path=f'fasttext/wiki.{src_lang}.vec',action=action)
                texts=aug_fasttext.augment(str(t),n=int(n_aug))

            d_aug={"text":texts,"label":np.full(len(texts),int(l),dtype=int)}
            d_aug=pd.DataFrame(d_aug)
            print(d_aug)
            df=pd.concat([df,d_aug])

    return df.reset_index(drop=True)




def app():
    st.markdown("<html><body><center><h1>Natural Language Processing Automation</h1></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><center><p>Texts are unstructured type of data which is present everywhere through the internet and more specifically in social media. Many Machine Learning tasks can be performed on text data such as detecting sentiments, classifying articles to their respective categories to even identifying given trends for a stock market by analyzing tweets of famous traders through plateforms like StockTwits. However, models cannot directly train with raw text that is why some preprocessing needs to be done in order to turn raw characters to numerical formats and also withdraw hidden features. In this section we will provided automated tools to apply different kind of transformations usually done on NLP tasks.</p></center></body></html>",unsafe_allow_html=True)

    st.markdown("<html><body><h3>Text Augmentation</h3></body></html>",unsafe_allow_html=True)
    st.markdown("Many data-specific scenarios may negatively impact the performance of the Machine Learning models. Lack of data will result in poor performance and no generalizability will be achieved, while data imbalance will lead to models biased towards the majority classes and neglecting minority ones. **Data Augmentation** allows to generate more data in our case more text to overcome both situations mentioned above.")

    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is not None:
        df=pd.read_csv(data)
        st.dataframe(df,width=600,height=500)
        augmentation_type=st.selectbox("Text Augmentation Type",('Synonym-Based','Embedding-Based','Translation'))

        if augmentation_type=="Synonym-Based": 

            st.write("Please Select the Min/Max of synonyms and the number of augmented text per instance")
            col1, col2, col3 = st.columns(3)
            with col1:
                min_sys=st.text_input("Min synonyms",5)

            with col2:
                max_sys=st.text_input("Max synonyms",15)

            with col3:
                n_aug=st.text_input("Number of Augmentation/Sentence",2)
            
            aug_button=st.button("Augment")

            if aug_button:
                df_aug=None
                df_aug=augment_synonym(df,min_sys,max_sys,n_aug)
                if df_aug is not None: 
                    st.markdown("<html><body><center><p>Augmented Dataset</p></center></body></html>",unsafe_allow_html=True)
                    st.dataframe(df_aug)
                    download = FileDownloader(df_aug.to_csv(),file_ext='csv').download()


        elif augmentation_type=="Translation":
            st.write("Please select the source language")
            col1, col2 = st.columns(2)
            with col1:
                src_lang=st.selectbox("Source Language",('en','fr','it','es','de','hu','cs','ar'))

            with col2:
                n_aug=st.text_input("Number of Augmentation/Sentence",2)
            
            aug_button=st.button("Augment")

            if aug_button:
                df_aug=None
                df_aug=augment_with_translation(df.iloc[:10],src_lang,n_aug)
                if df_aug is not None: 
                    st.markdown("<html><body><center><p>Augmented Dataset</p></center></body></html>",unsafe_allow_html=True)
                    st.dataframe(df_aug)
                    download = FileDownloader(df_aug.to_csv(),file_ext='csv').download()
        elif augmentation_type=='Embedding-Based':
            col1,col2,col3,col4=st.columns(4)
            src_lang=None
            with col1:
                embedding=st.selectbox("Embedding type ",('Word2Vec','Glove','Fasttext'))
            with col2:
                action=st.selectbox("Augmentation action \n \t",('insert','substitute'))
            with col3:
                n_aug=st.text_input("Number of Augmentation \n \t",2)
            if embedding=="Fasttext":
                with col4: 
                    src_lang=st.selectbox('Language',('en','fr','it','es','de','hu','cs','ar'))
            
            aug_button=st.button("Augment")
            if aug_button:
                df_aug=None
                df_aug=augment_with_contextual_embedding(df.iloc[:10],embedding,action,src_lang,n_aug)
                if df_aug is not None: 
                    st.markdown("<html><body><center><p>Augmented Dataset</p></center></body></html>",unsafe_allow_html=True)
                    st.dataframe(df_aug)
                    download = FileDownloader(df_aug.to_csv(),file_ext='csv').download()

                

            

            

    st.markdown("<html><body><h3>Text Stemming/Lemming</h3></body></html>",unsafe_allow_html=True)







