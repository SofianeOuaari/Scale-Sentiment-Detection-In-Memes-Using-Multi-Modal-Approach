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
import snowballstemmer
from stop_words import get_stop_words

class FileDownloader(object):
    
	
	def __init__(self, data,filename='myfile',file_ext='txt'):
		super(FileDownloader, self).__init__()
		self.data = data
		self.filename = filename
		self.file_ext = file_ext

	def download(self):
		b64 = base64.b64encode(self.data.encode()).decode()
        #timestr = time.strftime("%Y%m%d-%H%M%S")
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
                #texts=aug_syn.augment(str(t),n=int(n_aug)
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


def return_stemmed_text(df,src_lang,take_off_stopwords=True):
    lang={'en':'english','fr':'french','it':'italian','es':'spanish','de':'german','hu':'hungarian','cs':'czech','ar':'arabic'}

    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    stemmer = snowballstemmer.stemmer(lang[src_lang])

    stemmed_text=[]
    print(df.text.tolist())
    for t in df.text.tolist():
        no_punct = ""
        for char in str(t):
            if char not in punctuations:
                no_punct = no_punct + char
        res = ''.join([i for i in no_punct if not i.isdigit()])
        res=res.lower()
        tokens=res.split(" ")
        stop_words=[]
        if take_off_stopwords:
            stop_words = get_stop_words(src_lang)
        cleaned=""
        for token in tokens: 
            if token not in stop_words:
                cleaned+=stemmer.stemWord(token)+" "

        stemmed_text.append(cleaned)
    df["stemmed_text"]=stemmed_text

    return df
            
    #return cleaned





def app():
    st.markdown("<html><body><center><h1>Natural Language Processing Automation</h1></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><center><p>Texts are unstructured type of data which is present everywhere through the internet and more specifically in social media. Many Machine Learning tasks can be performed on text data such as detecting sentiments, classifying articles to their respective categories to even identifying given trends for a stock market by analyzing tweets of famous traders through plateforms like StockTwits. However, models cannot directly train with raw text that is why some preprocessing needs to be done in order to turn raw characters to numerical formats and also withdraw hidden features. In this section we will provided automated tools to apply different kind of transformations usually done on NLP tasks.</p></center></body></html>",unsafe_allow_html=True)

    st.markdown("<html><body><h3>Text Augmentation</h3></body></html>",unsafe_allow_html=True)
    st.markdown("Many data-specific scenarios may negatively impact the performance of the Machine Learning models. Lack of data will result in poor performance and no generalizability will be achieved, while data imbalance will lead to models biased towards the majority classes and neglecting minority ones. **Data Augmentation** allows to generate more data in our case more text to overcome both situations mentioned above.")

    data = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    if data is not None:
        df=pd.read_csv(data)
        st.dataframe(df)
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
            '''DownloadUtil.download_word2vec(dest_dir='.')
            DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.')

            # download fasttext model
            DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.')'''

            #DownloadUtil.download_fasttext(model_name='wiki.en.vec', dest_dir='.')
            #aug_glove = naw.WordEmbsAug(model_type='glove', model_path='gloves/glove.6B.300d.txt',action="substitute")
            #aug_fasttext = naw.WordEmbsAug(model_type='fasttext', model_path='fasttext/wiki.en.vec',action="insert")
            '''print(len(df.text.iloc[0]))
            print(len(aug_fasttext.augment(df.text.iloc[0])))'''
            col1,col2,col3,col4=st.columns(4)
            src_lang=None
            with col1:
                #st.write("Please select the embedding approach to be used")
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

    st.markdown("Machine Learning models aims to detect key features to distinguish between different classes in case of supervised learning, and in order to simplify the task for the models to focus on more complex pattern learning we have to clean the data from unnecessary information. Within the NLP framework such operations are called **Text Stemming/Lemming**")
    st.markdown("**_Definition_** Stemming is a useful _normalization_ technique for words. n the context of machine learning based NLP, stemming makes your training data more dense. It reduces the size of the dictionary (number of words used in the corpus) two or three-fold. Having the same corpus, but less input dimensions, ML will work better.")

    data_stem = st.file_uploader("Upload a Dataset For Stemming", type=["csv", "txt"])

    if data_stem is not None:
        df_stem=pd.read_csv(data_stem)
        st.dataframe(df_stem)

        st.markdown("Please select the information about the stemming operation")

        col1, col2, col3,col4 = st.columns(4)
        with col1:
            pass
        with col2:
            src_lang=st.selectbox('Language',('en','fr','it','es','de','hu','cs','ar'))

        with col3:
            remove_stop_words=st.selectbox('Remove Stopwords',(True,False))
        with col4:
            pass
        
        stem_button=st.button("Apply Text Stemming & Cleaning")
        if stem_button:
            df_stem=return_stemmed_text(df_stem,src_lang,take_off_stopwords=remove_stop_words)
            st.dataframe(df_stem)
            download = FileDownloader(df_stem.to_csv(),file_ext='csv').download()









