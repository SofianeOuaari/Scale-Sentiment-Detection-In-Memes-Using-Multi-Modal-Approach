import streamlit as st
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tomli import load
from preprocess import get_tokenized_padded_text,image_resize
from utils import load_keras_tokenizer,load_keras_model,get_glove_embedding_glove
import pytesseract



def app():
    st.markdown("<html><body><center><h1>Scale Sentiment Detection In Memes Using MultiModal Learning</h1></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><center><p>A meme is typically a photo or video with text on it which is currently widely spread in social media platforms like Facebook/Instagram / Twitter which express a culturally-relevant idea (an event,joke, critic). Eventhough memes were originally used for explicit humor and sarcasm, as many tools present on social media it may send negative messages and share hate.</p></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><center><p>In this section, mulimodal models are available to detect the overall sentiment ranging from very negative to very positive. Other models are available to detect more specific type of emotion (humour, sarcasm, offensive, motivational). \n All you have to do is to upload the image then an OCR software is available to read the text from the meme and then the model will either use the explicitly provided modalities namely the text and image or used other extracted patterns and features.</p></center></body></html>",unsafe_allow_html=True)
    pytesseract.pytesseract.tesseract_cmd=r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    WIDTH=224
    HEIGHT=224

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        image_to_text = pytesseract.image_to_string(image, lang='eng').replace("\n"," ")
        print(image_to_text)
        st.markdown(f"<html><body><center>OCR Extracted Text:<strong>{image_to_text}</strong></center></body></html>",unsafe_allow_html=True)
        tokenizer=load_keras_tokenizer("./text_tokenizer.pickle")
        tokenized_val_text=get_tokenized_padded_text(tokenizer,[image_to_text],25)
        print(tokenized_val_text)
        img_arr=np.array(image)
        img_arr=image_resize(img_arr,width=WIDTH,height=HEIGHT)
        print(img_arr.shape)

        residual_network=load_keras_model("residual_network.h5")

        

        emb=get_glove_embedding_glove(100,tokenizer,tokenized_val_text.shape[1])
        X_embedded_val=emb.predict(tokenized_val_text)

        preds=residual_network.predict([np.array(X_embedded_val),np.array([img_arr])])
        print(np.argmax(preds))
        st.write(f"Probabilities Distribution: {preds}")
        '''print(uploaded_file)
        print(cv2.imread(uploaded_file))'''
        '''plt.imshow(np.array(image))
        plt.show()'''