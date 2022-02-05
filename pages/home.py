import streamlit as st 
from PIL import Image



def display_all_tech_logos():
    col1,col2,col3=st.columns(3)
    with col1: 
        img_1=Image.open("Images/Tech_Logos/python.png")
        st.image(img_1,width=80)
    with col2:
        img_2=Image.open("Images/Tech_Logos/numpy.png")
        st.image(img_2,width=150)
    with col3:
        img_3=Image.open("Images/Tech_Logos/pandas.jpeg")
        st.image(img_3,width=150)
    
    col4,col5,col6,col7=st.columns(4)
    with col4: 
        img_4=Image.open("Images/Tech_Logos/opencv.png")
        st.image(img_4,width=80)
    with col5:
        img_5=Image.open("Images/Tech_Logos/sklearn.jpg")
        st.image(img_5,width=150)
    with col6:
        img_6=Image.open("Images/Tech_Logos/nltk.png")
        st.image(img_6,width=150)
    with col7: 
        img_7=Image.open("Images/Tech_Logos/tf.png")
        st.image(img_7,width=150)


def app():
    st.markdown("<html><body><center><h1>Data Analysis and ML-Powered Framework</h1></center></body></html>",unsafe_allow_html=True)
    st.markdown("<html><body><center><p>This project is part of my master thesis work within the MSc Data Science specialization at Eötvös Lorand University (ELTE). The topic of my master thesis is entitled \"Scale Sentiment Analysis Using Multimodal Learning For Memes\" where the aim is to explore the capabilities of expoliting the 2 modalties explicitly provided in a meme namely the text and the image, in addition to extracting and withdrawing further inputs. In parallel to the various research I am doing for this interesting topic, I decided to implement as well a data-driven framework with a simple to use GUT to allow users to not only test my models on memes and check the available sentiment, but also provide automated tools to apply preprocessing operations on text data as well as different types of text augmentation. Also there is the possibility to build deep learning models by easily and dynamically adding/deleting layers to it.</p></center></body></html>",unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)

    with col1: 
        st.markdown("<html><body><center><h6>Natural Language Processing Tools</h6></center></body></html>",unsafe_allow_html=True)
        img_nlp=Image.open("Images/NLP_Img.jpg")
        st.image(img_nlp,use_column_width=True)
    with col2:
        st.markdown("<html><body><center><h6>Customizable Model Creatio,Training & Deployement</h6></center></body></html>",unsafe_allow_html=True)
        img_nn=Image.open("Images/NN_Img.png")
        st.image(img_nn,use_column_width=True)
    with col3:
        st.markdown("<html><body><center><h6>DL-Powered Sentiment Identifiction In Memes</h6></center></body></html>",unsafe_allow_html=True)
        img_ssmm=Image.open("Images/SSMM_Img.jpg")
        st.image(img_ssmm,use_column_width=True)
    
    st.markdown("<html><body><center><br><br><strong>Powered By</strong></center></body></html>",unsafe_allow_html=True)

    #display_all_tech_logos()
    col1,col2,col3=st.columns(3)

    with col1: 
        st.text("")
    with col2:
        img_1=Image.open("Images/Tech_Logos/All_Techs.jpg")
        st.image(img_1,use_column_width=True)
    with col3:
        st.text(" ")



