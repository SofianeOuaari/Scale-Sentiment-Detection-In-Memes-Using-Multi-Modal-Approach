import streamlit as st
import streamlit.components.v1 as components
from pages import home,nlp_eng,ssmml,custom_model
from PIL import Image
import base64
from zipfile import Path


PAGES = {
    "Home": home,
    "NLP": nlp_eng,
    "Custom Models":custom_model,
    "Scale-Sentiment Mulimodal Memes":ssmml
}

img=Image.open("Images/ELTElogo_trans.png")
st.sidebar.image(img,width=100)

st.sidebar.title("Contents")
selection=st.sidebar.radio("",options=PAGES.keys())
p=PAGES[selection]
p.app()




