# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 16:24:51 2023

@author: Jeevika
"""

import numpy as np
import pandas as pd
import pickle
import streamlit as st
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer

#loading in the model the vectorizer file
pkl_in = open("vectorizer.pkl", 'rb')
loaded_vectorizer = pickle.load(pkl_in)

# loading in the model to predict on the data
pickle_in = open("emotions.pkl", 'rb')
classifier = pickle.load(pickle_in)


def welcome():
    return "Welcome All"

def Emotion_prediction(Comments):
    
    prediction=classifier.predict(loaded_vectorizer.transform([Comments]))
    print(prediction)
    
    if (prediction[0] == 0):
        return 'Emotion - Neutral'
    else:
        return 'Emotion -Positive'

def main():
    st.title('emotion Prediction Web App')
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Emotion Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    Comments = st.text_input("Comments", "Type Here")
    
    Emotion = ''
    
    if st.button('Emotion Result'):
        Emotion = Emotion_prediction(Comments)
        
    st.success(Emotion)
    
    
    
    
if __name__ == '__main__':
    main()

 




