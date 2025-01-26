import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


max_length = 100  


import helper
st.title("sentmint analysis for movie review app")


tf = pickle.load(open("artifacts/tf.pkl","rb"))
model = load_model('artifacts/model_imdb.keras')

review = st.text_input("Enter your review here:")


if st.button("result"):
    if review:  
        review = helper.normalize_text(review)
        review = helper.text2words(review)
        review = tf.texts_to_sequences([review])
        review = pad_sequences(review, maxlen=max_length, padding='post')
        pred = model.predict(review)
        if   pred > 0.5 :
              st.success(f"this is good review")
        else:
              st.success(f"this is bad review")
    else:
        st.warning("Please enter review again.")


