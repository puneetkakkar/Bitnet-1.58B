import streamlit as st
import requests
from PIL import Image
import io

st.title('Bitnet 1.58b virtual storyteller')

FLASK_ENDPOINT = 'http://127.0.0.1:5000/predict'

user_input = st.text_input("Enter your prompt", "")

if st.button('Generate'):
    data = {'user_input': user_input}

    response = requests.post(FLASK_ENDPOINT, json=data)

    if response.status_code == 200:
        prediction = response.json()
        st.write(f'Generated Text: {prediction["text"]}')
    else:
        st.write("Failed to get a response from the server")


