import streamlit as st
import cv2
import numpy as np
import pandas as pd
# all machine learning libraries
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image


with st.sidebar:
    st.write("Coffe Leaf Desease Detection System")
    st.subheader("Instructions")
    with st.expander(label="How to Use The App",expanded=True):
        st.write("Click the image of the Coffe Leaf or  upload the image of the sample \n Run the app to get the rsult")
        st.subheader("Upload Image Like this")
        st.image("sample_image.jpg")
        st.image("sample_image_2.jpg")
# cameera input

data_image = st.camera_input(label = "Take image of the leaf")


if data_image is not None:
    bytes_data = data_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    
    st.image(cv2_img)
    cv2_img = cv2.resize(cv2_img, (128, 128))
    # predictions
    model_1 = load_model("model.h5")
    new_image = img_to_array(cv2_img)
    new_image = np.expand_dims(new_image, axis = 0)
    prediction = model_1.predict(new_image)
    local_database = ['Cerscospora','Healthy','Leaf Rust','Minor','Phoma']
    st.write(local_database[prediction.argmax(-1)[0]])

st.subheader("Upload coffe leaf sample")
uploaded_file = st.file_uploader("Choose a coffe leaf image", type = ['jpg', 'png', 'jpeg'])
if(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    cv2_img = cv2.resize(cv2_img, (128, 128))
    st.image(cv2_img)
    # predictions
    model_1 = load_model("model.h5")
    new_image = img_to_array(cv2_img)
    new_image = np.expand_dims(new_image, axis = 0)
    prediction = model_1.predict(new_image)
    local_database = ['Cerscospora','Healthy','Leaf Rust','Minor','Phoma']
    st.write(local_database[prediction.argmax(-1)[0]])

