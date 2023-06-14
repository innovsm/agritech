import streamlit as st
import cv2
import numpy as np
import pandas as pd
st.write("hello world")
# cameera input
data_image = st.camera_input(label = "hello world")


if data_image is not None:
    bytes_data = data_image.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    st.image(cv2_img)
