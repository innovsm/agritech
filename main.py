import streamlit as st

st.write("hello world")
# cameera input
data_1 = st.camera_input(label = "hello world")
location = st.session_state.location
if location is not None:
    st.write(location)
#st.write(data_1)