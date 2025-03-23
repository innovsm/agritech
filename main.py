import streamlit as st
import cv2
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.utils import img_to_array
from keras.preprocessing import image

# Set page config
st.set_page_config(
    page_title="Coffee Leaf Disease Detection",
    page_icon="🍃",
    layout="wide"
)

# Disease information dictionary
disease_info = {
    'Cerscospora': {
        'description': 'Cercospora leaf spot is a fungal disease that appears as brown or black spots with yellow halos on coffee leaves.',
        'treatment': 'Apply copper-based fungicides. Ensure good air circulation by proper spacing and pruning. Remove and destroy infected leaves.'
    },
    'Healthy': {
        'description': 'This leaf appears healthy with no visible signs of disease.',
        'treatment': 'Continue regular maintenance and monitoring. Ensure proper nutrition, watering, and sunlight.'
    },
    'Leaf Rust': {
        'description': 'Coffee leaf rust is caused by the fungus Hemileia vastatrix, appearing as orange-yellow powdery spots on the undersides of leaves.',
        'treatment': 'Apply fungicides containing copper or systemic fungicides. Improve farm management with proper spacing and shade management. Consider resistant coffee varieties.'
    },
    'Minor': {
        'description': 'Minor damage or early stage infection that requires monitoring.',
        'treatment': 'Monitor the affected plants closely. Improve growing conditions and consider preventative fungicide applications.'
    },
    'Phoma': {
        'description': 'Phoma leaf spot causes small brown lesions that can expand into larger necrotic areas on coffee leaves.',
        'treatment': 'Apply copper-based fungicides. Improve drainage and reduce humidity. Remove and destroy infected plant material.'
    }
}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2E7D32;
        margin-top: 2rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    .healthy {
        background-color: #C8E6C9;
        border: 1px solid #2E7D32;
    }
    .diseased {
        background-color: #FFCDD2;
        border: 1px solid #C62828;
    }
    .info-text {
        font-size: 1.1rem;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Coffee Leaf Disease Detection System</h1>", unsafe_allow_html=True)

# Side Panel
with st.sidebar:
    st.image("sample_image.jpg", width=250, caption="Sample Coffee Leaf")
    
    st.markdown("## Instructions")
    st.markdown("""
    1. Upload an image of a coffee leaf or take a photo using the camera
    2. The system will analyze the leaf and identify any diseases
    3. View detailed information and treatment recommendations
    """)
    
    st.markdown("## About")
    st.markdown("""
    This application uses machine learning to detect common coffee leaf diseases:
    - Cercospora Leaf Spot
    - Coffee Leaf Rust
    - Phoma Leaf Spot
    - Minor Damage
    
    The model is trained on thousands of images to provide accurate disease detection.
    """)

# Create two columns for capture and upload options
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 class='sub-header'>Capture Leaf Image</h2>", unsafe_allow_html=True)
    data_image = st.camera_input(label="Take a photo of the coffee leaf")
    
    if data_image is not None:
        bytes_data = data_image.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        st.image(cv2_img, caption="Captured Image", use_column_width=True)
        
        # Resize for model
        cv2_img_resized = cv2.resize(cv2_img, (128, 128))
        
        # Predictions
        model_1 = load_model("model.h5")
        new_image = img_to_array(cv2_img_resized)
        new_image = np.expand_dims(new_image, axis=0)
        
        with st.spinner('Analyzing leaf...'):
            prediction = model_1.predict(new_image)
        
        classes = ['Cerscospora', 'Healthy', 'Leaf Rust', 'Minor', 'Phoma']
        result = classes[prediction.argmax(-1)[0]]
        
        # Display result with appropriate styling
        if result == 'Healthy':
            st.markdown(f"<div class='result-box healthy'><h3>Result: {result}</h3><p class='info-text'><b>Description:</b> {disease_info[result]['description']}</p><p class='info-text'><b>Recommendation:</b> {disease_info[result]['treatment']}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box diseased'><h3>Result: {result}</h3><p class='info-text'><b>Description:</b> {disease_info[result]['description']}</p><p class='info-text'><b>Treatment:</b> {disease_info[result]['treatment']}</p></div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h2 class='sub-header'>Upload Leaf Image</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a coffee leaf image", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file:
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
        
        st.image(cv2_img, caption="Uploaded Image", use_column_width=True)
        
        # Resize for model
        cv2_img_resized = cv2.resize(cv2_img, (128, 128))
        
        # Predictions
        model_1 = load_model("model.h5")
        new_image = img_to_array(cv2_img_resized)
        new_image = np.expand_dims(new_image, axis=0)
        
        with st.spinner('Analyzing leaf...'):
            prediction = model_1.predict(new_image)
        
        classes = ['Cerscospora', 'Healthy', 'Leaf Rust', 'Minor', 'Phoma']
        result = classes[prediction.argmax(-1)[0]]
        
        # Display result with appropriate styling
        if result == 'Healthy':
            st.markdown(f"<div class='result-box healthy'><h3>Result: {result}</h3><p class='info-text'><b>Description:</b> {disease_info[result]['description']}</p><p class='info-text'><b>Recommendation:</b> {disease_info[result]['treatment']}</p></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box diseased'><h3>Result: {result}</h3><p class='info-text'><b>Description:</b> {disease_info[result]['description']}</p><p class='info-text'><b>Treatment:</b> {disease_info[result]['treatment']}</p></div>", unsafe_allow_html=True)

# Add footer with additional information
st.markdown("---")
expander = st.expander("Learn More About Coffee Leaf Diseases")
with expander:
    st.markdown("""
    ### Common Coffee Leaf Diseases
    
    Coffee plants are susceptible to various diseases that can significantly impact yield and quality. Early detection and proper treatment are essential for maintaining healthy coffee plantations.
    
    #### Cercospora Leaf Spot
    Typically appears as brown or black spots with yellow halos. In severe cases, it can cause premature leaf drop.
    
    #### Coffee Leaf Rust
    One of the most devastating coffee diseases worldwide, causing orange-yellow powdery spots on leaf undersides. Severe infections lead to defoliation and yield loss.
    
    #### Phoma Leaf Spot
    Characterized by small brown lesions that can merge into larger necrotic areas. Common in high-altitude coffee farms.
    
    #### Prevention Tips
    - Maintain proper spacing between plants
    - Ensure good drainage and airflow
    - Apply preventative fungicides during wet seasons
    - Use resistant coffee varieties when possible
    - Practice regular field monitoring
    """)
