# Coffee Leaf Disease Detection System

An AI-powered application for detecting common diseases in coffee plants using image recognition.

## Features

- **Real-time Detection**: Analyze coffee leaf images captured from camera or uploaded files
- **Multiple Disease Detection**: Identifies Cercospora, Leaf Rust, Phoma, and other common issues
- **Treatment Recommendations**: Provides specific treatment advice for each detected disease
- **User-friendly Interface**: Clean and intuitive UI designed for farmers and agriculture specialists

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/innovsm/agritech.git
   cd agritech
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run main.py
   ```

## Usage

1. Launch the application using the command above
2. Use your device camera to capture an image of a coffee leaf or upload an existing image
3. The system will analyze the leaf and display the results
4. Review the detailed information about the detected disease and recommended treatments

## Technologies Used

- Streamlit for the user interface
- TensorFlow/Keras for the machine learning model
- OpenCV for image processing
- Python 3.7+

## Model Information

The disease detection model is trained on a dataset of coffee leaf images showing various diseases and healthy leaves. The model achieves approximately 90% accuracy in identifying the following conditions:

- Cercospora Leaf Spot
- Coffee Leaf Rust
- Phoma Leaf Spot
- Minor Damage
- Healthy Leaves

## License

This project is licensed under the MIT License - see the LICENSE file for details.
