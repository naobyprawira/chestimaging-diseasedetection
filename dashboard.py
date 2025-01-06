import streamlit as st  
import numpy as np
from PIL import Image, ImageOps 
from predict import import_and_predict  # Import the prediction function

# Set page configuration
st.set_page_config(
    page_title="Pneumonia and COVID Classification",
    page_icon=":mask:",
    initial_sidebar_state='auto'
)

# Hide Streamlit default elements (optional)
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Sidebar

# Main Page
st.write("""
# Pneumonia and COVID Classification
Upload a chest X-ray image to determine if the subject has Pneumonia, COVID-19, or is Healthy.
""")

# File uploader
file = st.file_uploader("", type=["jpg", "png", "jpeg"])

# Classification logic
if file is None:
    st.text("Please upload an X-ray image file.")
else:
    image = Image.open(file)
    st.image(image, width=400)
    predictions, detected_class = import_and_predict(image)

    # Display results
    st.markdown(f"### Result: **{detected_class}**")
    
    # Show model confidence below the detected class
    st.info(f"Model Confidence: {np.max(predictions) * 100:.2f}%")
    
    # Provide recommendations based on the detected class
    if detected_class == 'Healthy':
        st.balloons()
        st.success("The subject is healthy. No issues detected.")
    elif detected_class == 'Tuberculosis':
        st.warning("Tuberculosis detected.")
    elif detected_class == 'COVID-19':
        st.error("COVID-19 detected.")

    # # Optional: Add accuracy details
    # st.sidebar.info(f"Model Confidence: {np.max(predictions) * 100:.2f}%")
