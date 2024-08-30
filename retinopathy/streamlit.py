import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import streamlit as st
from PIL import Image

# Setting the style for the app
st.set_page_config(
    page_title="Diabetic Retinopathy Classification",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="auto"
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: #f2f2f2;
    }
    .sidebar .sidebar-content {
        background: #f2f2f2;
    }
    h1 {
        color: #2C3E50;
        font-family: 'Arial Black', Gadget, sans-serif;
        font-size: 36px;
    }
    h2 {
        color: #3498DB;
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #3498DB;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #2980B9;
        color: white;
    }
    .stFileUploader {
        background-color: #ECF0F1;
        border-radius: 10px;
    }
    .stTextInput>div>input {
        background-color: #ECF0F1;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def preprocess_image(image, target_size=(224, 224)):
    """Load and preprocess an image."""
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  
    img = cv2.resize(img, target_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) 
    img = img / 255.0  
    return img

def classify_image(model, img):
    """Classify an image using the provided model."""
    prediction = model.predict(img)
    class_idx = np.argmax(prediction, axis=1)[0]
    return class_idx, prediction

def generate_heatmap(model, img, class_idx, layer_name="conv5_block3_out"):
    """Generate a heatmap using Grad-CAM."""
    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = tf.maximum(heatmap, 0).numpy()
    heatmap /= np.max(heatmap)  # Convert to NumPy array before normalization
    return heatmap

def overlay_heatmap(heatmap, img, alpha=0.4):
    """Overlay the heatmap on the original image."""
    img = np.array(img)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
    return superimposed_img

def process_and_display_image(model, image):
    """Process an image, classify it, generate heatmap, and display results."""
    img = preprocess_image(image)
    class_idx, _ = classify_image(model, img)
    
    class_names = ['No Diabetic Retinopathy', 'Diabetic Retinopathy']
    st.subheader(f"Classification Result: **{class_names[class_idx]}**")

    heatmap = generate_heatmap(model, img, class_idx)
    img_with_heatmap = overlay_heatmap(heatmap, image)
    
    # Display images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)
    
    with col2:
        st.image(img_with_heatmap, caption=f"Heatmap Overlay", use_column_width=True)

# Streamlit app layout and interactions
st.title("Diabetic Retinopathy Classification")
st.markdown("**Upload an image to determine whether it shows signs of Diabetic Retinopathy.**")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image', use_column_width=True)
    # st.write("")
    
    model_path = st.text_input("Enter the path to the model (.h5 file):", value="/media/ava/DATA2/rayari/other/kag/RES_weights.h5")
    
    if st.button('Classify and Show Heatmap'):
        with st.spinner('Classifying...'):
            model = load_model(model_path)
            process_and_display_image(model, image)





