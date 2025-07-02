import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# --- Configuration ---
# Path to your trained model. Make sure this matches where you saved it.
MODEL_PATH = 'steganalysis_cnn_model.h5'
# The image size your model was trained on. Must match IMAGE_SIZE in the training script.
IMAGE_SIZE = (128, 128)

# --- Streamlit App Layout (MUST BE FIRST STREAMLIT CALL) ---
st.set_page_config(
    page_title="Steganalysis Detector",
    page_icon="🕵️‍♀️",
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("🕵️‍♀️ Steganalysis Detector")
st.markdown("""
    Upload an image to detect if it contains hidden steganographic data.
    This tool uses a Convolutional Neural Network (CNN) trained to identify subtle patterns
    introduced by steganography.
""")

st.markdown("---")


# --- Load the trained model (Moved AFTER set_page_config) ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_steganalysis_model():
    """Loads the pre-trained CNN model."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}. Please ensure you have trained the model and saved it.")
        st.stop() # Stop the app if model is not found
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_steganalysis_model()


# --- Preprocessing function for uploaded images ---
def preprocess_image(img, target_size=IMAGE_SIZE):
    """
    Preprocesses an uploaded PIL Image for model prediction.
    Resizes, converts to numpy array, and normalizes pixel values.
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img)
    # Normalize pixel values to [0, 1] as done during training
    img_array = img_array.astype('float32') / 255.0
    # Add batch dimension (1, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    st.info("Analyzing image...")

    try:
        # Load and preprocess the image
        img = Image.open(uploaded_file)
        processed_img = preprocess_image(img)

        # Make prediction
        prediction = model.predict(processed_img)
        # The model outputs a probability (0 to 1).
        # Closer to 0 means cover image, closer to 1 means stego image.
        stego_probability = prediction[0][0]

        st.markdown("### Prediction Result:")

        if stego_probability > 0.5: # Threshold for classification
            st.success(f"**Stego Image Detected!**")
            st.write(f"Confidence: **{stego_probability:.2f}** (indicating likelihood of hidden data)")
            st.markdown("""
                This image likely contains hidden steganographic data.
                The model has identified patterns consistent with steganographic embedding.
            """)
        else:
            st.info(f"**Cover Image (No Stego Data Detected)**")
            st.write(f"Confidence: **{1 - stego_probability:.2f}** (indicating likelihood of no hidden data)")
            st.markdown("""
                This image appears to be a normal cover image with no detectable steganographic data.
            """)

        st.markdown("---")
        st.markdown("#### Technical Details:")
        st.write(f"Raw prediction score: `{stego_probability:.4f}`")
        st.write(f"Image processed to size: `{IMAGE_SIZE}`")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please ensure the uploaded file is a valid image and the model is correctly loaded.")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and TensorFlow.")
