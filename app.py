import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# ✅ Fixed: Removed invalid argument `allow_output_mutation`
@st.cache_resource
def load_model(path):
    # Xception Model
    xception_model = tf.keras.models.Sequential([
        tf.keras.applications.xception.Xception(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # DenseNet Model
    densenet_model = tf.keras.models.Sequential([
        tf.keras.applications.densenet.DenseNet121(include_top=False, weights='imagenet', input_shape=(512, 512, 3)),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(4, activation='softmax')
    ])

    # Ensembling the Models
    inputs = tf.keras.Input(shape=(512, 512, 3))
    xception_output = xception_model(inputs)
    densenet_output = densenet_model(inputs)
    outputs = tf.keras.layers.average([densenet_output, xception_output])
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Loading the Weights of Model
    model.load_weights(path)

    return model

# Hiding Streamlit Menu and Footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Load the model (weights should be in 'model.h5')
model = load_model('model.h5')

# Title and Description
st.title('🌿 Plant Disease Detection')
st.write("Upload your plant's leaf image to detect if it's healthy or affected by a disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Crunching image...")
    my_bar = st.progress(0)

    # Step 1: Read and display uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(image.resize((700, 400), Image.Resampling.LANCZOS)), caption="Uploaded Image", use_column_width=True)
    my_bar.progress(20)

    # Step 2: Preprocess image
    image = clean_image(image)
    my_bar.progress(40)

    # Step 3: Run model prediction
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(70)

    # Step 4: Format result
    result = make_results(predictions, predictions_arr)

    # Cleanup progress
    progress.empty()
    my_bar.empty()

    # Step 5: Display result
    st.success(f"The plant is **{result['status']}** with a prediction of **{result['prediction']}**.")
