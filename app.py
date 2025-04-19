import streamlit as st
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from utils import clean_image, get_prediction, make_results

# Loading the Model and saving to cache using `st.cache_resource` for model caching
@st.cache_resource(allow_output_mutation=True)
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

# Loading the Model
model = load_model('model.h5')

# Title and Description
st.title('Plant Disease Detection')
st.write("Just upload your plant's leaf image and get predictions on whether the plant is healthy or not.")

# Setting the file types that can be uploaded
uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

# If there is an uploaded file, start making predictions
if uploaded_file is not None:
    # Display progress and text
    progress = st.text("Crunching image...")
    my_bar = st.progress(0)
    i = 0

    # Reading the uploaded image
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(np.array(image.resize((700, 400), Image.Resampling.LANCZOS)), caption="Uploaded Image", use_column_width=True)
    my_bar.progress(i + 20)

    # Cleaning the image
    image = clean_image(image)
    my_bar.progress(i + 40)

    # Making the predictions
    predictions, predictions_arr = get_prediction(model, image)
    my_bar.progress(i + 70)

    # Making the results
    result = make_results(predictions, predictions_arr)

    # Removing progress bar and text after prediction is done
    progress.empty()
    my_bar.empty()

    # Displaying the results
    st.write(f"The plant is {result['status']} with a prediction of {result['prediction']}.")

