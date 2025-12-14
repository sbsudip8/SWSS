import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("waste_classifier_final.h5")

class_names = ['glass','metal','organic','paper','plastic','textile']

st.title("Smart Waste Segregation System")

uploaded = st.file_uploader("Upload a waste image", type=['jpg','jpeg','png'])
if uploaded:
    img = load_img(uploaded, target_size=(224,224))
    st.image(img, caption="Uploaded Image")

    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)

    st.write("### Predicted Category:", class_names[idx])
    st.write("Confidence: {:.2f}%".format(preds[idx] * 100))

# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# .\.venv\Scripts\Activate.ps1
# streamlit run app.py