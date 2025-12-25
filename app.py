
import os
os.environ["ULTRALYTICS_NO_CV2"] = "1"

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Mask Detection", layout="centered")

st.title("Number Plate Detection App")
st.write("Upload an image to detect ")

# Load model 
@st.cache_resource
def load_model():
    model = YOLO("best.pt")
    return model

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Run Detection"):
        with st.spinner("Detecting..."):
            results = model(np.array(image))

            # Result image
            result_img = results[0].plot()
            st.image(result_img, caption="Detection Result")

            # Show detections
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                st.success(f"Detections Found: {len(boxes)}")
            else:
                st.warning("No object detected")
