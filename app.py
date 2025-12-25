import os
os.environ["ULTRALYTICS_NO_CV2"] = "1"

import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel  # required for safe load

# Page config
st.set_page_config(page_title="Number Plate Detection", layout="centered")
st.title("Number Plate Detection App")
st.write("Upload an image to detect")

# Load model safely
@st.cache_resource
def load_model():
    # Safe context to allow DetectionModel class
    with torch.serialization.safe_globals([DetectionModel]):
        model = YOLO("best.pt")  # path to your trained YOLO model
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
