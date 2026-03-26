import streamlit as st
import numpy as np
from PIL import Image
from xgboost import XGBClassifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

st.set_page_config(page_title="CT Stroke Prediction", layout="centered")
st.title("🧠 CT Stroke + Stroke Type Prediction")

# ------------------------
# Load CT Detector (CNN)
# ------------------------
# You need to train this model separately:
#   - Input: 224x224 RGB images
#   - Output: 1 = CT scan, 0 = Not CT scan
# Example filename: 'ct_detector.h5'
ct_detector = load_model("ct_detector.h5")

# ------------------------
# Load XGBoost Stroke Model
# ------------------------
xgb_model = XGBClassifier()
xgb_model.load_model("hybrid_stroke_model.json")  # your existing model

# ------------------------
# Preprocessing functions
# ------------------------
def preprocess_for_cnn(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img)
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr

def preprocess_for_xgb(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    features = arr.flatten()
    return features.reshape(1, -1)

# ------------------------
# Upload and predict
# ------------------------
file = st.file_uploader("Upload CT scan image", type=["jpg","png","jpeg"])

if file is not None:
    try:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Step 1: Check if image is a CT scan
        cnn_input = preprocess_for_cnn(img)
        ct_prob = ct_detector.predict(cnn_input)[0][0]
        if ct_prob < 0.5:
            st.warning("⚠️ This does not appear to be a CT scan. Please upload a valid CT scan image.")
        else:
            # Step 2: Predict stroke type with XGBoost
            features = preprocess_for_xgb(img)
            if st.button("Predict Stroke Type"):
                pred = xgb_model.predict(features)
                if pred[0] == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischaemic Stroke Detected")
    except Exception as e:
        st.warning(f"⚠️ Invalid image. Please upload a valid CT scan. Error: {e}")
