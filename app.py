import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.title("🧠 CT Stroke Prediction")

# Load trained XGBoost model
model = XGBClassifier()
model.load_model("hybrid_stroke_model.json")

def preprocess_ct(img):
    # Grayscale, resize, normalize, flatten + stats
    img = img.convert("L")
    img = img.resize((64, 64))
    arr = np.array(img) / 255.0
    features = np.hstack([arr.flatten(), np.mean(arr), np.std(arr), np.max(arr), np.min(arr)])
    return features.reshape(1, -1)

file = st.file_uploader("Upload CT scan", type=["jpg","png","jpeg"])

if file is not None:
    try:
        img = Image.open(file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # TODO: Replace this with a real CNN CT vs non-CT check
        # For now, we assume user uploads CT
        is_ct_scan = True  # Placeholder — currently always True

        if not is_ct_scan:
            st.warning("⚠️ This does not look like a CT scan.")
        else:
            features = preprocess_ct(img)
            if st.button("Predict"):
                pred = model.predict(features)
                if pred[0] == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischaemic Stroke Detected")

    except:
        st.warning("⚠️ Invalid image file.")
