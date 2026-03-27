import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier

st.set_page_config(page_title="CT Stroke Detection", page_icon="🧠")

st.title("🧠 Brain Stroke Detection (CT Scan Only)")
st.markdown("Upload a **Brain CT scan** to detect stroke")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("ct_stroke_model.json")
    return model

model = load_model()

# ---------------- FILE ----------------
file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

# ---------------- FUNCTIONS ----------------

def extract_features(img):
    img = cv2.resize(img, (64,64))
    pixel = img.flatten()
    mean = np.mean(img)
    std = np.std(img)
    maxv = np.max(img)
    minv = np.min(img)
    return np.hstack([pixel, mean, std, maxv, minv]).reshape(1,-1)

# 👉 CT detector (important)
def is_ct_scan(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    brightness = np.mean(gray)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges) / (gray.shape[0]*gray.shape[1])

    # CT usually:
    # - bright skull
    # - strong edges
    if brightness > 80 and edge_density > 5:
        return True
    return False

# ---------------- MAIN ----------------
if file is not None:
    try:
        img = Image.open(file).convert("RGB")
        st.image(img, use_container_width=True)

        img = np.array(img)

        if st.button("Predict"):

            # STEP 1: Check CT
            if not is_ct_scan(img):
                st.error("❌ Invalid Image: Not a Brain CT Scan")
            
            else:
                # STEP 2: Prediction
                features = extract_features(img)

                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                if confidence < 0.65:
                    st.warning("⚠️ Low confidence prediction")
                
                if pred == 0:
                    st.error("⚠️ Hemorrhagic Stroke Detected")
                else:
                    st.success("✅ Ischemic Stroke Detected")

                st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
