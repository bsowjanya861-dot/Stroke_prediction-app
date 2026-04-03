import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier
def set_bg():
    url = "https://www.google.com/imgres?q=real%20world%20brain%20stroke%20images&imgurl=https%3A%2F%2Fmedia.istockphoto.com%2Fid%2F2182844689%2Fvector%2Fabstract-blue-brains-with-red-spot-brains-deseases-medical-treatment-and-pharmacy-concept.jpg%3Fs%3D612x612%26w%3D0%26k%3D20%26c%3D0R5l4vPSwWl7rmZurI5-x_lwzvVWdcCVicC0r1vkSQg%3D&imgrefurl=https%3A%2F%2Fwww.istockphoto.com%2Fillustrations%2Fbrain-stroke&docid=bmr7CTXkhKElJM&tbnid=8fPiaWcgr0Ha-M&vet=12ahUKEwi566i1tNGTAxXJSWwGHbbGNYY4ChCc8A56BAg8EAE..i&w=612&h=365&hcb=2&ved=2ahUKEwi566i1tNGTAxXJSWwGHbbGNYY4ChCc8A56BAg8EAE"  # replace with your image link

    css = f"""
    <style>
    .stApp {{
        background-image: url("{url}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

set_bg()

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)

# -------------------- TITLE --------------------
st.title("🧠 Brain Stroke Prediction")
st.markdown("Upload a **Brain MRI image** to predict stroke type")

# -------------------- SIDEBAR --------------------
st.sidebar.title("About")
st.sidebar.info(
    "This app uses an XGBoost model to detect stroke type.\n\n"
    "Classes:\n"
    "- Hemorrhagic Stroke\n"
    "- Ischemic Stroke\n\n"
    "⚠️ This is a demo project (not for medical use)"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("hybrid_stroke_model.json")
    return model

model = load_model()

# -------------------- VALIDATION FUNCTION --------------------
def is_valid_mri(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    mean = np.mean(gray)
    std = np.std(gray)

    # Edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges) / edges.size

    # Strict MRI-like conditions
    if std < 20:
        return False

    if mean < 30 or mean > 220:
        return False

    if edge_density < 0.02:
        return False

    return True

# -------------------- FILE UPLOAD --------------------
file = st.file_uploader("📤 Upload Brain MRI Image", type=["jpg", "png", "jpeg"])

# -------------------- MAIN LOGIC --------------------
if file is not None:
    try:
        # Display image
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Convert to array
        img = np.array(img)

        # Resize (must match training)
        img = cv2.resize(img, (64, 64))

        # -------------------- VALIDATION FIRST --------------------
        if not is_valid_mri(img):
            st.error("❌ Invalid Image: Not a Brain MRI")
        else:
            # Feature extraction
            pixel_features = img.flatten()

            mean = np.mean(img)
            std = np.std(img)
            maxv = np.max(img)
            minv = np.min(img)

            features = np.hstack([pixel_features, mean, std, maxv, minv])
            features = features.reshape(1, -1)

            # -------------------- PREDICT --------------------
            if st.button("🔍 Predict"):
                proba = model.predict_proba(features)
                confidence = float(np.max(proba))
                pred = int(np.argmax(proba))

                # -------------------- DOUBLE VALIDATION --------------------
                if confidence < 0.92:
                    st.error("❌ Invalid or unclear MRI image")
                    st.write(f"Confidence: {confidence:.2f}")

                else:
                    if pred == 0:
                        st.error("⚠️ Hemorrhagic Stroke Detected")
                    else:
                        st.success("✅ Ischemic Stroke Detected")

                    st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
