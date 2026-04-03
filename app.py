import streamlit as st
import numpy as np
import cv2
from PIL import Image
from xgboost import XGBClassifier
# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Brain Stroke Prediction",
    page_icon="🧠",
    layout="centered"
)
# ---------- Helper Functions ----------

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return np.array([int(hex_color[i:i+2], 16) for i in (0, 2, 4)]) / 255.0


def apply_rgb_curves(image, r_curve, g_curve, b_curve):
    x = np.linspace(0, 1, len(r_curve))

    r = np.interp(image[:,:,0], x, r_curve)
    g = np.interp(image[:,:,1], x, g_curve)
    b = np.interp(image[:,:,2], x, b_curve)

    return np.stack([r, g, b], axis=-1)


def apply_adjustments(image, brightness, contrast, saturation):
    img = image.astype(np.float32) / 255.0

    # Brightness
    img = np.clip(img + brightness, 0, 1)

    # Contrast
    img = np.clip(img + contrast * (img - 0.5), 0, 1)

    # Saturation
    gray = np.dot(img, [0.299, 0.587, 0.114])
    img = gray[:, :, None] + (img - gray[:, :, None]) * saturation

    return np.clip(img, 0, 1)


def apply_temperature(image, temp):
    R, G, B = image[:,:,0], image[:,:,1], image[:,:,2]

    if temp > 0:
        R *= (1 + temp)
        B /= (1 + temp)
    else:
        B *= (1 - temp)
        R /= (1 - temp)

    return np.stack([R, G, B], axis=-1)


def apply_vignette(image, amount):
    h, w = image.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    cx, cy = w / 2, h / 2

    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = np.exp(-(r**2) / (0.5 * w)**2)
    mask = mask[:, :, None]

    return image * (1 - amount) + image * mask * amount


# ---------- Presets ----------

presets = {
    "None": {"brightness":0.0,"contrast":0.0,"saturation":1.0,"temperature":0.0},
    "Vintage": {"brightness":0.2,"contrast":-0.1,"saturation":0.9,"temperature":0.25},
    "Moody": {"brightness":0.1,"contrast":0.2,"saturation":1.1,"temperature":-0.2}
}

# ---------- UI ----------

st.title("Simple Color Grading App")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Preset selection
    preset = st.sidebar.selectbox("Preset", list(presets.keys()))
    preset_vals = presets[preset]

    # Sliders
    brightness = st.sidebar.slider("Brightness", -1.0, 1.0, preset_vals["brightness"])
    contrast = st.sidebar.slider("Contrast", -1.0, 1.0, preset_vals["contrast"])
    saturation = st.sidebar.slider("Saturation", 0.0, 2.0, preset_vals["saturation"])
    temperature = st.sidebar.slider("Temperature", -1.0, 1.0, preset_vals["temperature"])
    vignette = st.sidebar.slider("Vignette", 0.0, 1.0, 0.0)

    # RGB Curves (simple fixed 3-point curves)
    red_curve = [0.0, 0.5, 1.0]
    green_curve = [0.0, 0.5, 1.0]
    blue_curve = [0.0, 0.5, 1.0]

    # Processing pipeline
    img = apply_adjustments(image, brightness, contrast, saturation)
    img = apply_temperature(img, temperature)
    img = apply_vignette(img, vignette)
    img = apply_rgb_curves(img, red_curve, green_curve, blue_curve)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.header("Original")
        st.image(image)

    with col2:
        st.header("Processed")
        st.image((img * 255).astype(np.uint8))

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
