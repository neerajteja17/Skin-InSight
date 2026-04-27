import streamlit as st
from PIL import Image
import cv2
import numpy as np

from utils.preprocess import preprocess_image
from utils.predict import load_model, predict
from utils.disease_info import disease_info
from utils.pdf_report import generate_pdf

st.set_page_config(layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.text_input("Age")
gender = st.sidebar.text_input("Gender")
blood = st.sidebar.text_input("Blood Group")

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center; color:#7a4b00;'>Skin In-Sight</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:#7a4b00;'>Disease Diagnosis with Explanation Generation</h4>", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns([1, 2])
    image = Image.open(uploaded_file)

    # ---------------- COLUMN 1 ----------------
    with col1:
        st.image(image, caption="Input Image")

        img = np.array(image)

        # Resize
        img = cv2.resize(img, (224, 224))

        # Convert to uint8
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        # Handle RGBA
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        # Heatmap
        heatmap = cv2.applyColorMap(img, cv2.COLORMAP_JET)

        st.image(heatmap, caption="Heatmap")

    # ---------------- COLUMN 2 ----------------
    with col2:
        if st.button("🔍 Analyze"):

            # PREPROCESS
            processed = preprocess_image(image)

            # PREDICTION
            label, confidence = predict(model, processed)

            # INFO
            info = disease_info.get(label, {
                "overview": "Not available",
                "symptoms": "Not available",
                "causes": "Not available",
                "precautions": "Consult doctor"
            })

            # ---------------- DISPLAY ----------------
            st.markdown("## 🧾 Diagnosis Report")

            st.markdown(f"### 🩺 Disease Name : {label}")
            st.markdown(f"### 📊 Confidence Score : {confidence:.2f}")

            st.markdown("---")

            st.markdown("### 📖 Overview")
            st.write(info["overview"])

            st.markdown("### ⚠️ Symptoms")
            st.write(info["symptoms"])

            st.markdown("### 🧬 Causes")
            st.write(info["causes"])

            st.markdown("### 💊 Precautions & Medication")
            st.write(info["precautions"])

            st.markdown("---")

            # ---------------- PDF GENERATION (UPDATED) ----------------
            pdf_path = generate_pdf(
                name, age, gender, blood,
                label, confidence, info,
                image, heatmap   # 🔥 THIS IS THE IMPORTANT CHANGE
            )

            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 Download Full Report",
                    f,
                    file_name="Skin_Disease_Report.pdf"
                )