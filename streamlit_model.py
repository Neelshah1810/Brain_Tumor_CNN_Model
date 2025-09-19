import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps

# -------------------------------
# 1. Must be FIRST Streamlit call
# -------------------------------
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

# -------------------------------
# 2. Load your trained model
# -------------------------------
@st.cache_resource  # caches model so it doesn’t reload every time
def load_brain_tumor_model():
    model = load_model("brain_tumor_detection_model_v1_96%_Kaggle.h5")
    return model


model = load_brain_tumor_model()

# -------------------------------
# 3. App title
# -------------------------------
st.title("🧠 Brain Tumor MRI Classification")
st.write("Upload an MRI image to classify it as **Glioma**, **Meningioma**, **Pituitary**, or **No Tumor**.")

# -------------------------------
# 4. Image Preprocessing Function
# -------------------------------
IMG_SIZE = (150, 150)  # must match your training size

def preprocess_image(image):
    image = ImageOps.fit(image, IMG_SIZE, Image.Resampling.LANCZOS)
    img_array = np.asarray(image).astype("float32") / 255.0
    if img_array.shape[-1] == 4:  # handle PNG with alpha
        img_array = img_array[..., :3]
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# -------------------------------
# 5. File Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    if st.button("🔍 Predict Tumor Type"):
        with st.spinner("Analyzing MRI..."):
            x = preprocess_image(image)
            preds = model.predict(x)[0]  # shape: (4,)
            predicted_class = np.argmax(preds)

            class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            result = class_names[predicted_class]

            # Final result
            st.success(f"### 🧾 Prediction: **{result}**")

            # Show class probabilities
            st.subheader("📊 Prediction Probabilities:")
            for i, prob in enumerate(preds):
                st.write(f"- {class_names[i]}: **{prob*100:.2f}%**")

            # Optional: Progress bars for visualization
            st.write("---")
            for i, prob in enumerate(preds):
                st.progress(float(prob))
                st.write(f"{class_names[i]}: {prob*100:.2f}%")
