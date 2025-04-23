import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import uuid

st.set_page_config(page_title="License Plate Detection - YOLOv8", layout="centered")

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("/content/drive/Othercomputers/Vignesh MacBook Air/License_Plate_Detection/runs/detect/train3/weights/best.pt")

model = load_model()

# App Header
st.title("🔍 License Plate Detection using YOLOv8")
st.markdown("""
This web application allows you to detect **license plates** in uploaded or webcam-captured images using the YOLOv8 object detection model.

- Trained on a Roboflow license plate dataset
- Powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Runs in **Google Colab + Streamlit + ngrok**

👉 Upload image(s) or use your **webcam** to detect license plates.
""")

# Sidebar - Detection Settings
st.sidebar.header("⚙️ Detection Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# ----------- Section 1: Upload Image(s) -----------
st.subheader("📁 Upload Image(s)")

uploaded_files = st.file_uploader("Choose JPG/PNG image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.subheader(f"📷 Uploaded: `{uploaded_file.name}`")
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

        with st.spinner("🔍 Detecting license plates..."):
            results = model.predict(image, conf=conf_threshold)
            output_filename = f"output_{uuid.uuid4().hex}.jpg"
            results[0].save(filename=output_filename)

        st.image(output_filename, caption="Detected Output", use_container_width=True)

        # Show detection info
        detections = results[0].boxes
        if detections is not None and len(detections) > 0:
            st.success(f"✅ {len(detections)} license plate(s) detected")
            for i, box in enumerate(detections.data):
                cls = int(box[5].item())
                conf = box[4].item()
                class_name = model.names.get(cls, "Unknown")
                st.write(f"🔹 Detection {i+1}: Class = `{class_name}`, Confidence = `{conf:.2f}`")
        else:
            st.warning("⚠️ No license plates detected.")

        # Download option
        with open(output_filename, "rb") as file:
            st.download_button("📥 Download Output Image", data=file, file_name=output_filename, mime="image/jpeg")

# ----------- Section 2: Webcam Capture -----------
st.markdown("---")
st.subheader("📸 Or Capture from Webcam")

camera_image = st.camera_input("Take a photo using your webcam")

if camera_image is not None:
    st.subheader("📷 Captured Image")
    image = Image.open(camera_image)
    st.image(image, caption="Webcam Capture", use_container_width=True)

    with st.spinner("🔍 Detecting license plates..."):
        results = model.predict(image, conf=conf_threshold)
        output_filename = f"webcam_output_{uuid.uuid4().hex}.jpg"
        results[0].save(filename=output_filename)

    st.image(output_filename, caption="Detected Output", use_container_width=True)

    detections = results[0].boxes
    if detections is not None and len(detections) > 0:
        st.success(f"✅ {len(detections)} license plate(s) detected")
        for i, box in enumerate(detections.data):
            cls = int(box[5].item())
            conf = box[4].item()
            class_name = model.names.get(cls, "Unknown")
            st.write(f"🔹 Detection {i+1}: Class = `{class_name}`, Confidence = `{conf:.2f}`")
    else:
        st.warning("⚠️ No license plates detected.")

    with open(output_filename, "rb") as file:
        st.download_button("📥 Download Output Image", data=file, file_name=output_filename, mime="image/jpeg")
