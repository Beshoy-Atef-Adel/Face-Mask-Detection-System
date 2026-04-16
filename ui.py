import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")
st.title("Face Mask Detection System 😷")
st.write("Upload an image and the YOLOv8 model will detect faces with/without masks.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    if st.button("Predict"):
        with st.spinner("Running detection..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/predict",
                    files={"file": ("image.jpg", byte_im, "image/jpeg")},
                    timeout=60,
                )
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to the FastAPI server. Make sure it's running on port 8000.")
                st.stop()

            if response.status_code != 200:
                st.error(f"Server returned error {response.status_code}")
                st.code(response.text)
                st.stop()

            data = response.json()
            detections = data.get("detections", [])

            if not detections:
                st.warning("No faces detected in this image.")
            else:
                st.success(f"Detected {len(detections)} object(s)")
                for i, det in enumerate(detections, 1):
                    name = det.get("name", "unknown")
                    conf = det.get("confidence", 0)
                    emoji = "😷" if name == "with_mask" else "❌" if name == "without_mask" else "⚠️"
                    st.write(f"{emoji} **{i}. {name}** — confidence: **{conf:.2%}**")

                with st.expander("Show raw JSON response"):
                    st.json(data)
