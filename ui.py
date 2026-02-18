import streamlit as st
import requests
from PIL import Image
import io

st.title("Face Mask Detection System 😷")

# اختيار الصورة من جهازك
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # تجهيز الصورة عشان تتبعت للـ API
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    byte_im = buf.getvalue()

    if st.button('Predict'):
        # هنا بنكلم الـ API اللي شغالة في الـ Docker
        response = requests.post("http://127.0.0.1:8000/predict", files={"file": byte_im})
        st.write(response.json())