import streamlit as st
from PIL import Image

st.title("Camera Input with Streamlit")

st.write("Capture an image using your camera!")

# Camera input widget
camera_image = st.camera_input("Take a picture")

# Display the captured image
if camera_image:
    st.write("Here is your image:")
    img = Image.open(camera_image)
    st.image(img, caption="Captured Image", use_column_width=True)