import streamlit as st
import numpy as np
from model import inference

def main():
    st.title("Speed Breaker Detection 🛣️")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        with st.spinner("This may take upto 20-30 seconds!"):

            output_image = inference(uploaded_file)

            st.image(output_image, caption="Processed Image", use_container_width=True)


if __name__ == "__main__":
    main()