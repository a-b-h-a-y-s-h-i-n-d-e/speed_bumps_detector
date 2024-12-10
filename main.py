import streamlit as st
import numpy as np
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoTransformerBase

# Create a class to handle video frame transformations
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to an OpenCV image (BGR format)
        img = frame.to_ndarray(format="bgr24")

        # Draw a rectangle on the image (you can adjust the coordinates and size)
        start_point = (100, 100)  # top-left corner
        end_point = (300, 300)    # bottom-right corner
        color = (255, 0, 0)       # Rectangle color (Blue)
        thickness = 5             # Thickness of the rectangle
        img = cv2.rectangle(img, start_point, end_point, color, thickness)

        return img

# Streamlit UI
st.title("Live Camera with Rectangle Overlay")

# Start the WebRTC stream and apply the transformation
webrtc_streamer(key="example", mode=WebRtcMode.SENDRECV, video_transformer_factory=VideoTransformer)
