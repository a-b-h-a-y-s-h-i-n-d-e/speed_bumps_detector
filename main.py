import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# Custom video processor to resize the frame
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # Resize the image to 416x416
        resized_img = cv2.resize(img, (416, 416))
        return resized_img

st.title("Webcam Stream with Streamlit")

# Stream the camera feed with frame resizing
webrtc_streamer(
    key="camera",
    mode=WebRtcMode.SENDRECV,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)
