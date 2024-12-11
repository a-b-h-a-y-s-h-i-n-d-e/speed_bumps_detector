import streamlit as st
import numpy as np
from model import inference


hide_decoration_bar_style = '''
    <style>
    
    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }
    #MainMenu {
        visibility: hidden;
        height: 0%;
    }
    header {
        visibility: hidden;
        height: 0%;
    }
    footer {
        visibility: hidden;
        height: 0%;
    }

    .reportview-container .main {visibility: hidden;}

    .stDeploymentInfo {
        visibility: hidden;
        display: none !important;
    }
    .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_,
    .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none !important;
        visibility: hidden !important;
    }
    </style>
    '''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

title = '<p style="text-align:center; font-size:50px; color:#f4d03f;"> Speed Breaker Detection 🛣️</p>'
st.markdown(title, unsafe_allow_html=True)

info = """<b> Why to detect Speed Breakers ? </b> &nbsp; Speed breakers and bumpy roads are a major threat to drivers that questions their safety.
 The mishap happens because of no sign boards indicating the speed breaker, poor visibility at night
 and road works that are often carried out with no proper signs of road deviations and also the negligence of the driver !! 
"""
st.markdown(info, unsafe_allow_html=True)

st.image(image="./assets/style1.jpg")
st.markdown("<br>", unsafe_allow_html=True)

with st.sidebar:

    st.image(image="./assets/car.gif")
    st.markdown("</br>", unsafe_allow_html=True)
    
    st.markdown('You can use below sample images to try out')

    col1, col2 = st.columns(2)
    with col1:
        st.image(image="./assets/image1.jpg")
    with col2:
        st.image(image="./assets/image2.jpg")

    with col1:
        st.image(image="./assets/image3.jpg")
    with col2:
        st.image(image="./assets/image4.jpg")
    
    with col1:
        st.image(image="./assets/image5.jpg")
    with col2:
        st.image(image="./assets/image6.jpg")

    with col1:
        st.image(image="./assets/image7.jpg")
    with col2:
        st.image(image="./assets/image8.jpg")

    with col1:
        st.image(image="./assets/image9.jpg")
    with col2:
        st.image(image="./assets/image10.jpg")



def main():

    # File uploader widget
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        with st.spinner("This may take upto 20-30 seconds!"):

            output_image = inference(uploaded_file)
            st.image(output_image, caption="Processed Image",width=350, use_container_width=True)


if __name__ == "__main__":
    main()