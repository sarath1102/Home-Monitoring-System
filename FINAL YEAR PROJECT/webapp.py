import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os


def load_image(image_file):
	img = Image.open(image_file)
	return img


def main():
    """Train Images"""

    st.title("House Monitoring System")

    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:teal ;padding:10px">
    <h2 style="color:white;text-align:center;">Face Recognition WebApp</h2>
    </div>
    </body>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])

    filename=st.text_input("Enter the name of person")

    if st.button("Upload"):
        if image_file is not None:
            our_image = Image.open(image_file)
            st.text("Original Image")
            st.image(our_image)
            st.image(load_image(image_file),width=250)
            with open(os.path.join("C:\\Users\\asus\\OneDrive\\Desktop\\ie\\Training_images",filename+str(".jpg")),"wb") as f:
                f.write((image_file).getbuffer())
if __name__ == '__main__':
    main()