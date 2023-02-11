from inference import Transformer
from utils import read_image, resize_image
import streamlit as st
import cv2
import numpy as np
from PIL import Image

PAGE_CONFIG = {"page_title":"AnimeGAN", "layout":"wide"}
st.set_page_config(**PAGE_CONFIG)

st.title('This project transform raw image to anime image')
transformer = Transformer('hayao')


file_up = st.file_uploader("Upload an image", type=['jpg','png','jpeg'])
col1, col2 = st.columns(2)

if file_up is not None:
    url = 'temp.jpg'
    im = Image.open(file_up)
    shape = im.size
    image = np.array(im)
    cv2.imwrite(url, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    image = resize_image(read_image(url), 1184, 800)
    anime_img = ((transformer.transform(image) + 1) / 2)[0]

    with col1:
        st.write('Original Image')
        image = cv2.resize(image, shape)
        st.image(image,use_column_width="auto")
    with col2:
        st.write('Transform Image')
        anime_img = cv2.resize(anime_img, shape)
        st.image(anime_img, use_column_width="auto")
        
