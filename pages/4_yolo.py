import torch
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.write("""
# Cервис детекции козочек, сторожевых собак и куриц
### по фотографиям фермы 
""")

model = torch.hub.load('ultralytics/yolov5', 'custom', path='res_models/4yolo_best.pt')
img_file = st.file_uploader('Choose file', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
if img_file:
    img = Image.open(img_file)
    results = model(img)
    fig, ax = plt.subplots()
    plt.axis('off')
    ax.imshow(results.render()[0])
    st.pyplot(fig)
