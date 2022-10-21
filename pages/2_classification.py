import streamlit as st

import json
from PIL import Image
import requests

import pandas as pd
import numpy as np

import torch
import torchvision.transforms as T

import matplotlib.pyplot as plt

xception = torch.load('res_models/classification/xception.pt')
resnet50 = torch.load('res_models/classification/resnet50.pt')
resnet18 = torch.load('res_models/classification/resnet18.pt')

classes = json.load(open("2_ResNet18_classification/classes.txt"))

def predict_multi(model, top, img):
    model.eval()
    predictions = model(img)
    predictions[0] = torch.nn.functional.softmax(predictions[0], dim=0)

    indexes = torch.topk(predictions, top)[1].squeeze(0).tolist()
    cls = np.array(list(classes.keys()))[indexes]
    prob = torch.topk(predictions, top)[0].squeeze(0).detach().numpy()
    return pd.DataFrame(np.transpose([cls, prob]), columns=['class', 'probability'])


st.write('''
# Classification

''')

way = st.radio('Выбери способ загрузки изображения', ['По URL-ссылке', 'С компьютера'])

if way == 'По URL-ссылке':
    url = st.text_input('Вставь ссылку сюда:')
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        st.image(image, caption='Ваше изображение', use_column_width=True)
        image = T.ToTensor()(image).unsqueeze(0)
    except:
        pass

if way == 'С компьютера':
    uploaded_file = st.file_uploader('Загрузи изображение')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(uploaded_file, caption='Ваше изображение', use_column_width=True)
        image = image.convert('RGB')
        image = T.ToTensor()(image).unsqueeze(0)


model = st.radio('Выбери модель', ['Xception', 'ResNet50', 'ResNet18'])
top = st.slider('', min_value=1, max_value=10, step=1, label_visibility="hidden")

if model == 'Xception':
    st.dataframe(data=predict_multi(xception, top, image), use_container_width=True)

if model == 'ResNet50':
    st.dataframe(data=predict_multi(resnet50, top, image), use_container_width=True)

if model == 'ResNet18':
    st.dataframe(data=predict_multi(resnet18, top, image), use_container_width=True)
