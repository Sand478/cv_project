import streamlit as st

import timm

from PIL import Image
import requests

import torchvision.transforms as T
import torch.nn.functional as F


xception = timm.create_model('xception', pretrained=True)
import urllib
url, filename = ('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', 'imagenet_classes.txt')
urllib.request.urlretrieve(url, filename)
with open('imagenet_classes.txt', 'r') as f:
    categories = [s.strip() for s in f.readlines()]

st.write('''
# Xception
''')

way = st.radio('Выбери способ загрузки изображения', ['По URL-ссылке', 'С компьютера'])

def predict_xception(image):
    xception.eval()
    predictions = xception(image)
    predictions = F.softmax(predictions[0], dim=0)
    return categories[predictions.argmax()]

if way == 'По URL-ссылке':
    url = st.text_input('Вставь ссылку сюда:')
    try:
        image = Image.open(requests.get(url, stream=True).raw)
        st.image(image, caption='Ваше изображение', use_column_width=True)
        image = T.ToTensor()(image).unsqueeze(0)
    except:
        pass

# AttributeError
if way == 'С компьютера':
    uploaded_file = st.file_uploader('Загрузи изображение')
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(uploaded_file, caption='Ваше изображение', use_column_width=True)
        image = image.convert('RGB')
        image = T.ToTensor()(image).unsqueeze(0)

try:
    st.metric(label="Результат:", value=predict_xception(image))
    #st.write(f'{predict_xception(image)}')
except:
    pass