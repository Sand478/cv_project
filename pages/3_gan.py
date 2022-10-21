import streamlit as st
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib
import matplotlib.pyplot as plt

from PIL import Image
import requests


def denorm(img_tensors):
    stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    return img_tensors * stats[1][0] + stats[0][0]

latent_size= 64
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(latent_size, 512, kernel_size=4, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
        )

        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )

        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.upsample4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )

        self.upsample5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.upsample4(x)
        x = self.upsample5(x)

        return x

generator = torch.load('res_models/generator.pt')

st.write('''
# Conditional GAN

[Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)

''')

# top = st.slider('', min_value=1, max_value=10, step=1, label_visibility="hidden")

url = 'https://raw.githubusercontent.com/Mckinsey666/Anime-Face-Dataset/master/test.jpg'
image = Image.open(requests.get(url, stream=True).raw)
st.image(image, caption='', use_column_width=False)

st.write('''
Let's try to generate!
''')

width = st.slider("plot width", 100, 1000, 10)

gen_button = st.button(label="Generate!")

if gen_button:
    lt = torch.rand((64)).unsqueeze(1).unsqueeze(1).unsqueeze(0)
    img = denorm(generator(lt))
    st.image(img.squeeze(0).permute(1, 2, 0).detach().numpy(), width=width)