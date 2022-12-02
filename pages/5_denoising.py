from turtle import width
import torch
import streamlit as st
from PIL import Image
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms as T
import numpy as np


st.write("""
# Вдохнем жизнь в грязные и помятые документы!
""")
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=12, padding=10),
            nn.BatchNorm2d(64),
            nn.SELU()
        )

        self.layer_2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=4),
            nn.BatchNorm2d(32),
            nn.SELU()
        )

        self.layer_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.SELU()
        )

        self.decoder_1 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=2),
            nn.SELU()
        )

        self.decoder_2 = nn.Sequential(
            nn.ConvTranspose2d(32, 64, kernel_size=4),
            nn.SELU()
        )

        self.decoder_2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.SELU()
        )

        self.decoder_3 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=12, padding=10),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        out = self.decoder_1(x)
        out = self.decoder_2_2(self.decoder_2(out))
        out = self.decoder_3(out)
        return out

trans = T.Compose([
    T.ToTensor()])

resize = T.Compose(
    [T.Resize((250, 500))])


model = AE()
model.load_state_dict(torch.load('res_models/5_denoising.pt'))
model.eval()

img_file5 = st.file_uploader('Choose file', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
if img_file5:
    img5 = Image.open(img_file5)
    img5 = trans(img5.convert("L")) 
    original_size = img5.squeeze(0).shape
    img_pr = resize(img5)
    to_orig_size = T.Compose([T.Resize(original_size)])
    results = to_orig_size(model(torch.unsqueeze(img_pr,0)))
    st.write('## Оригинальное изображение')
    st.image(img5.squeeze(0).detach().cpu().numpy(), use_column_width = True)
    st.write('## Очищенное изображение')
    st.image(results.detach().cpu().numpy()[0][0], use_column_width = True)
    