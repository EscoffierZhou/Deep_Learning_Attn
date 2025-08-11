import torch
import numpy as np
import cv2
from PIL import Image
import requests
from transformers import ViTModel, ViTImageProcessor
import matplotlib.pyplot as plt

# 设置随机种子以保证结果的一致性
torch.manual_seed(51)
np.random.seed(51)

# 加载预训练的ViT模型和图像处理器
model_name = "google/vit-base-patch16-224-in21k"
model = ViTModel.from_pretrained(model_name, output_attentions=True)
processor = ViTImageProcessor.from_pretrained(model_name)
model.eval()

# 加载一张示例图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# 预处理图片
inputs = processor(images=image, return_tensors="pt")
pixel_values = inputs.pixel_values

# 将图片送入模型
with torch.no_grad():
    outputs = model(pixel_values)

# 获取最后一层的注意力图
# outputs.attentions 是一个元组，包含了每一层的注意力权重
# 形状为 (batch_size, num_heads, sequence_length, sequence_length)
last_layer_attention = outputs.attentions[-1]

# (1, 12, 197, 197) -> (12, 197, 197)，197 = 1 (CLS) + 196 (Patches)
last_layer_attention = last_layer_attention.squeeze(0)