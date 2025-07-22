import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 固定随机性
torch.manual_seed(51)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ======================== CBAM 模块 ========================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        reduced = max(1, in_planes // ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(in_planes, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))
        max_out = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        max, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg, max], dim=1)
        return self.sigmoid(self.conv(x_cat))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        ca = self.ca(x)
        x_ca = x * ca
        sa = self.sa(x_ca)
        x_sa = x_ca * sa
        out = x_sa + x
        return x_sa  # 仅返回注意力加权后的特征图


# ======================== 热力图工具 ========================
def overlay_heatmap(att_map, original_image, alpha=0.5, colormap=cv2.COLORMAP_JET, normalize=True):
    att_map = att_map.squeeze().detach().cpu().numpy()
    att_map = cv2.resize(att_map, original_image.size)
    if normalize:
        att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)

    heatmap = np.uint8(255 * att_map)
    heatmap_color = cv2.applyColorMap(heatmap, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.array(original_image) * (1 - alpha) + heatmap_color * alpha
    return np.uint8(overlay)


# ======================== 主流程 ========================
def run_cbam_visualization(image_path, iteration_steps=[1, 5, 50]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 图像加载
    pil_img = Image.open(image_path).convert('RGB').resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 模型准备
    model = models.resnet(pretrained=True).to(device)
    model.eval()
    feature_extractor = nn.Sequential(*list(model.children())[:-4])
    cbam = None
    heatmaps = []

    with torch.no_grad():
        features = feature_extractor(img_tensor)
        C = features.shape[1]
        cbam = CBAM(C).to(device)
        cbam.eval()

        for i in range(1, max(iteration_steps) + 1):
            refined_feat = cbam(features)
            if i in iteration_steps:
                att_map = torch.sum(refined_feat, dim=1, keepdim=True)
                overlay = overlay_heatmap(att_map, pil_img)
                heatmaps.append((f"CBAM Iteration {i}", overlay))

    # 显示结果
    num_images = len(heatmaps) + 1
    plt.figure(figsize=(6 * num_images, 6))
    plt.subplot(1, num_images, 1)
    plt.title("Original Image")
    plt.imshow(pil_img)
    plt.axis('off')

    for idx, (title, overlay_img) in enumerate(heatmaps, start=2):
        plt.subplot(1, num_images, idx)
        plt.title(title)
        plt.imshow(overlay_img)
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# ======================== 执行 ========================
if __name__ == "__main__":
    image_path = r"F:\desktop\GitHub\Deep_Learning_Attn\input.jpg"
    run_cbam_visualization(image_path, iteration_steps=[1, 5, 50])
