import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
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
        out = x_sa + x  # 残差连接
        return x_sa, out  # 返回注意力处理结果 & 残差结果


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
def run_cbam_visualization(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 图像加载 & 预处理
    pil_img = Image.open(image_path).convert('RGB').resize((224, 224))
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    # 2. 特征提取（ResNet前半部分）
    model = models.resnet50(pretrained=True).to(device)
    model.eval()
    feature_extractor = nn.Sequential(*list(model.children())[:-4])  # 提取到layer2
    with torch.no_grad():
        features = feature_extractor(img_tensor)
        C = features.shape[1]

        # 3. 注意力模块初始化
        ca = ChannelAttention(C).to(device)
        ca.eval()
        sa = SpatialAttention().to(device)
        sa.eval()
        cbam = CBAM(C).to(device)
        cbam.eval()

        # 通道注意力图
        ca_weight = ca(features)
        ca_feat = features * ca_weight
        ca_att_map = torch.sum(ca_feat, dim=1, keepdim=True)

        # 空间注意力图
        sa_weight = sa(features)
        sa_att_map = sa_weight  # 直接是 [1,1,H,W]

        # 完整 CBAM 注意力图（仅返回 attention 后，无残差）
        cbam_feat, _ = cbam(features)
        cbam_att_map = torch.sum(cbam_feat, dim=1, keepdim=True)

    # 4. 可视化叠加
    overlay_ca = overlay_heatmap(ca_att_map, pil_img)
    overlay_sa = overlay_heatmap(sa_att_map, pil_img)
    overlay_cbam = overlay_heatmap(cbam_att_map, pil_img)

    # 5. 显示
    plt.figure(figsize=(18, 10))


    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(pil_img)
    plt.axis('off')

    # plt.subplot(2, 2, 2)
    # plt.title("Channel Attention")
    # plt.imshow(overlay_ca)
    # plt.axis('off')
    #
    # plt.subplot(2, 2, 3)
    # plt.title("Spatial Attention")
    # plt.imshow(overlay_sa)
    # plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("CBAM (Channel + Spatial)")
    plt.imshow(overlay_cbam)
    plt.axis('off')

    plt.tight_layout()
    plt.show()
# 测试注意力图是否一致（空间注意力）
    att_map1 = sa(features)
    att_map2 = sa(features)
    diff = torch.abs(att_map1 - att_map2).max()
    print(f"Max diff between repeated attention maps: {diff.item()}")


# ======================== 执行 ========================
if __name__ == "__main__":
    image_path = r"F:\desktop\GitHub\Deep_Learning_Attn\input.jpg"
    run_cbam_visualization(image_path)
