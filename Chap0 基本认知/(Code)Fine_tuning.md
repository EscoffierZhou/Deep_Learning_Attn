```Python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import ViTModel, ViTImageProcessor, logging
import os
from tqdm import tqdm
import transformers.models.vit.modeling_vit
# 关闭不必要的警告
logging.set_verbosity_error()

# --- 中文显示设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 配置参数 (★ 建议增加轮次来放大变化 ★)
# ==============================================================================
DATA_DIR = "./dataset"
TEST_IMAGE_PATH = "CIEL.jpg"
NUM_EPOCHS = 300  # ★ 增加到30或50轮，给模型更多时间学习
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
VIZ_INTERVAL = 100

#(max(1, NUM_EPOCHS // 500))  # 每隔多少轮次可视化一次，至少为1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"将使用设备: {DEVICE}")


# (ViTForImageClassification 和 ImageFolderDataset 类与之前完全相同，此处省略以保持简洁)
# ... 请确保您保留了这两个类的代码 ...
# ==============================================================================
# 2. 自定义模型：在预训练ViT上添加一个分类头
# ==============================================================================
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", attn_implementation="eager")
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, output_attentions=False):
        outputs = self.vit(pixel_values=pixel_values, output_attentions=output_attentions)
        cls_token_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token_output)

        if output_attentions:
            return logits, outputs.attentions
        else:
            return logits


# ==============================================================================
# 3. 自定义数据集
# ==============================================================================
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(os.listdir(root_dir)))}
        print(f"找到类别: {self.class_to_idx}")
        for cls_name, idx in self.class_to_idx.items():
            cls_path = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_path):
                self.image_paths.append(os.path.join(cls_path, img_name))
                self.labels.append(idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        processed_inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = processed_inputs['pixel_values'].squeeze(0)
        return pixel_values, torch.tensor(label)


# ==============================================================================
# 4. 生成注意力热图的辅助函数 (无变化)
# ==============================================================================
def generate_attention_map(model, image_path, processor, device):
    # ... (此函数与上一版完全相同) ...
    model.eval()
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误: 找不到测试图片 '{image_path}'。")
        return None, None
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        _, attentions = model(inputs['pixel_values'], output_attentions=True)
    last_layer_attention = attentions[-1].squeeze(0).cpu()
    attention_matrix = last_layer_attention.mean(dim=0)
    cls_attention = attention_matrix[0, 1:]
    num_patches_side = int(np.sqrt(cls_attention.shape[0]))
    attention_heatmap = cls_attention.reshape(num_patches_side, num_patches_side).numpy()
    model_input_size = (processor.size['height'], processor.size['width'])
    heatmap_resized = cv2.resize(attention_heatmap, model_input_size)
    heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (
                np.max(heatmap_resized) - np.min(heatmap_resized))
    image_resized_for_display = image.resize(model_input_size)
    return image_resized_for_display, heatmap_normalized


# ==============================================================================
# 5. 主程序：(★ 核心逻辑修改 ★)
# ==============================================================================
if __name__ == "__main__":
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
    dataset = ImageFolderDataset(root_dir=DATA_DIR, processor=processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    num_classes = len(dataset.class_to_idx)

    model = ViTForImageClassification(num_labels=num_classes).to(DEVICE)

    print("正在冻结模型大部分参数...")
    for name, param in model.vit.named_parameters():
        if 'encoder.layer.11' not in name and 'pooler' not in name:
            param.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    attention_visualizations = []

    # --- 训练前 ---
    print("\n--- [阶段] 正在生成 Epoch 0 (训练前) 的注意力图 ---")
    img, heat = generate_attention_map(model, TEST_IMAGE_PATH, processor, DEVICE)
    if img:
        attention_visualizations.append({
            "title": "Epoch 0 (训练前)", "image": img, "heatmap": heat
        })

    # --- 开始训练 ---
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for pixel_values, labels in progress_bar:
            pixel_values, labels = pixel_values.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(pixel_values)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch + 1} 平均损失: {running_loss / len(dataloader):.4f}")

        # --- ★ 按照您的要求，在指定间隔生成注意力图 ★ ---
        if (epoch + 1) % VIZ_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
            print(f"\n--- [阶段] 正在生成 Epoch {epoch + 1} 的注意力图 ---")
            img, heat = generate_attention_map(model, TEST_IMAGE_PATH, processor, DEVICE)
            if img:
                attention_visualizations.append({
                    "title": f"Epoch {epoch + 1}", "image": img, "heatmap": heat
                })

    # --- 训练结束，显示所有注意力图和差异图 ---
    print("\n训练完成！显示注意力演变过程及差异图。")
    num_viz = len(attention_visualizations)

    # +1 是为了给差异图留出位置
    fig, axes = plt.subplots(1, num_viz + 1, figsize=((num_viz + 1) * 5, 5))

    for i, viz_data in enumerate(attention_visualizations):
        axes[i].imshow(viz_data["image"])
        axes[i].imshow(viz_data["heatmap"], cmap='jet', alpha=0.6)
        axes[i].set_title(viz_data["title"])
        axes[i].axis('off')

    # --- ★ 新增：计算并显示差异图 ★ ---
    initial_heatmap = attention_visualizations[0]["heatmap"]
    final_heatmap = attention_visualizations[-1]["heatmap"]
    difference_map = np.abs(final_heatmap - initial_heatmap)  # 取绝对值差异

    # 找到最大差异值用于归一化，使得可视化效果更明显
    vmax = np.max(difference_map) if np.max(difference_map) > 0 else 1.0

    im = axes[num_viz].imshow(difference_map, cmap='magma', vmin=0, vmax=vmax)
    axes[num_viz].set_title("差异图 (Epoch 最终 vs 最初)")
    axes[num_viz].axis('off')
    fig.colorbar(im, ax=axes[num_viz], shrink=0.8)

    plt.suptitle("注意力热图随训练的演变", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
```