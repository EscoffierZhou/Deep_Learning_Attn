```python
"""
一个简化且独立的 PyTorch Vision Transformer (ViT) 模型。

此代码基于 Hugging Face 的 `transformers` 库中的 ViT 实现，
但进行了大幅简化，以便于理解和直接使用。

移除了以下特性：
- Hugging Face 特定的 PreTrainedModel 基类和配置系统。
- `**kwargs` 和复杂的函数签名。
- 头剪枝 (head pruning) 和梯度检查点 (gradient checkpointing)。
- 对 `output_attentions` 和 `output_hidden_states` 的支持。
- 除了标准 eager 模式之外的其他注意力实现。
- 掩码图像建模 (Masked Image Modeling) 相关的功能。
"""
import torch
from torch import nn
from typing import Optional
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """将图像转换为一系列展平的块嵌入。"""

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("图片尺寸必须是块尺寸的整数倍。")

        self.num_patches = (image_size // patch_size) ** 2
        # 使用一个卷积层来实现分块和嵌入
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 的形状: (B, C, H, W) -> (B, D, H/P, W/P)
        x = self.projection(x)
        # 将 H/P 和 W/P 维度展平并移动到序列维度
        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        # N 是块的数量 (num_patches)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):
    """多头自注意力机制模块。"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if head_dim * num_heads != dim:
            raise ValueError("`dim` 必须可以被 `num_heads` 整除。")

        self.scale = head_dim ** -0.5

        # 将输入同时投影到 Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        # 将多头结果合并后的输出投影
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape  # B: Batch Size, N: Sequence Length, C: Channels(dim)

        # qkv(): (B, N, C) -> (B, N, 3 * C)
        # reshape: (B, N, 3, H, D) where H is num_heads, D is head_dim
        # permute: (3, B, H, N, D)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # 分离 Q, K, V
        # 每个的形状都是 (B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        # (B, H, N, D) @ (B, H, D, N) -> (B, H, N, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 用注意力分数加权 V
        # (B, H, N, N) @ (B, H, N, D) -> (B, H, N, D)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FeedForward(nn.Module):
    """前馈网络 (MLP)。"""

    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None,
                 act_layer=nn.GELU, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Transformer 编码器中的一个区块。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, qkv_bias: bool = False, drop: float = 0.0,
                 attn_drop: float = 0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FeedForward(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT 使用 Pre-Norm 结构
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
                 qkv_bias=True, drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # CLS token + position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=nn.LayerNorm
            )
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=.02)
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_module_weights)

    def _init_module_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _interpolate_pos_encoding(self, x, w, h):
        """
        根据输入图像的分辨率插值位置编码
        """
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed

        cls_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]

        w0 = w // self.patch_size
        h0 = h // self.patch_size
        patch_pos_embed = patch_pos_embed.reshape(1, int(N**0.5), int(N**0.5), dim).permute(0, 3, 1, 2)
        patch_pos_embed = F.interpolate(patch_pos_embed, size=(h0, w0), mode='bicubic', align_corners=False)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return torch.cat((cls_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self._interpolate_pos_encoding(x, W, H)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_token_output = x[:, 0]
        logits = self.head(cls_token_output)
        return logits

if __name__ == '__main__':
    # 创建一个 ViT-Base 模型实例
    # 这是 ViT-Base/16 的标准配置
    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_classes=1000,  # ImageNet-1k 的类别数
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
    )

    # 将模型移到 GPU (如果可用)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()  # 设置为评估模式

    # 创建一个假的输入图像张量
    # (batch_size, channels, height, width)
    dummy_image = torch.randn(4, 3, 224, 224).to(device)

    # 进行一次前向传播
    with torch.no_grad():  # 在评估时不需要计算梯度
        logits = model(dummy_image)

    # 输出 logits 的形状
    # 应该是 (batch_size, num_classes)
    print(f"模型已成功创建并运行。")
    print(f"输入图像形状: {dummy_image.shape}")
    print(f"输出 Logits 形状: {logits.shape}")  # 预期输出: torch.Size([4, 1000])

    # 计算模型参数数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数量: {num_params / 1e6:.2f} M")  # ViT-Base 大约为 86M
```