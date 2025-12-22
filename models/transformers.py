"""
从 mosei 项目移植的 Transformer 组件
用于处理三模态 (Text, Audio, Visual) 预提取特征

Text: 使用 Transformer 架构
Audio/Visual: 使用简化的 MLP 或 LinearEmbed

特征维度 (MOSEI/MOSI):
- Text: [B, T, 300]  GloVe embeddings
- Audio: [B, T, 74]  COVAREP features  
- Visual: [B, T, 35] Facet features
"""

import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =============================================================================
# 权重初始化
# =============================================================================

def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ViT weight initialization, original timm impl"""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_xavier(module: nn.Module, name: str = ''):
    """Xavier initialization"""
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# =============================================================================
# 基础组件
# =============================================================================

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP as used in Vision Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# =============================================================================
# Attention 模块
# =============================================================================

class Attention(nn.Module):
    """Multi-head Self Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossAttention(nn.Module):
    """Cross-Modal Attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, context):
        """
        Args:
            x: query tensor [B, N, C]
            context: key/value tensor [B, M, C]
        """
        B, N, C = x.shape
        _, M, _ = context.shape
        
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(context).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# =============================================================================
# Transformer Block
# =============================================================================

class TransformerBlock(nn.Module):
    """Standard Transformer Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# =============================================================================
# Embedding 层
# =============================================================================

class LinearEmbed(nn.Module):
    """线性嵌入层 - 将特征维度映射到 embed_dim"""
    def __init__(self, input_dim=300, output_dim=128):
        super().__init__()
        self.fc = nn.Conv1d(input_dim, output_dim, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        # x: [B, T, D] -> [B, D, T] -> [B, embed_dim, T] -> [B, T, embed_dim]
        x = x.transpose(1, 2)
        x = self.fc(x)
        x = x.transpose(1, 2)
        return x


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# =============================================================================
# 单模态编码器
# =============================================================================

class TextTransformerEncoder(nn.Module):
    """
    Text Transformer Encoder
    使用 Transformer 架构处理文本特征序列
    """
    def __init__(self, input_dim=300, embed_dim=128, depth=4, num_heads=4, 
                 mlp_ratio=4., drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 特征嵌入
        self.patch_embed = LinearEmbed(input_dim=input_dim, output_dim=embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=1000, dropout=drop_rate)
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.cls_token, std=.02)
        self.apply(init_weights_xavier)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] 文本特征序列
        Returns:
            cls_feat: [B, embed_dim] CLS token 特征
            seq_feat: [B, T+1, embed_dim] 完整序列特征
        """
        B = x.shape[0]
        
        # 特征嵌入
        x = self.patch_embed(x)  # [B, T, embed_dim]
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, embed_dim]
        
        # 位置编码
        x = self.pos_encoding(x)
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # 返回 CLS token 作为文本特征
        cls_feat = x[:, 0]
        
        return cls_feat, x


class AudioTransformerEncoder(nn.Module):
    """
    Audio Transformer Encoder (与 mosei 一致)
    使用 LinearEmbed + Transformer Blocks 处理音频特征
    """
    def __init__(self, input_dim=74, embed_dim=128, depth=4, num_heads=4, 
                 mlp_ratio=4., drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 特征嵌入 (与 mosei 的 LinearEmbed 一致)
        self.patch_embed = LinearEmbed(input_dim=input_dim, output_dim=embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(init_weights_xavier)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] 音频特征序列
        Returns:
            cls_feat: [B, embed_dim] CLS token 特征
            seq_feat: [B, T+1, embed_dim] 完整序列特征
        """
        B = x.shape[0]
        
        # 特征嵌入
        x = self.patch_embed(x)  # [B, T, embed_dim]
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, embed_dim]
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # 返回 CLS token 作为特征
        cls_feat = x[:, 0]
        
        return cls_feat, x


class VisualTransformerEncoder(nn.Module):
    """
    Visual Transformer Encoder (与 mosei 一致)
    使用 LinearEmbed + Transformer Blocks 处理视觉特征
    """
    def __init__(self, input_dim=35, embed_dim=128, depth=4, num_heads=4, 
                 mlp_ratio=4., drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # 特征嵌入 (与 mosei 的 LinearEmbed 一致)
        self.patch_embed = LinearEmbed(input_dim=input_dim, output_dim=embed_dim)
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=dpr[i]
            ) for i in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(init_weights_xavier)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] 视觉特征序列
        Returns:
            cls_feat: [B, embed_dim] CLS token 特征
            seq_feat: [B, T+1, embed_dim] 完整序列特征
        """
        B = x.shape[0]
        
        # 特征嵌入
        x = self.patch_embed(x)  # [B, T, embed_dim]
        
        # 添加 CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [B, T+1, embed_dim]
        
        # Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)
        
        # 返回 CLS token 作为特征
        cls_feat = x[:, 0]
        
        return cls_feat, x


# 保留旧的 MLP 版本以便向后兼容
class AudioMLPEncoder(nn.Module):
    """
    Audio MLP Encoder (旧版本，已弃用)
    使用 MLP 处理音频特征 (已经是预提取的特征)
    """
    def __init__(self, input_dim=74, embed_dim=128, hidden_dim=256, 
                 num_layers=2, drop_rate=0.1, pool_type='mean'):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        # 特征嵌入
        self.embed = LinearEmbed(input_dim=input_dim, output_dim=embed_dim)
        
        # MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_rate)
                ])
            elif i == num_layers - 1:
                layers.extend([
                    nn.Linear(hidden_dim, embed_dim),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_rate)
                ])
        
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.apply(init_weights_xavier)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] 音频特征序列
        Returns:
            feat: [B, embed_dim] 池化后的特征
            seq_feat: [B, T, embed_dim] 完整序列特征
        """
        # 特征嵌入
        x = self.embed(x)  # [B, T, embed_dim]
        
        # MLP
        x = self.mlp(x)  # [B, T, embed_dim]
        x = self.norm(x)
        
        # 池化
        if self.pool_type == 'mean':
            feat = x.mean(dim=1)
        elif self.pool_type == 'max':
            feat = x.max(dim=1)[0]
        elif self.pool_type == 'first':
            feat = x[:, 0]
        else:
            feat = x.mean(dim=1)
        
        return feat, x


class VisualMLPEncoder(nn.Module):
    """
    Visual MLP Encoder (旧版本，已弃用)
    使用 MLP 处理视觉特征 (已经是预提取的特征)
    """
    def __init__(self, input_dim=35, embed_dim=128, hidden_dim=256, 
                 num_layers=2, drop_rate=0.1, pool_type='mean'):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pool_type = pool_type
        
        # 特征嵌入
        self.embed = LinearEmbed(input_dim=input_dim, output_dim=embed_dim)
        
        # MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(embed_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_rate)
                ])
            elif i == num_layers - 1:
                layers.extend([
                    nn.Linear(hidden_dim, embed_dim),
                    nn.ReLU()
                ])
            else:
                layers.extend([
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop_rate)
                ])
        
        self.mlp = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        self.apply(init_weights_xavier)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, input_dim] 视觉特征序列
        Returns:
            feat: [B, embed_dim] 池化后的特征
            seq_feat: [B, T, embed_dim] 完整序列特征
        """
        # 特征嵌入
        x = self.embed(x)  # [B, T, embed_dim]
        
        # MLP
        x = self.mlp(x)  # [B, T, embed_dim]
        x = self.norm(x)
        
        # 池化
        if self.pool_type == 'mean':
            feat = x.mean(dim=1)
        elif self.pool_type == 'max':
            feat = x.max(dim=1)[0]
        elif self.pool_type == 'first':
            feat = x[:, 0]
        else:
            feat = x.mean(dim=1)
        
        return feat, x


# =============================================================================
# 融合模块
# =============================================================================

class ConcatFusion(nn.Module):
    """Concatenation Fusion"""
    def __init__(self, input_dim=384, output_dim=3):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, t, a, v):
        combined = torch.cat([t, a, v], dim=1)
        out = self.fc(combined)
        return out


class GatedFusion(nn.Module):
    """Gated Fusion"""
    def __init__(self, embed_dim=128, output_dim=3):
        super().__init__()
        self.gate_t = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
        self.gate_a = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
        self.gate_v = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(embed_dim * 3, output_dim)
    
    def forward(self, t, a, v):
        combined = torch.cat([t, a, v], dim=1)
        
        g_t = self.gate_t(combined)
        g_a = self.gate_a(combined)
        g_v = self.gate_v(combined)
        
        t = t * g_t
        a = a * g_a
        v = v * g_v
        
        out = self.fc(torch.cat([t, a, v], dim=1))
        return out


# =============================================================================
# 三模态 Transformer 模型
# =============================================================================

class TriModalTransformer(nn.Module):
    """
    三模态 Transformer 模型 (与 mosei 一致)
    - Text: Transformer 编码器 (LinearEmbed + Transformer Blocks)
    - Audio: Transformer 编码器 (LinearEmbed + Transformer Blocks)
    - Visual: Transformer 编码器 (LinearEmbed + Transformer Blocks)
    """
    def __init__(self, args):
        super().__init__()
        
        # 从 args 获取配置
        self.dataset = getattr(args, 'dataset', 'MOSEI')
        n_classes = getattr(args, 'n_classes', 3)
        
        # 特征维度
        text_dim = getattr(args, 'text_dim', 300)
        audio_dim = getattr(args, 'audio_dim', 74)
        visual_dim = getattr(args, 'visual_dim', 35)
        
        # 嵌入维度
        embed_dim = getattr(args, 'embed_dim', 128)
        hidden_dim = getattr(args, 'hidden_dim', 256)
        
        # Transformer 配置
        depth = getattr(args, 'transformer_depth', 4)
        num_heads = getattr(args, 'transformer_heads', 4)
        mlp_ratio = getattr(args, 'mlp_ratio', 4.0)
        drop_rate = getattr(args, 'drop_rate', 0.1)
        
        self.embed_dim = embed_dim
        self.n_classes = n_classes
        
        # 编码器 (全部使用 Transformer 架构，与 mosei 一致)
        self.text_encoder = TextTransformerEncoder(
            input_dim=text_dim, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate
        )
        
        self.audio_encoder = AudioTransformerEncoder(
            input_dim=audio_dim, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate
        )
        
        self.visual_encoder = VisualTransformerEncoder(
            input_dim=visual_dim, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate
        )
        
        # 单模态分类头
        self.head_text = nn.Linear(embed_dim, n_classes)
        self.head_audio = nn.Linear(embed_dim, n_classes)
        self.head_visual = nn.Linear(embed_dim, n_classes)
        
        # 融合分类头
        fusion_method = getattr(args, 'fusion_method', 'concat')
        if fusion_method == 'concat':
            self.fusion_head = ConcatFusion(input_dim=embed_dim * 3, output_dim=n_classes)
        elif fusion_method == 'gated':
            self.fusion_head = GatedFusion(embed_dim=embed_dim, output_dim=n_classes)
        else:
            self.fusion_head = ConcatFusion(input_dim=embed_dim * 3, output_dim=n_classes)
    
    def forward(self, text, audio, visual):
        """
        Forward pass
        
        Args:
            text: [B, T, D_text] 文本特征
            audio: [B, T, D_audio] 音频特征
            visual: [B, T, D_visual] 视觉特征
        
        Returns:
            t_feat, a_feat, v_feat: 各模态特征
            out_t, out_a, out_v: 单模态输出
            out_fusion: 融合输出
        """
        # 编码
        t_feat, _ = self.text_encoder(text)
        a_feat, _ = self.audio_encoder(audio)
        v_feat, _ = self.visual_encoder(visual)
        
        # 单模态分类
        out_t = self.head_text(t_feat)
        out_a = self.head_audio(a_feat)
        out_v = self.head_visual(v_feat)
        
        # 融合分类
        out_fusion = self.fusion_head(t_feat, a_feat, v_feat)
        
        return t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == '__main__':
    # 创建一个简单的 args 对象用于测试
    class Args:
        dataset = 'MOSEI'
        n_classes = 3
        text_dim = 300
        audio_dim = 74
        visual_dim = 35
        embed_dim = 128
        hidden_dim = 256
        transformer_depth = 4
        transformer_heads = 4
        mlp_ratio = 4.0
        drop_rate = 0.1
        fusion_method = 'concat'
    
    args = Args()
    
    # 创建模型
    model = TriModalTransformer(args)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    B, T_text, T_audio, T_visual = 4, 50, 100, 50
    text = torch.randn(B, T_text, 300)
    audio = torch.randn(B, T_audio, 74)
    visual = torch.randn(B, T_visual, 35)
    
    t_feat, a_feat, v_feat, out_t, out_a, out_v, out_fusion = model(text, audio, visual)
    
    print(f"Text feature: {t_feat.shape}")
    print(f"Audio feature: {a_feat.shape}")
    print(f"Visual feature: {v_feat.shape}")
    print(f"Text output: {out_t.shape}")
    print(f"Audio output: {out_a.shape}")
    print(f"Visual output: {out_v.shape}")
    print(f"Fusion output: {out_fusion.shape}")

