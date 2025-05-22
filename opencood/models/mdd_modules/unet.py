import sys
sys.path.append('/raid/luogy/gkx/CoAlign-main')
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from opencood.models.sub_modules.attention import SelfAttention, CrossAttention

from dataclasses import dataclass
from typing import Tuple, Optional
import time
# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True) #16


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class SelfAttentionBlock(nn.Module):
    """增强版自注意力块，支持多头实现"""
    def __init__(self, channels, num_heads=8, attn_drop=0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.norm = Normalize(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]
        """
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        
        # 生成QKV投影并重塑为多头形式
        qkv = self.qkv(x_norm)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H*W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]  # 各自形状为 [B, heads, H*W, head_dim]
        
        # 高效注意力计算
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, H*W]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        out = (attn @ v)  # [B, heads, H*W, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # 输出投影
        out = self.proj(out)
        
        return x + out

class CrossAttentionBlock(nn.Module):
    def __init__(self, num_heads, in_channels, cond_channels, attn_drop=0.):
        super().__init__()
        self.channels = in_channels
        self.cond_dim = cond_channels
        self.num_heads = num_heads
        assert self.channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = self.channels // num_heads
        # self.scale = self.head_dim ** -0.5
        self.scale = nn.Parameter(torch.ones(1) * 2.0)  # 可学习的权重
        
        self.norm = Normalize(self.channels)
        self.q_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True)
        
        # 分离键值投影以提高效率
        self.k_proj = nn.Linear(self.cond_dim, self.channels)
        self.v_proj = nn.Linear(self.cond_dim, self.channels)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        
         # 初始化权重
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x, cond):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
            cond: 条件特征 [B, L, D] 其中L是序列长度，D是条件维度
        Returns:
            输出特征图 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 处理特征图查询
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W)
        q = q.permute(0, 1, 3, 2)  # [B, heads, H*W, head_dim]
        
        # 处理条件键值
        k = self.k_proj(cond)  # [B, L, C]
        v = self.v_proj(cond)  # [B, L, C]
        
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, L, head_dim]
        
        # 计算注意力
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重
        out = (attn @ v)  # [B, heads, H*W, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        
        # 输出投影
        out = self.proj(out)
        
        return x + out

class FeedForwardBlock(nn.Module):
    """标准前馈网络块，使用GEGLU激活"""
    def __init__(self, channels, expansion_factor=8, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        
        self.norm = Normalize(channels)
        self.proj1 = nn.Conv2d(channels, channels * expansion_factor, kernel_size=1)
        self.proj2 = nn.Conv2d(channels * expansion_factor // 2, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight)
        # 0.0近似残差预激活
        nn.init.zeros_(self.proj2.bias)
        
    def geglu(self, x):
        """GEGLU激活函数"""
        x, gate = x.chunk(2, dim=1)
        return x * F.gelu(gate)
        
    def forward(self, x):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
        Returns:
            输出特征图 [B, C, H, W]
        """
        h = self.norm(x)
        h = self.proj1(h)
        h = self.geglu(h)
        h = self.drop(h)
        h = self.proj2(h)
        return x + h

class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, channels: int, d_cond=None,ff_expansion=8, attn_dropout=0.0, dropout=0.0, ):
        super().__init__()
        super().__init__()
        self.channels = channels
        self.cond_dim = d_cond
        
        # 自注意力子块
        self.self_attn = SelfAttentionBlock(
            channels=channels,
            num_heads=n_head,
            attn_drop=attn_dropout
        )
        
        # 条件交叉注意力子块（如果需要）
        if d_cond is not None:
            self.cross_attn = CrossAttentionBlock(
                in_channels=channels,
                cond_channels=d_cond,
                num_heads=n_head,
                attn_drop=attn_dropout
            )
        else:
            self.cross_attn = None
            
        # 前馈网络子块
        self.ff = FeedForwardBlock(
            channels=channels,
            expansion_factor=ff_expansion,
            dropout=dropout
        )
        
    def forward(self, x, cond=None):
        """
        Args:
            x: 输入特征图 [B, C, H, W]
            cond: 可选条件特征 [B, L, D]，如果self.cond_dim不为None
        Returns:
            输出特征图 [B, C, H, W]
        """
        # 自注意力
        x = self.self_attn(x)
        
        # 条件交叉注意力（如果有）
        if self.cross_attn is not None and cond is not None:
            x = self.cross_attn(x, cond)
            
        # 前馈网络
        x = self.ff(x)
        
        return x

class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = getattr(config.model, 'resolution', 32)
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.d_cond = getattr(config.model, 'd_cond', None)
        self.n_head = config.model.n_head

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res <= attn_resolutions:
                    attn.append(AttentionBlock(n_head = self.n_head,channels = block_in, d_cond=self.d_cond))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.att = AttentionBlock(n_head = self.n_head,channels = block_in, d_cond=self.d_cond)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res <= attn_resolutions:
                    attn.append(AttentionBlock(n_head = self.n_head,channels = block_in, d_cond=self.d_cond))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x, t, cond = None):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch).to(x.device)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h, cond)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.att(h, cond)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                t = torch.cat([h, hs.pop()], dim=1)
                h = self.up[i_level].block[i_block](t, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, cond)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h



# @dataclass
# class ModelConfig:
#     ch: int  # Base channel count
#     out_ch: int  # Output channels
#     ch_mult: Tuple[int, ...]  # Channel multiplier at different resolutions
#     num_res_blocks: int  # Number of residual blocks
#     attn_resolutions: Tuple[int, ...]  # Resolutions at which to apply attention
#     dropout: float  # Dropout rate
#     in_channels: int  # Input channels (note: in your code this is * 2)
#     resamp_with_conv: bool  # Whether to use convolution for resampling
#     n_head: int  # num of multicross head
#     d_cond: Optional[int] = None  # condition embedding dim, 可以为None或int类型


# @dataclass
# class Config:
#     model: ModelConfig
    
# def print_trainable_params(model, model_name):
#     total_params = 0
#     trainable_params = 0
#     trainable_bytes = 0  # 统计可训练参数字节数
    
#     # 数据类型与字节数的映射
#     dtype_bytes = {
#         torch.float32: 4,    torch.float: 4,
#         torch.float64: 8,    torch.double: 8,
#         torch.float16: 2,    torch.half: 2,
#         torch.bfloat16: 2,
#         torch.int8: 1,       torch.uint8: 1,
#         torch.int16: 2,      torch.short: 2,
#         torch.int32: 4,      torch.int: 4,
#         torch.int64: 8,      torch.long: 8,
#         torch.bool: 1
#     }
    
#     print(f"\n{model_name} 训练状态：")
#     for name, param in model.named_parameters():
#         num = param.numel()
#         total_params += num
#         if param.requires_grad:
#             trainable_params += num
#             status = "可训练"
#             # 计算该参数的字节大小
#             bytes_per_elem = dtype_bytes.get(param.dtype, 4)  # 未知类型默认4字节
#             trainable_bytes += num * bytes_per_elem
#         else:
#             status = "冻结"
#         print(f"{name.ljust(60)} | {status.ljust(8)} | 形状: {tuple(param.shape)}")
    
#     # 转换为MB (1 MB = 1024*1024 Bytes)
#     trainable_mb = trainable_bytes / (1024 ** 2)
    
#     print(f"\n总参数量: {total_params:,}")
#     print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params:.2%})")
#     print(f"可训练参数总大小: {trainable_mb:.2f} MB")

# # Example usage:
# config = Config(
#     model=ModelConfig(
#         in_channels= 5,
#         out_ch= 10,
#         ch=8,
#         ch_mult= [1, 1],
#         num_res_blocks= 2,
#         attn_resolutions= [16],
#         dropout= 0.0,
#         resamp_with_conv= True,
#         n_head=8
#     )
# )

# diffUNet = DiffusionUNet(config)
# print_trainable_params(diffUNet,"diffUNet")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")
# diffUNet = diffUNet.to(device)
# batch_size = 1
# channels = 10 #in_channels*2
# height = 100 #any
# width = 100 #any
# # d_cond = 768 #any
# d_channel = 64

# # Sample input tensor
# x = torch.randn(batch_size, channels, height, width).to(device)

# t = torch.randint(0, 1000, (batch_size,)).to(device)

# # Condition tensor (assuming same shape as input)
# # cond = torch.randn(batch_size, d_channel, d_cond)

# # Forward pass
# output = diffUNet(x, t, None)

# # Print shapes for verification
# print(f"Input shape: {x.shape}")
# print(f"Time shape: {t.shape}")
# # print(f"Condition shape: {cond.shape}")
# print(f"Output shape: {output.shape}")

# 总参数量: 1,209,216
# 可训练参数: 1,209,216 (100.00%)
# 可训练参数总大小: 4.61 MB
# Input shape: torch.Size([4, 128, 64, 64])
# Time shape: torch.Size([4])
# Condition shape: torch.Size([4, 64, 768])
# Output shape: torch.Size([4, 64, 64, 64])