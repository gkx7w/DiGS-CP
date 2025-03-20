import sys
sys.path.append('/raid/luogy/gkx/CoAlign-main')
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from opencood.models.sub_modules.attention import SelfAttention, CrossAttention

from dataclasses import dataclass
from typing import Tuple, Optional

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
    return torch.nn.GroupNorm(num_groups=4, num_channels=in_channels, eps=1e-6, affine=True)


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

# 不用这个，不能加引导条件
class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_
        

class AttentionBlock(nn.Module):
    def __init__(self, n_head: int, channels: int, d_cond=None):
        super().__init__()
        # channels = n_head * n_embd
        # 有空把这个num_groups改成超参数
        self.groupnorm = nn.GroupNorm(2, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        # 如果有条件输入d_cond，则使用交叉注意力
        if d_cond is not None:
            self.layernorm_2 = nn.LayerNorm(channels)
            self.attention_2 = CrossAttention(n_head, channels, d_cond, in_proj_bias=False)
        else:
            self.layernorm_2 = None
            self.attention_2 = None
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, cond=None):
        residue_long = x
        # 通道分组归一化
        x = self.groupnorm(x)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)  

        residue_short = x
        x = self.layernorm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # 只有当存在条件输入和attention_2时才执行交叉注意力
        if self.attention_2 is not None and cond is not None:
            residue_short = x
            x = self.layernorm_2(x)
            x = self.attention_2(x, cond)
            x += residue_short

        residue_short = x
        x = self.layernorm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.linear_geglu_2(x)
        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))    # (n, c, h, w)  

        return self.conv_output(x) + residue_long


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2
        resolution = 128
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