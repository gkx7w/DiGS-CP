import torch
from torch import nn
from torch.nn import functional as F
import math

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True) #16

class SelfAttentionBlock(nn.Module):
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

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)

        qkv = self.qkv(x_norm)  # [B, 3*C, H, W]
        qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, H*W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # [3, B, heads, H*W, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, H*W]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v)  # [B, heads, H*W, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
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
        self.scale = nn.Parameter(torch.ones(1) * 2.0)  
        
        self.norm = Normalize(self.channels)
        self.q_proj = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=True)
        
        self.k_proj = nn.Linear(self.cond_dim, self.channels)
        self.v_proj = nn.Linear(self.cond_dim, self.channels)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(self.channels, self.channels, kernel_size=1)
        
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
    
    def forward(self, x, cond):
        B, C, H, W = x.shape
        
        x_norm = self.norm(x)
        q = self.q_proj(x_norm)
        q = q.reshape(B, self.num_heads, self.head_dim, H*W)
        q = q.permute(0, 1, 3, 2)  # [B, heads, H*W, head_dim]
        
        k = self.k_proj(cond)
        v = self.v_proj(cond) 
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) 
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, H*W, L]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        out = (attn @ v)  # [B, heads, H*W, head_dim]
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        out = self.proj(out)
        return x + out

class FeedForwardBlock(nn.Module):
    def __init__(self, channels, expansion_factor=8, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.expansion_factor = expansion_factor
        self.dropout = dropout
        
        self.norm = Normalize(channels)
        self.proj1 = nn.Conv2d(channels, channels * expansion_factor, kernel_size=1)
        self.proj2 = nn.Conv2d(channels * expansion_factor // 2, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)
        
        nn.init.xavier_uniform_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)
        nn.init.xavier_uniform_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)
        
    def geglu(self, x):
        x, gate = x.chunk(2, dim=1)
        return x * F.gelu(gate)
        
    def forward(self, x):
        h = self.norm(x)
        h = self.proj1(h)
        h = self.geglu(h)
        h = self.drop(h)
        h = self.proj2(h)
        return x + h
