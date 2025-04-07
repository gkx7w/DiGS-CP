# 防止matplotlib报错，循环导入，应在主文件使用
# import os
# os.environ['MPLBACKEND'] = 'Agg'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from functools import partial
import time
from opencood.utils.MDD_utils import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config, extract_into_tensor, make_beta_schedule, noise_like, detach
import os
from opencood.models.attresnet_modules.self_attn import AttFusion
from opencood.models.attresnet_modules.auto_encoder import AutoEncoder
from opencood.models.mdd_modules.unet import DiffusionUNet
INTERPOLATE_MODE = 'bilinear'
def tolist(a):
    try:
        return [tolist(i) for i in a]
    except TypeError:
        return a

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    from DIT
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb
    

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim*2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, kv):
        B, N, C = x.shape
        _, T, _ = kv.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # (B, heads, N, C//heads)
        kv = self.kv(kv).reshape(B, T, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # (2, B, heads, T, C//heads)
        k, v = kv[0], kv[1]# (B, heads, T, C//heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, heads, N, T), N is 1
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class CrossBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = CrossAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.mlp_kv = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_kv = norm_layer(dim)
        self.norm2_kv = norm_layer(dim)

    def forward(self, x, kv):
        kv = kv + self.drop_path(self.mlp_kv(self.norm2_kv(kv)))

        x = x + self.drop_path(self.attn(self.norm1(x), self.norm_kv(kv)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, kv

class Denosier(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.depth = 4
        self.crossblocks = nn.ModuleList([CrossBlock(embed_dim, num_heads=4) for _ in range(self.depth)])

        self.pre_layer = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1)

        
        self.initialize_weights()

        self.t_embedder = TimestepEmbedder(embed_dim)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        self.apply(_basic_init)


    def forward(self, feat, noisy_masks, upsam=False, t=None):

        out = {}
        t = self.t_embedder(t)[:,:,None,None]

        resize_scale = 0.25
        cur_feat = F.interpolate(feat, scale_factor=resize_scale, mode='bilinear', align_corners=False)
        B, C, H, W = cur_feat.shape
        kv = cur_feat.reshape(B, C, -1).transpose(1,2) # (B, N, C)

        x = noisy_masks
        x = self.pre_layer(x)
        x = x + t
        x0 = x
        cur_kv = kv

        x = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)
        B, C, H, W = x.shape
        x = x.reshape(B,C,-1).transpose(1,2)

        for _d in range(self.depth):
            cur_kv, x = self.crossblocks[_d](cur_kv, x)
        x = cur_kv

        x = x.reshape(B, H, W, C).permute(0,3,1,2)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        out = x + x0
        
        return out


class Config:
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Config(v)
			else:
				self.__dict__[k] = v

class Cond_Diff_Denoise(nn.Module):
    def __init__(self,  model_cfg, embed_dim,):
        super().__init__()
        ### hyper-parameters
        # self.parameterization = 'eps'
        self.parameterization = 'x0'
        beta_schedule="linear"
        config = Config(model_cfg)
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        timesteps = config.diffusion.num_diffusion_timesteps
        linear_start=5e-3
        linear_end=5e-2
        self.v_posterior = v_posterior =0 
        self.loss_type="l2"
        self.signal_scaling_rate = 1
        ###

        new_embed_dim = embed_dim
        #self.denoiser = Denosier(new_embed_dim)
        
        self.denoiser = DiffusionUNet(config)

        # q sampling
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # diffusion loss
        learn_logvar = False
        logvar_init = 0
        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        self.l_simple_weight = 1.

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        if len(lvlb_weights) > 1:
            lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()



    def q_sample(self, x_start, t, noise=None):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x_noisy, cond, upsam, t, clip_denoised: bool):
        model_out = self.gen_pred( x_noisy, cond, upsam, t)

        x = x_noisy
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        if upsam: # last sampling step
            model_mean = x_recon
            posterior_variance, posterior_log_variance = 0, 0
        else:
            x_recon = F.interpolate(x_recon, x.shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t) # Why do we need this?

        return model_mean, posterior_variance, posterior_log_variance, model_out


    def p_sample(self, x_noisy, cond, t, upsam, clip_denoised=False, repeat_noise=False):
        model_mean, _, model_log_variance, model_out = self.p_mean_variance(x_noisy, cond, upsam, t=t, clip_denoised=clip_denoised)

        x = x_noisy
        b, *_, device = *x.shape, x.device
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if not upsam:
            out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise #, model_out
        else:
            out = model_mean

        return out

    def gen_pred(self, x_noisy, cond, upsam=False, t=None):
        model_out = self.denoiser(x_noisy, t.float(), cond)
        # model_out = self.denoiser(torch.cat([feat, noisy_masks], dim=1), t.float())
        return model_out

    def p_sample_loop(self, x_noisy, cond, latent_shape):
        b = latent_shape[0]
        num_timesteps = self.num_timesteps 
        
        for t in reversed(range(0, num_timesteps)):
            x_noisy = F.interpolate(x_noisy, latent_shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            x_noisy = self.p_sample(x_noisy, cond, torch.full((b,), t, device=x_noisy.device, dtype=torch.long), 
                                                upsam=True if t==0 else False)
        return x_noisy

    def forward(self, data_dict):
        # 获取batch中每个GT框的特征
        batch_gt_spatial_features = data_dict['batch_gt_spatial_features']
        coords = data_dict['voxel_coords']
        gt_masks = data_dict['voxel_gt_mask']
        # 解耦后的融合特征为条件，无则为none
        cond = data_dict.get('fused_object_factors', None)
        if cond is not None and len(cond) == 0:
            cond = None
        combined_pred = cond
        # 最终结果存储 - 保持与输入相同的嵌套结构
        batch_pred_features = []
        if self.training:
            # 对每个batch处理
            for batch_idx, gt_features in enumerate(batch_gt_spatial_features):
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]  # (batch_idx_voxel, 4)
                # 获取该batch中的gt_mask
                this_gt_mask = gt_masks[batch_mask]  # (batch_idx_voxel, )
                unique_values_in_mask = torch.unique(this_gt_mask)
                # 获取相应的条件
                gt_cond = None
                if cond is not None:
                    if isinstance(cond, list) and batch_idx < len(cond):
                        gt_cond = cond[batch_idx]
                    else:
                        gt_cond = combined_pred
                    # 创建一个布尔掩码，指示哪些元素要保留
                    # 假设gt_cond的长度与可能的mask值对应
                    keep_indices = []
                    for i in range(len(gt_cond)):
                        # 检查索引+1是否在mask的唯一值中
                        if i in unique_values_in_mask:
                            keep_indices.append(i)
                    # 如果有要保留的索引，则过滤gt_cond
                    if keep_indices:
                        # 使用索引选择器保留需要的元素
                        gt_cond = gt_cond[keep_indices]
                    else:
                        # 如果没有要保留的元素，创建一个空tensor保持原始维度结构
                        gt_cond = torch.zeros((0,) + gt_cond.shape[1:], device=gt_cond.device, dtype=gt_cond.dtype)
                
                x_start = gt_features  # 当前GT框的特征
                latent_shape = x_start.shape
                t = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device, dtype=torch.long)
                noise = default(None, lambda: torch.randn_like(x_start))
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                # 逆扩散过程
                for _t in reversed(range(1, self.num_timesteps)):
                    _t = torch.full((x_start.shape[0],), _t, device=x_start.device, dtype=torch.long)
                    x_noisy = self.p_sample(x_noisy, gt_cond, _t, upsam=False)
                _t = 0
                _t = torch.full((x_start.shape[0],), _t, device=x_start.device, dtype=torch.long)
                x_noisy = self.p_sample(x_noisy, gt_cond, _t, upsam=True)
                # 存储当前GT框的预测结果
                batch_pred_features.append(x_noisy)
            data_dict['pred_feature'] = batch_pred_features
        else:
            # 对每个batch处理
            for batch_idx, gt_features in enumerate(batch_gt_spatial_features):
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]  # (batch_idx_voxel, 4)
                # 获取该batch中的gt_mask
                this_gt_mask = gt_masks[batch_mask]  # (batch_idx_voxel, )
                unique_values_in_mask = torch.unique(this_gt_mask)
                
                # 获取相应的条件
                gt_cond = None
                if cond is not None:
                    if isinstance(cond, list) and batch_idx < len(cond):
                        gt_cond = cond[batch_idx]
                    else:
                        gt_cond = combined_pred
                    
                    # 创建一个布尔掩码，指示哪些元素要保留
                    keep_indices = []
                    for i in range(len(gt_cond)):
                        # 检查索引是否在mask的唯一值中
                        if i in unique_values_in_mask:
                            keep_indices.append(i)
                    # 如果有要保留的索引，则过滤gt_cond
                    if keep_indices:
                        # 使用索引选择器保留需要的元素
                        gt_cond = gt_cond[keep_indices]
                    else:
                        # 如果没有要保留的元素，创建一个空tensor保持原始维度结构
                        gt_cond = torch.zeros((0,) + gt_cond.shape[1:], device=gt_cond.device, dtype=gt_cond.dtype)
                
                x_start = gt_features  # 当前GT框的特征
                latent_shape = x_start.shape
                noise = default(None, lambda: torch.randn_like(x_start))
                t = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device, dtype=torch.long)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                
                # 逆扩散过程
                x_noisy = self.p_sample_loop(x_noisy, gt_cond, x_start.shape)
                
                # 存储当前GT框的预测结果
                batch_pred_features.append(x_noisy)
            data_dict['pred_feature'] = batch_pred_features
        
        return data_dict


def main():
    config = {
        "model": {
            "in_channels": 5, #in_channels*2
            "out_ch": 10, #out_ch=in_channels*2
            "ch": 8,
            "ch_mult": [1, 1],
            "num_res_blocks": 2,
            "attn_resolutions": [16],
            "dropout": 0.0,
            "resamp_with_conv": True,
            "n_head": 8
        },
        "diffusion": {
            "beta_schedule": "linear",
            "beta_start": 0.0005,
            "beta_end": 0.02,
            "num_diffusion_timesteps": 3
        }
    }
    model = Cond_Diff_Denoise(config, 6)
    data_dict = {}
    data_dict['spatial_features'] = torch.randn(1, 10, 100, 100)
    data_dict = model(data_dict)

if __name__ == "__main__":
    main()

