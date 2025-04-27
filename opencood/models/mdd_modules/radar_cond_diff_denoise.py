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
import copy
from collections import namedtuple
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from scipy import linalg
from opencood.models.attresnet_modules.self_attn import AttFusion
from opencood.models.attresnet_modules.auto_encoder import AutoEncoder
from opencood.models.mdd_modules.unet import DiffusionUNet
from opencood.models.mdd_modules.my_unet import Unet

INTERPOLATE_MODE = 'bilinear'
def tolist(a):
    try:
        return [tolist(i) for i in a]
    except TypeError:
        return a

from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])
def identity(t, *args, **kwargs):
    return t

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
        self.train_mode = 1
        config = Config(model_cfg)
        self.parameterization = getattr(config.diffusion, 'parameterization', 'x0')
        beta_schedule=config.diffusion.beta_schedule
        self.num_timesteps = config.diffusion.num_diffusion_timesteps
        timesteps = config.diffusion.num_diffusion_timesteps
        self.if_normalize = getattr(config.diffusion, 'if_normalize', False)
        self.guidance_scale = getattr(config.diffusion, 'guidance_scale', 1.0)  # 默认值为1.0
        self.ddim_sampling_eta = getattr(config.diffusion, 'ddim_sampling_eta', 0.0)  # 默认值为0.0
        self.sampling_timesteps = getattr(config.diffusion, 'sampling_timesteps', 3) # default num sampling timesteps to number of timesteps at training
        linear_start=5e-3
        linear_end=5e-2
        self.v_posterior = v_posterior =0 
        self.loss_type="l2"
        self.signal_scaling_rate = 1
        ###

        # new_embed_dim = embed_dim
        # self.denoiser = Denosier(new_embed_dim)
        
        # self.denoiser = DiffusionUNet(config)
        self.denoiser = Unet(config)
        H, W, _ = config.diffusion.grid_size
        C = config.model.ch
        self.norm_layer = nn.LayerNorm([W, H],elementwise_affine=False)
        self.group_norm = nn.GroupNorm(num_groups=C, num_channels=C)
        self.instance_norm = nn.InstanceNorm2d(C, affine=False)

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


    def denormalize_from_minus1_1(self, x):
        # 从 [-1, 1] 反归一化到原始范围 [original_min, original_max]
        # 先将 [-1, 1] 转回 [0, 1]
        x_zero_one = (x + 1) / 2
        # 然后缩放到原始范围
        return x_zero_one * (self.original_max - self.original_min) + self.original_min
    
    def normalize_to_minus1_1(self, x):
        x_min, x_max = x.min(), x.max()
        
        # 保存这些值以供反归一化使用
        self.original_min = x_min.item()
        self.original_max = x_max.item()
        
        # 归一化逻辑
        x_normalized = (x - x_min) / (x_max - x_min)
        return x_normalized * 2 - 1
    
    def min_max_normalize_to_range(self, x, target_min=-1.0, target_max=1.0):
        # 获取每个通道的最大和最小值
        # 保持维度以便广播
        min_vals = x.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
        max_vals = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        
        # 避免除以零
        denominator = max_vals - min_vals
        # 添加一个小的epsilon值来避免除以零
        denominator = torch.clamp(denominator, min=1e-8)
        
        # 缩放到 [0, 1]
        x_scaled = (x - min_vals) / denominator
        
        # 缩放到目标范围 [target_min, target_max]
        return x_scaled * (target_max - target_min) + target_min

    def channel_balance_norm(self, x):
        # 先对每个通道独立归一化
        mean = x.mean(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        std = x.std(dim=(2, 3), keepdim=True)    # [N, C, 1, 1]
        x_norm = (x - mean) / (std + 1e-5)
        
        # 再做通道间的均衡
        channel_max = x_norm.amax(dim=(2, 3), keepdim=True)  # [N, C, 1, 1]
        return x_norm / (channel_max + 1e-5)
    
    def q_sample(self, x_start, t, noise=None):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x_0):
            return (
                (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x_0) / \
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
            )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    # 添加CFG预测函数
    def gen_pred_with_cfg(self, x_noisy, t, cond=None, guidance_scale=3.0):
        """使用Classifier-Free Guidance进行预测"""
        if guidance_scale == 1.0 or cond is None:
            # 不需要CFG，直接返回条件预测
            return self.gen_pred(x_noisy, cond, False, t)
        
        # 无条件预测
        uncond_pred = self.gen_pred(x_noisy, None, False, t)
        
        # 有条件预测
        cond_pred = self.gen_pred(x_noisy, cond, False, t)
        
        # 应用CFG公式: 无条件预测 + guidance_scale * (有条件预测 - 无条件预测)
        return uncond_pred + guidance_scale * (cond_pred - uncond_pred)
    
    @torch.inference_mode()
    def model_predictions(self, x, t, cond = None, guidance_scale=1.0, clip_x_start = True, rederive_pred_noise = True):
        if guidance_scale == 1.0 or cond is None:
            model_output = self.denoiser(x, t.float(), cond)
        else:
            # CFG预测逻辑
            with torch.no_grad():
                uncond_output = self.denoiser(x, t.float(), None)
            cond_output = self.denoiser(x, t.float(), cond)
            model_output = uncond_output + guidance_scale * (cond_output - uncond_output)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity
        if self.parameterization == "eps":
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            
            if self.if_normalize:
                x_start = maybe_clip(x_start)
                if clip_x_start and rederive_pred_noise:
                    pred_noise = self.predict_noise_from_start(x, t, x_start)
            

        elif self.parameterization == "x0":
            x_start = model_output
            if self.if_normalize:
                x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)
        
        return ModelPrediction(pred_noise, x_start)
    
    def p_mean_variance(self, x_noisy, cond, upsam, t, clip_denoised: bool, guidance_scale=1.0):
        """计算逆扩散过程的均值和方差, 支持CFG"""
        if guidance_scale > 1.0 and cond is not None:
            # 使用CFG进行预测
            model_out = self.gen_pred_with_cfg(x_noisy, t, cond, guidance_scale)
        else:
            # 标准预测
            model_out = self.gen_pred(x_noisy, cond, upsam, t)

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


    def p_sample(self, x_noisy, cond, t, upsam, clip_denoised=False, repeat_noise=False, guidance_kwargs=None, guidance_scale=1.0):
        """单步采样, 支持CFG"""
        model_mean, variance, model_log_variance, model_out = self.p_mean_variance(
            x_noisy, cond, upsam, t=t, clip_denoised=clip_denoised, guidance_scale=guidance_scale
        )

        x = x_noisy
        b, *_, device = *x.shape, x.device
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if not upsam:
            out = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise #, model_out
        else:
            out = model_mean
        if exists(guidance_kwargs):
            pass
        
        return out

    def gen_pred(self, x_noisy, cond, upsam=False, t=None):
        model_out = self.denoiser(x_noisy, t.float(), cond)
        # model_out = self.denoiser(torch.cat([feat, noisy_masks], dim=1), t.float())
        return model_out

    def p_sample_loop(self, x_noisy, cond, latent_shape, guidance_scale=1.0):
        """完整采样循环, 支持CFG"""
        b = latent_shape[0]
        num_timesteps = self.sampling_timesteps
        
        for t in reversed(range(0, num_timesteps)):
            x_noisy = F.interpolate(x_noisy, latent_shape[2:], mode=INTERPOLATE_MODE, align_corners=False)
            x_noisy = self.p_sample(
                x_noisy, cond, torch.full((b,), t, device=x_noisy.device, dtype=torch.long),
                upsam=True if t==0 else False, guidance_scale=guidance_scale
            )
        return x_noisy

    @torch.inference_mode()
    def ddim_sample(self, shape, device, cond, x_start = None, return_all_timesteps = False, guidance_scale=1.0):
        """DDIM采样, 支持CFG"""
        batch, total_timesteps, sampling_timesteps, eta = shape[0], self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = torch.linspace(-1, total_timesteps - 1, steps = sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        x_noisy = default(x_start, torch.randn(shape, device = device))
        
        
        # t = torch.full((shape[0],), 3, device=device, dtype=torch.long)
        # noise = default(None, lambda: torch.randn_like(x_start))
        # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # x_noisy = self.normalize_to_minus1_1(x_noisy)
        
        # x_noisy_init_np = normalize_to_0_1(x_noisy).cpu().numpy()
        # # fid_noisy_init_score = compute_frechet_distance(x_noisy_init_np, self.gt_np)
        # psnr_noisy_init_value = calculate_psnr(x_noisy_init_np, self.gt_np)
        # ssim_noisy_init_value = calculate_ssim(x_noisy_init_np, self.gt_np)
        # self.noisy_versions.append({
        #     'type': 'Pure Noise',
        #     'tensor': x_noisy_init_np,
        #     'psnr': psnr_noisy_init_value,
        #     'ssim': ssim_noisy_init_value,
        #     # 'fd': fid_noisy_init_score
        # })
        
        x_noisys = [x_noisy]

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device = x_noisy.device, dtype = torch.long)
            pred_noise, x_start, *_ = self.model_predictions(x_noisy, time_cond, cond, guidance_scale=guidance_scale)

            if time_next < 0:
                x_noisy = x_start
                x_noisys.append(x_noisy)
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(x_noisy)

            x_noisy = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

            x_noisys.append(x_noisy)

        ret = x_noisy if not return_all_timesteps else torch.stack(x_noisys, dim = 1)
        if self.if_normalize:
            ret = self.denormalize_from_minus1_1(ret)
        return ret
    
    @torch.inference_mode()
    def diffusion_inference(self,x_t, cond,num_inference_steps=50, eta=0.0):
        """
        使用diffusers中的DDIM scheduler实现推理过程
        
        Args:
            model: 您的UNet或其他噪声预测模型
            x_t: 初始噪声图像
            num_inference_steps: 推理步数
            eta: DDIM的随机性参数，0表示完全确定性
        
        Returns:
            去噪后的图像
        """
        from diffusers.schedulers import DDIMScheduler
        # 初始化scheduler
        if self.parameterization == "eps":
            prediction_type = "epsilon"
        elif self.parameterization == "x0":
            prediction_type = "sample"
        
        if self.if_normalize:
            clip_sample = True
        else:
            clip_sample = False
                
        
        scheduler = DDIMScheduler(
            beta_start=5e-3,  # 改为与训练相同
            beta_end=5e-2,    # 改为与训练相同
            beta_schedule="linear", # 假设您使用线性调度
            clip_sample=clip_sample,
            set_alpha_to_one=True,
            prediction_type=prediction_type
        )
        
        x_noisy_init_np = normalize_to_0_1(x_t).cpu().numpy()
        # fid_noisy_init_score = compute_frechet_distance(x_noisy_init_np, self.gt_np)
        psnr_noisy_init_value = calculate_psnr(x_noisy_init_np, self.gt_np)
        ssim_noisy_init_value = calculate_ssim(x_noisy_init_np, self.gt_np)
        self.noisy_versions.append({
            'type': 'Pure Noise',
            'tensor': x_noisy_init_np,
            'psnr': psnr_noisy_init_value,
            'ssim': ssim_noisy_init_value,
            # 'fd': fid_noisy_init_score
        })
        
        # 设置时间步
        scheduler.set_timesteps(num_inference_steps)
        
        # 推理循环
        for t in scheduler.timesteps:
            # 1. 通过模型预测噪声
            timestep = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
            with torch.no_grad():
                model_output = self.denoiser(x_t, timestep,cond)
            
            # 2. 使用scheduler计算x_{t-1}
            x_t = scheduler.step(
                model_output=model_output,
                timestep=t,
                sample=x_t,
                eta=eta,
                use_clipped_model_output=False
            ).prev_sample
            
        if self.if_normalize:
            x_t = self.denormalize_from_minus1_1(x_t)
        return x_t
    
    
    def forward(self, data_dict):
        # 获取batch中每个box框的特征
        batch_gt_spatial_features = data_dict['batch_gt_spatial_features']
        coords = data_dict['voxel_coords']
        box_masks = data_dict['voxel_gt_mask']
        # 解耦后的融合特征为条件，无则为none
        cond = data_dict.get('fused_object_factors', None)
        if cond is not None and len(cond) == 0:
            cond = None
        combined_pred = cond
        # 最终结果存储 - 保持与输入相同的嵌套结构
        batch_model_out = []
        batch_gt_noise = []
        batch_gt_x0 = []
        batch_t = []
        
        #----------------------------------------评测----------------------------
        # inf = True
        inf = False
        x_noisy_with_cond_list = []
        x_noisy_no_cond_list = []
        noise_init_list =[]
        noise_list = []
        x_list = []
        
        if self.training:
            # 对每个batch处理
            for batch_idx, gt_features in enumerate(batch_gt_spatial_features):
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]  # (batch_idx_voxel, 4)
                # 获取该batch中的box_mask
                this_box_mask = box_masks[batch_mask]  # (batch_idx_voxel, )
                unique_values_in_mask = torch.unique(this_box_mask)
                # 获取相应的条件
                box_cond = None
                if cond is not None:
                    if isinstance(cond, list) and batch_idx < len(cond):
                        box_cond = cond[batch_idx]
                    else:
                        box_cond = combined_pred
                    # 创建一个布尔掩码，指示哪些元素要保留
                    # 假设box_cond的长度与可能的mask值对应
                    keep_indices = []
                    for i in range(len(box_cond)):
                        # 检查索引+1是否在mask的唯一值中
                        if i in unique_values_in_mask:
                            keep_indices.append(i)
                    # 如果有要保留的索引，则过滤box_cond
                    if keep_indices:
                        # 使用索引选择器保留需要的元素
                        box_cond = box_cond[keep_indices]
                    else:
                        # 如果没有要保留的元素，创建一个空tensor保持原始维度结构
                        box_cond = torch.zeros((0,) + box_cond.shape[1:], device=box_cond.device, dtype=box_cond.dtype)
                
                
                # x_start = gt_features
                # t = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device, dtype=torch.long)
                # noise = default(None, lambda: torch.randn_like(x_start))
                # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                # #逆扩散过程
                # for _t in reversed(range(1, self.num_timesteps)):
                #     _t = torch.full((x_start.shape[0],), _t, device=x_start.device, dtype=torch.long)
                #     x_noisy = self.p_sample(x_noisy, box_cond, _t, upsam=False)
                # _t = 0
                # _t = torch.full((x_start.shape[0],), _t, device=x_start.device, dtype=torch.long)
                # model_out = self.p_sample(x_noisy, box_cond, _t, upsam=True)
                
                
                if self.if_normalize:
                    # x_start = self.norm_layer(gt_features)  # 当前box框的特征
                    # x_start = self.channel_balance_norm(x_start) 
                    x_start = self.normalize_to_minus1_1(gt_features[:, -1:, :, :])
                else:
                    x_start = gt_features[:, -1:, :, :]   
                batch_gt_x0.append(x_start)
                latent_shape = x_start.shape
                # 随机采样时间t
                t = torch.randint(0, self.num_timesteps, (latent_shape[0],), device=x_start.device).long()
                batch_t.append(t)
                noise = default(None, lambda: torch.randn_like(x_start))
                batch_gt_noise.append(noise)
                # noise sample
                # x = noise # 训练根据引导从纯噪声开始的引导能力
                x = self.q_sample(x_start = x_start, t = t, noise = noise)
                x_list.append(x)
                # predict and take gradient step
                model_out = self.denoiser(x, t, box_cond) #box_cond
                
                #----------------------------------------评测----------------------------
                if inf:
                    self.noisy_versions = []
                    self.gt_np = normalize_to_0_1(x_start).cpu().numpy()
                    #----------------------------------------评测----------------------------
                    # t = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device, dtype=torch.long)
                    # noise = default(None, lambda: torch.randn_like(x_start))
                    # x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                    # 逆扩散过程
                    # for _t in reversed(range(1, self.num_timesteps)):
                    #     _t = torch.full((x_start.shape[0],), _t, device=x_start.device, dtype=torch.long)
                    #     x_noisy = self.p_sample(x_noisy, box_cond, _t, upsam=False)
                    # _t = 0
                    # _t = torch.full((x_start.shape[0],), _t, device=x_start.device, dtype=torch.long)
                    # x_noisy = self.p_sample(x_noisy, box_cond, _t, upsam=True)

                    with torch.inference_mode():
                        t = torch.full((x_start.shape[0],), 250, device=x_start.device, dtype=torch.long)
                        noise = default(None, lambda: torch.randn_like(x_start))
                        noise = self.q_sample(x_start=x_start, t=t, noise=noise)
                        noise_list.append(noise)
                        
                        # x_noisy = self.diffusion_inference(noise, box_cond, self.sampling_timesteps)
                        x_noisy = self.ddim_sample(x.shape, x.device, box_cond,guidance_scale=self.guidance_scale)#x_start=noise,
                        x_noisy_with_cond_list.append(x_noisy)
                        x_noisy_init_result_np = normalize_to_0_1(x_noisy).cpu().numpy()
                        # fid_noisy_init_score = compute_frechet_distance(x_noisy_init_result_np, self.gt_np)
                        psnr_noisy_init_value = calculate_psnr(x_noisy_init_result_np, self.gt_np)
                        # ssim_noisy_init_value = calculate_ssim(x_noisy_init_result_np, self.gt_np)
                        self.noisy_versions.append({
                            'type': 'Pure Noise Init W Cond',
                            'tensor': x_noisy_init_result_np,
                            'psnr': psnr_noisy_init_value,
                            # 'ssim': ssim_noisy_init_value,
                            # 'fd': fid_noisy_init_score
                        })
                        
                        x_noisy_no_cond = self.ddim_sample(x.shape, x.device, None,guidance_scale=self.guidance_scale)#x_start=noise,
                        # x_noisy_no_cond = self.diffusion_inference(noise, None, self.sampling_timesteps)
                        x_noisy_no_cond_list.append(x_noisy_no_cond)
                        x_noisy_init_no_cond_result_np = normalize_to_0_1(x_noisy_no_cond).cpu().numpy()
                        # fid_noisy_init_no_cond_score = compute_frechet_distance(x_noisy_init_no_cond_result_np, self.gt_np)
                        psnr_noisy_init_no_cond_value = calculate_psnr(x_noisy_init_no_cond_result_np, self.gt_np)
                        # ssim_noisy_init_no_cond_value = calculate_ssim(x_noisy_init_no_cond_result_np, self.gt_np)
                        self.noisy_versions.append({
                            'type': 'Pure Noise Init No Cond',
                            'tensor': x_noisy_init_no_cond_result_np,
                            'psnr': psnr_noisy_init_no_cond_value,
                            # 'ssim': ssim_noisy_init_no_cond_value,
                            # 'fd': fid_noisy_init_no_cond_score
                        })
                        
                        noise_init = torch.full(x.shape, -0.35, device=x_start.device)
                        noise_init_list.append(noise_init)
                        noisy_init_np = normalize_to_0_1(noise_init).cpu().numpy()
                        # fid_noisy_score = compute_frechet_distance(noisy_init_np, self.gt_np)
                        psnr_noisy_value = calculate_psnr(noisy_init_np, self.gt_np)
                        # ssim_noisy_value = calculate_ssim(noisy_init_np, self.gt_np)
                        self.noisy_versions.append({
                            'type': 'Pure Noise Init -0.35',
                            'tensor': noise_init,
                            'psnr': psnr_noisy_value,
                            # 'ssim': ssim_noisy_value,
                            # 'fd': fid_noisy_score
                        })
                        loss = F.mse_loss(noise_init, x_start, reduction='mean')
                        print("noise loss：", loss.item())
                        
                        print("噪声类型\t\tPSNR (dB)\tSSIM\t\tFD")
                        print("-" * 70)
                        for result in self.noisy_versions:
                            # 调整类型字段的宽度，确保对齐
                            type_str = result['type'].ljust(20)
                            print(f"{type_str}\t{result['psnr']:.2f}\t\t")#\t\t{result['ssim']:.4f}
                    
                # 存储当前box框的预测结果
                batch_model_out.append(model_out)

            if inf:
                data_dict['pred_out_inf_with_cond'] = x_noisy_with_cond_list
                data_dict['pred_out_inf_no_cond'] = x_noisy_no_cond_list
                data_dict['noise_init'] =  noise_init_list
                data_dict['noise'] =  noise_list
                data_dict['x'] =  x_list
            
            data_dict['pred_out'] = batch_model_out
            data_dict['gt_noise'] = batch_gt_noise
            data_dict['gt_x0'] = batch_gt_x0
            data_dict['t'] = batch_t
            data_dict['target'] = self.parameterization
        else:
            # 对每个batch处理
            for batch_idx, gt_features in enumerate(batch_gt_spatial_features):
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]  # (batch_idx_voxel, 4)
                # 获取该batch中的box_mask
                this_box_mask = box_masks[batch_mask]  # (batch_idx_voxel, )
                unique_values_in_mask = torch.unique(this_box_mask)
                
                # 获取相应的条件
                box_cond = None
                if cond is not None:
                    if isinstance(cond, list) and batch_idx < len(cond):
                        box_cond = cond[batch_idx]
                    else:
                        box_cond = combined_pred
                    
                    # 创建一个布尔掩码，指示哪些元素要保留
                    keep_indices = []
                    for i in range(len(box_cond)):
                        # 检查索引是否在mask的唯一值中
                        if i in unique_values_in_mask:
                            keep_indices.append(i)
                    # 如果有要保留的索引，则过滤box_cond
                    if keep_indices:
                        # 使用索引选择器保留需要的元素
                        box_cond = box_cond[keep_indices]
                    else:
                        # 如果没有要保留的元素，创建一个空tensor保持原始维度结构
                        box_cond = torch.zeros((0,) + box_cond.shape[1:], device=box_cond.device, dtype=box_cond.dtype)
                
                # 可以尝试加个归一化
                x_start = gt_features  # 当前box框的特征
                latent_shape = x_start.shape
                noise = default(None, lambda: torch.randn_like(x_start))
                t = torch.full((x_start.shape[0],), self.num_timesteps-1, device=x_start.device, dtype=torch.long)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                
                # 逆扩散过程
                if self.train_mode == 1:
                    x_noisy = self.ddim_sample(x_noisy.shape, x_noisy.device, None)
                else:
                    x_noisy = self.ddim_sample(x_noisy.shape, x_noisy.device, box_cond, x_start=x_noisy, guidance_scale=self.guidance_scale)
                
                # 存储当前box框的预测结果
                batch_model_out.append(x_noisy)
            data_dict['pred_feature'] = batch_model_out
        
        return data_dict


def add_gaussian_noise(array, mean=0, std_scale=0.1):
    """添加高斯噪声，std_scale控制噪声强度"""
    noise = np.random.normal(loc=mean, scale=std_scale * np.abs(array).mean(), size=array.shape)
    noisy_array = array + noise
    return noisy_array

def generate_noisy_versions_and_calculate_metrics(data_norm):
    # 存储不同噪声版本及对应的PSNR和SSIM值
    results = []
    
    # 不同强度的高斯噪声
    for std in [0.05, 0.1, 0.2, 0.3]:
        noisy = add_gaussian_noise(data_norm, std_scale=std)
        noisy_norm = np.clip(noisy, 0, 1)  # 确保值在0-1范围内
        
        # 计算PSNR和SSIM
        psnr_val = calculate_psnr(noisy_norm, data_norm)
        ssim_val = calculate_ssim(noisy_norm, data_norm)
        
        noisy_fd = noisy_norm.reshape(1, *noisy_norm.shape) if len(noisy_norm.shape) == 3 else noisy_norm
        data_fd = data_norm.reshape(1, *data_norm.shape) if len(data_norm.shape) == 3 else data_norm
        fd_val = compute_frechet_distance(noisy_fd, data_fd)
        
        results.append({
            'type': f'Gaussian (std={std})',
            'tensor': noisy_norm,
            'psnr': psnr_val,
            'ssim': ssim_val,
            'fd': fd_val
        })
    
    return results


def compute_frechet_distance(array1, array2, channel_wise=True, eps=1e-6):
    """计算两个NumPy数组之间的Fréchet距离
    
    参数：
        array1: 形状为[B, C, H, W]的第一个NumPy数组
        array2: 形状为[B, C, H, W]的第二个NumPy数组
        channel_wise: 是否按通道计算距离，默认为True
        eps: 数值稳定性参数，默认为1e-6
    
    返回:
        frechet_distance: 计算得到的Fréchet距离
    """
    # 确保输入是NumPy数组
    if not isinstance(array1, np.ndarray):
        raise TypeError("array1必须是NumPy数组")
    if not isinstance(array2, np.ndarray):
        raise TypeError("array2必须是NumPy数组")
        
    # 检查输入是否为4D数组，如果是则按通道处理
    if len(array1.shape) == 4 and len(array2.shape) == 4 and channel_wise:
        total_dist = 0
        channels = array1.shape[1]  # 通道数
        
        for c in range(channels):
            # 提取第c个通道并展平为[B, H*W]
            channel1 = array1[:, c, :, :].reshape(array1.shape[0], -1)
            channel2 = array2[:, c, :, :].reshape(array2.shape[0], -1)
            
            # 计算均值向量
            mu1 = np.mean(channel1, axis=0)
            mu2 = np.mean(channel2, axis=0)
            
            # 计算协方差矩阵
            sigma1 = np.cov(channel1, rowvar=False)
            sigma2 = np.cov(channel2, rowvar=False)
            
            # 计算均值差的平方范数
            diff = mu1 - mu2
            mean_diff_squared = np.sum(diff * diff)
            
            # 数值稳定性
            sigma1 = sigma1 + np.eye(sigma1.shape[0]) * eps
            sigma2 = sigma2 + np.eye(sigma2.shape[0]) * eps
            
            # 计算协方差矩阵乘积的平方根
            try:
                covmean = linalg.sqrtm(sigma1.dot(sigma2))
                
                # 确保没有复数部分
                if np.iscomplexobj(covmean):
                    covmean = covmean.real
                
                # 计算trace项
                trace_term = np.trace(sigma1 + sigma2 - 2 * covmean)
                
                # 计算通道距离
                dist = mean_diff_squared + trace_term
                total_dist += dist
            except np.linalg.LinAlgError:
                # 处理可能的奇异矩阵错误
                print(f"警告: 通道{c}的协方差矩阵可能接近奇异，跳过该通道")
                if channels > 1:  # 如果有多个通道，忽略这一个
                    channels -= 1
                else:  # 如果只有一个通道，返回一个大值
                    return float('inf')
        
        # 返回平均距离
        return total_dist / max(channels, 1)  # 避免除以零

def normalize_to_0_1(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    if max_val - min_val > 0:  # 避免除以零
        return (tensor - min_val) / (max_val - min_val)
    else:
        return torch.zeros_like(tensor)  # 如果所有值都相同

def calculate_psnr(img1, img2, max_val=1):
    return psnr(img1, img2, data_range=max_val)

def calculate_ssim(img1, img2, max_val=1):
    return ssim(img1, img2, data_range=max_val)


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

