import importlib

import torch
from torch import optim
import torch.nn as nn
import numpy as np
import math
import scipy

from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont
from einops import repeat


def log_txt_as_img(wh, xc, size=10):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        font = ImageFont.truetype('data/DejaVuSans.ttf', size=size)
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def ismap(x):
    if not isinstance(x, torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] > 3)


def isimage(x):
    if not isinstance(x,torch.Tensor):
        return False
    return (len(x.shape) == 4) and (x.shape[1] == 3 or x.shape[1] == 1)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def mean_flat(tensor):
    """
    https://github.com/openai/guided-diffusion/blob/27c20a8fab9cb472df5d6bdd6c8d11c8f430b924/guided_diffusion/nn.py#L86
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


class AdamWwithEMAandWings(optim.Optimizer):
    # credit to https://gist.github.com/crowsonkb/65f7265353f403714fce3b2595e0b298
    def __init__(self, params, lr=1.e-3, betas=(0.9, 0.999), eps=1.e-8,  # TODO: check hyperparameters before using
                 weight_decay=1.e-2, amsgrad=False, ema_decay=0.9999,   # ema decay to match previous code
                 ema_power=1., param_names=()):
        """AdamW that saves EMA versions of the parameters."""
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= ema_decay <= 1.0:
            raise ValueError("Invalid ema_decay value: {}".format(ema_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, ema_decay=ema_decay,
                        ema_power=ema_power, param_names=param_names)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            ema_params_with_grad = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']
            ema_decay = group['ema_decay']
            ema_power = group['ema_power']

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of parameter values
                    state['param_exp_avg'] = p.detach().float().clone()

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                ema_params_with_grad.append(state['param_exp_avg'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                # update the steps for each param group update
                state['step'] += 1
                # record the step after step update
                state_steps.append(state['step'])

            optim._functional.adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'],
                    maximize=False)

            cur_ema_decay = min(ema_decay, 1 - state['step'] ** -ema_power)
            for param, ema_param in zip(params_with_grad, ema_params_with_grad):
                ema_param.mul_(cur_ema_decay).add_(param.float(), alpha=1 - cur_ema_decay)

        return loss

def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return nn.BatchNorm2d(channels)
    # return GroupNorm32(32, channels)

def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    if not repeat_only:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        embedding = repeat(timesteps, 'b -> b d', d=dim)
    return embedding

# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])

        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads

def detach(dict):
    for key, val in dict.items():
        val.detach()

# 评估diffusion重建效果，使用att_bev_backbone作为特征提取网络？？     
def calculate_fid(real_features, gen_features):
    # 计算均值和协方差
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)

    # 计算均值差的平方
    mean_diff_squared = np.sum((mu_real - mu_gen) ** 2)

    # 计算协方差项
    # 首先计算协方差矩阵的平方根乘积
    sqrt_product = scipy.linalg.sqrtm(sigma_real @ sigma_gen)

    # 确保结果是实数
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    # 完整的FID公式
    fid = mean_diff_squared + np.trace(sigma_real + sigma_gen - 2 * sqrt_product)

    return fid

def normalize_statistical_features(features):
    """
        使用语义一致的归一化，确保：
        1. 零值在所有通道都有一致的表示
        2. 归一化方式反映数据的实际意义
    """
    norm_features = features.clone()
    params = {}
    
     # 检查是否是单通道输入
    if features.shape[1] == 1:
        # 单通道情况 - 将其视为点数通道
        points = features[:, 0]  # 形状为 [B,H,W]
        zero_mask = (points == 0)
        
        # 点数映射：0 -> -1, max -> +1, 其他按比例
        if torch.max(points) > 0:
            log_points = torch.log1p(points)  # log(1+x)
            max_log = log_points.max()
            params['max_log_points'] = max_log.item()
            
            if max_log > 0:
                # 归一化到[-1, 1]，确保0点映射到-1
                normalized = -1.0 + 2.0 * log_points / max_log
                
                # 额外检查确保0点确实映射到-1
                normalized[zero_mask] = -1.0
                
                # 保存归一化结果
                norm_features[:, 0] = normalized
            else:
                # 如果log后最大值为0
                norm_features[:, 0] = torch.zeros_like(points)
                norm_features[:, 0][zero_mask] = -1.0
        else:
            # 所有点数都是0
            norm_features[:, 0] = torch.full_like(points, -1.0)
        
        return norm_features, params
    
    # === 几何特征（前3个通道）===
    for i in range(3):
        channel = features[:, i]
        
        # 找出0和非0值
        zero_mask = (channel == 0)
        non_zero = channel[~zero_mask]
        
        if len(non_zero) > 0:
            # 对非零值应用标准化
            min_val = non_zero.min()
            max_val = non_zero.max()
            params[f'min_{i}'] = min_val.item()
            params[f'max_{i}'] = max_val.item()
            
            # 如果有足够的数值范围，进行归一化
            if max_val - min_val > 1e-10:
                # 创建临时张量存储归一化结果
                normalized = torch.zeros_like(channel)
                
                # 只对非零值应用归一化，确保0映射到0（中间值）
                normalized[~zero_mask] = -1.0 + 2.0 * (non_zero - min_val) / (max_val - min_val)
                
                # 确保0值映射到特定值
                normalized[zero_mask] = -1
                
                norm_features[:, i] = normalized
            else:
                # 如果非零值范围很小，简化处理
                normalized = torch.zeros_like(channel)
                normalized[~zero_mask] = 0.5  # 非零值映射到一个适中的正值
                norm_features[:, i] = normalized
        else:
            # 全是0的情况
            norm_features[:, i] = torch.zeros_like(channel)
    
    # === 点数通道（第4个通道）===
    points = features[:, 3]
    zero_mask = (points == 0)
    
    # 点数映射：0 -> -1, max -> +1, 其他按比例
    if torch.max(points) > 0:
        log_points = torch.log1p(points)  # log(1+x)
        max_log = log_points.max()
        params['max_log_points'] = max_log.item()
        
        if max_log > 0:
            # 归一化到[-1, 1]，确保0点映射到-1
            norm_features[:, 3] = -1.0 + 2.0 * log_points / max_log
            
            # 额外检查确保0点确实映射到-1
            if not torch.all(norm_features[zero_mask, 3] == -1.0):
                norm_features[zero_mask, 3] = -1.0
    else:
        # 所有点数都是0
        norm_features[:, 3] = torch.full_like(points, -1.0)
    
    return norm_features, params

def denormalize_statistical_features(normalized_features, normalization_params):
    """
    将语义一致的归一化特征反归一化回原始范围
    
    输入:
        normalized_features [M, 4] - 归一化到[-1, 1]范围的特征
        normalization_params - 归一化时保存的参数字典
    
    输出:
        反归一化后的特征 [M, 4]
    """
    # 复制一份特征，避免修改原始数据
    denormalized_features = normalized_features.clone()
    
    # 检查是否是单通道输入
    if normalized_features.shape[1] == 1:
        # 单通道情况 - 将其视为点数通道
        points_normalized = normalized_features[:, 0]  # 形状为 [B,H,W]
        
        if 'max_log_points' in normalization_params:
            max_log = normalization_params['max_log_points']
            
            # 找出映射到-1的零值（原始点数为0的位置）
            zero_points_mask = (points_normalized <= -0.99)  # 允许一点数值误差
            
            # 对非零点数进行反归一化
            if not torch.all(zero_points_mask):
                # 将[-1,1]映射回[0,max_log]的log空间
                log_points = ((points_normalized + 1.0) / 2.0) * max_log
                
                # 从log空间映射回原始点数
                denorm_points = torch.expm1(log_points)  # exp(x)-1是log1p的逆运算
                
                # 保存反归一化结果
                denormalized_features[:, 0] = denorm_points
                
                # 确保原始零点数被准确恢复
                denormalized_features[:, 0][zero_points_mask] = 0.0
            else:
                # 所有值都是零点
                denormalized_features[:, 0] = torch.zeros_like(points_normalized)
        else:
            # 如果没有参数，将所有-1值视为0点数
            denormalized_features[:, 0] = torch.zeros_like(points_normalized)
            denormalized_features[:, 0][points_normalized > -0.99] = 1.0  # 非-1值设为至少1个点
        
        return denormalized_features
    
    # === 几何特征通道（前3个通道）===
    for i in range(3):
        channel = normalized_features[:, i]
        
        # 识别不同的值区域
        zero_value_mask = (channel == -1.0)  # 被映射到-1.0的原始零值
        
        # 检查是否有归一化参数
        if f'min_{i}' in normalization_params and f'max_{i}' in normalization_params:
            min_val = normalization_params[f'min_{i}']
            max_val = normalization_params[f'max_{i}']
            
            # 检查是否有足够的数值范围
            if max_val - min_val > 1e-10:
                # 创建临时张量存储反归一化结果
                denorm_channel = torch.zeros_like(channel)
                
                # 将表示原始零值的-1.0映射回0
                denorm_channel[zero_value_mask] = 0.0
                
                # 将非零值（现在在[-1,1]之外的值）映射回原始范围
                non_zero_mask = ~zero_value_mask
                if non_zero_mask.any():
                    # 从[-1,1]反归一化到原始范围
                    denorm_channel[non_zero_mask] = min_val + (
                        (channel[non_zero_mask] + 1.0) / 2.0) * (max_val - min_val)
                
                denormalized_features[:, i] = denorm_channel
            else:
                # 如果原始数据范围很小
                denormalized_features[:, i] = torch.zeros_like(channel)
                denormalized_features[:, i][~zero_value_mask] = min_val
        else:
            # 如果没有提供参数，保持0值
            denormalized_features[:, i] = torch.zeros_like(channel)
    
    # === 点数通道（第4个通道）===
    if 'max_log_points' in normalization_params:
        max_log = normalization_params['max_log_points']
        
        # 从[-1,1]映射回原始点数
        points_normalized = normalized_features[:, 3]
        
        # 找出映射到-1的零值（原始点数为0的位置）
        zero_points_mask = (points_normalized <= -0.99)  # 允许一点数值误差
        
        # 对非零点数进行反归一化
        if not torch.all(zero_points_mask):
            # 将[-1,1]映射回[0,max_log]的log空间
            log_points = ((points_normalized + 1.0) / 2.0) * max_log
            
            # 从log空间映射回原始点数
            denormalized_features[:, 3] = torch.expm1(log_points)  # exp(x)-1是log1p的逆运算
        
        # 确保原始零点数被准确恢复
        denormalized_features[:, 3][zero_points_mask] = 0.0
    else:
        # 如果没有参数，将所有-1值视为0点数
        denormalized_features[:, 3] = torch.zeros_like(normalized_features[:, 3])
        denormalized_features[:, 3][normalized_features[:, 3] > -0.99] = 1.0  # 非-1值设为至少1个点
    
    return denormalized_features