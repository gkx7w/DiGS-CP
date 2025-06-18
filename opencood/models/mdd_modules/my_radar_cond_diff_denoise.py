import os
from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import torch
from diffusers import UNet2DModel
import torch.nn.functional as F
from diffusers import DDPMScheduler,DDIMScheduler
from matplotlib import pyplot as plt
import torchvision
import torch.nn as nn
import numpy as np
from opencood.models.mdd_modules.unet import DiffusionUNet
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from opencood.utils.MDD_utils import normalize_statistical_features,denormalize_statistical_features
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.cuda.amp import autocast

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
        self.config = Config(model_cfg)
        self.parameterization = getattr(self.config.diffusion, 'parameterization', 'x0')
        self.if_normalize = getattr(self.config.diffusion, 'if_normalize', False)
        self.guidance_scale = getattr(self.config.diffusion, 'guidance_scale', 1.0)  # 默认值为1.0
        self.ddim_sampling_eta = getattr(self.config.diffusion, 'ddim_sampling_eta', 0.0)  # 默认值为0.0
        self.sampling_timesteps = getattr(self.config.diffusion, 'sampling_timesteps', 3) # default num sampling timesteps to number of timesteps at training
        if self.parameterization == "eps":
            self.prediction_type = "epsilon"
        elif self.parameterization == "x0":
            self.prediction_type = "sample"
        elif self.parameterization == "v":
            self.prediction_type = "v_prediction"
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000,prediction_type = self.prediction_type) #sample epsilon v_prediction
        self.unet = DiffusionUNet(self.config)
        
        self.condition_encoder = nn.Sequential(
            nn.GroupNorm(32, 512),        # 32个组，适合512维
            nn.Linear(512, 256),
            nn.GELU(),
            nn.GroupNorm(16, 256),        # 16个组，适合256维
        )

    def normalize_to_minus1_1(self, x):
        return x * 2 - 1
    
    def evaluate(self, pipeline,milestone=None):
        # Sample some images from random noise (this is the backward diffusion process).
        # The default pipeline output type is `List[PIL.Image]`
        images = pipeline(
            batch_size=self.config.eval_batch_size,
            generator=torch.Generator(device='cpu').manual_seed(self.config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
        ).images

        # Make a grid out of the images
        image_grid = make_image_grid(images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        if milestone is None:
            milestone = "sample"
        image_grid.save(f"{test_dir}/{milestone}.png")
        
    def ddim_sample(
        self,
        batch_size=1,
        image_shape=None,
        generator=None,
        num_inference_steps=50,
        eta=0.0,
        use_clipped_model_output=False,
        output_type="pil",
        return_dict=True,
        guidance_scale=1.0,
        x_self_cond=None,    
        initial_noise=None,  # 新增参数：自定义初始噪声
    ):
        """
        使用DDIM算法从噪声中生成样本
        """
        # 获取设备
        device = next(self.unet.parameters()).device
        
        # 使用自定义噪声或生成新噪声
        if initial_noise is not None:
            image = initial_noise.to(device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=device)
        
        
        # 设置时间步
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        
        # 条件引导控制
        do_classifier_free_guidance = guidance_scale > 1.0
        
        # 迭代时间步
        # for t in self.progress_bar(self.ddim_scheduler.timesteps):
        for t in self.ddim_scheduler.timesteps:
            # 创建批次时间步张量
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 预测噪声
            if do_classifier_free_guidance:
                # 无条件分支
                noise_pred_uncond = self.unet(image, timestep, None)
                
                # 有条件分支
                noise_pred_cond = self.unet(image, timestep, x_self_cond)
                
                # 执行分类器自由引导
                model_output = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 直接使用模型预测
                model_output = self.unet(image, timestep, x_self_cond)
            
            # 使用调度器的step方法进行更新
            image = self.ddim_scheduler.step(
                model_output, t, image, generator=generator).prev_sample #use_clipped_model_output=use_clipped_model_output, eta=eta,
            
            # # DDIM采样步骤
            # image = self.ddim_scheduler.step(
            #     model_output, t, image, eta=eta, use_clipped_model_output=use_clipped_model_output, generator=generator
            # ).prev_sample
        # 后处理
        # image = (image / 2 + 0.5).clamp(0, 1)
        image = denormalize_statistical_features(image, self.norm_params)
        # image = image.cpu().permute(0, 2, 3, 1)
        
        if output_type == "pil":
            image = self.numpy_to_pil(image)
        
        if not return_dict:
            return (image,)
        
        return {"images": image}

    def numpy_to_pil(self, images):
        """将numpy数组转换为PIL图像"""
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    def progress_bar(self, iterable):
        """简单进度条"""
        try:
            from tqdm.auto import tqdm
            return tqdm(iterable)
        except ImportError:
            return iterable
    
    
    def generate_comparison_samples(
        self,
        batch_size=16,
        num_inference_steps=1000,
        eta=0.0,
        guidance_scale=7.5,  # 有条件推理的引导尺度
        x_self_cond=None,    # 条件输入
        output_dir="samples",
        base_filename="sample",
        output_type="pil",
    ):
        """
        生成有条件和无条件的图像样本并保存比较
        
        参数:
            batch_size: 生成图像的批次大小
            num_inference_steps: 推理步数
            eta: DDIM中的eta参数
            guidance_scale: 有条件推理的引导尺度 (>1.0)
            x_self_cond: 条件输入
            output_dir: 保存图像的目录
            base_filename: 基础文件名
            output_type: 输出类型 ('pil' 或 'numpy')
        
        返回:
            (条件图像, 无条件图像) 的元组
        """
        import os
        from PIL import Image
        import numpy as np
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 确保使用相同的随机种子以便比较
        seed = np.random.randint(0, 2147483647)
        generator = torch.Generator(device=next(self.unet.parameters()).device).manual_seed(seed)
        
        # 生成有条件图像 (guidance_scale > 1.0)
        print(f"生成有条件图像 (guidance_scale={guidance_scale})...")
        cond_result = self.ddim_sample(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps,
            eta=eta,
            guidance_scale=guidance_scale,  # > 1.0 启用CFG
            x_self_cond=x_self_cond,
            output_type=output_type,
        )
        cond_images = cond_result["images"] if isinstance(cond_result, dict) else cond_result[0]
        
        # 保存有条件图像
        if output_type == "pil":
            # 创建图像网格
            grid_size = int(np.ceil(np.sqrt(batch_size)))
            grid_width = grid_size * cond_images[0].width
            grid_height = grid_size * cond_images[0].height
            grid_image = Image.new('RGB', (grid_width, grid_height))
            
            # 填充网格
            for i, img in enumerate(cond_images):
                row = i // grid_size
                col = i % grid_size
                grid_image.paste(img, (col * img.width, row * img.height))
            
            # 保存网格
            grid_image.save(os.path.join(output_dir, f"{base_filename}_with_cond.png"))
        else:
            # 如果是numpy数组，转换为PIL然后保存
            for i, img_array in enumerate(cond_images):
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                img.save(os.path.join(output_dir, f"{base_filename}_with_cond_{i}.png"))
        
        # 重置生成器以使用相同的初始噪声
        generator = torch.Generator(device=next(self.unet.parameters()).device).manual_seed(seed)
        
        # 生成无条件图像 (guidance_scale = 1.0)
        print("生成无条件图像 (guidance_scale=1.0)...")
        uncond_result = self.ddim_sample(
            batch_size=batch_size,
            generator=generator,
            num_inference_steps=num_inference_steps,
            eta=eta,
            guidance_scale=1.0,  # 设为1.0禁用CFG
            x_self_cond=None,    # 设为None以跳过条件应用
            output_type=output_type,
        )
        uncond_images = uncond_result["images"] if isinstance(uncond_result, dict) else uncond_result[0]
        
        # 保存无条件图像
        if output_type == "pil":
            # 创建图像网格
            grid_size = int(np.ceil(np.sqrt(batch_size)))
            grid_width = grid_size * uncond_images[0].width
            grid_height = grid_size * uncond_images[0].height
            grid_image = Image.new('RGB', (grid_width, grid_height))
            
            # 填充网格
            for i, img in enumerate(uncond_images):
                row = i // grid_size
                col = i % grid_size
                grid_image.paste(img, (col * img.width, row * img.height))
            
            # 保存网格
            grid_image.save(os.path.join(output_dir, f"{base_filename}_no_cond.png"))
        else:
            # 如果是numpy数组，转换为PIL然后保存
            for i, img_array in enumerate(uncond_images):
                img = Image.fromarray((img_array * 255).astype(np.uint8))
                img.save(os.path.join(output_dir, f"{base_filename}_no_cond_{i}.png"))
        
        print(f"生成完成。图像已保存至 {output_dir} 目录")
        
        # 如果你想并排查看两个图像的比较，可以创建一个比较图
        if output_type == "pil" and batch_size > 0:
            # 获取第一个图像的尺寸
            width, height = cond_images[0].width, cond_images[0].height
            
            # 为每对图像创建比较图
            for i in range(min(batch_size, len(cond_images), len(uncond_images))):
                comparison = Image.new('RGB', (width * 2, height))
                comparison.paste(uncond_images[i], (0, 0))
                comparison.paste(cond_images[i], (width, 0))
                comparison.save(os.path.join(output_dir, f"{base_filename}_comparison_{i}.png"))
            
            print(f"已生成比较图像 {base_filename}_comparison_*.png")
        
        return cond_images, uncond_images
    
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
        batch_norm_gt_x0 = []
        batch_t = []
        
        #----------------------------------------评测----------------------------
        # inf = True
        inf = False
    
        x_noisy_with_cond_list = []
        x_noisy_with_cond_nonorm_list = []
        x_noisy_no_cond_list = []
        x_noisy_no_cond_nonorm_list = []
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
            
                if self.if_normalize:
                    # x_start = self.normalize_to_minus1_1(gt_features) #[:,-1:,:,:]
                    x_start, self.norm_params = normalize_statistical_features(gt_features[:,-1:,:,:]) #[:,-1:,:,:]
                else:
                    x_start = gt_features[:,-1:,:,:]#[:,-1:,:,:]
                batch_gt_x0.append(gt_features[:,-1:,:,:])#[:,-1:,:,:]
                batch_norm_gt_x0.append(x_start)
                latent_shape = x_start.shape
                
                noise = torch.randn(x_start.shape, device=x_start.device)
                batch_gt_noise.append(noise)
                bs = x_start.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=x_start.device,
                    dtype=torch.int64
                )
                batch_t.append(timesteps)

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = self.noise_scheduler.add_noise(x_start, noise, timesteps)

                
                box_cond = self.condition_encoder(box_cond)
                cond_drop_prob = 0.2 #.2
                # 随机决定是否丢弃条件
                keep_cond = torch.rand(1).item() >= cond_drop_prob
                x_self_cond = box_cond if keep_cond else None
                
                # Predict the noise residual
                # noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                with autocast(enabled=True): 
                    model_out = self.unet(noisy_images, timesteps, x_self_cond).float()
            
                #----------------------------------------评测----------------------------
                if inf:
                    self.noisy_versions = []
                    self.gt_np = normalize_to_0_1(x_start).cpu().numpy()
                    #----------------------------------------评测----------------------------
                    with torch.inference_mode():
                        t = torch.full((x_start.shape[0],), 100, device=x_start.device, dtype=torch.long)
                        noise = torch.randn(x_start.shape, device=x_start.device)
                        noise = self.noise_scheduler.add_noise(x_start, noise, t)#[:,-1:,:,:]
                        noise_list.append(noise)
                        
                        # x_noisy = self.diffusion_inference(noise, box_cond, self.sampling_timesteps)
                        # self.ddim_scheduler = DDIMScheduler.from_config(self.noise_scheduler.config)
                        self.ddim_scheduler = self.noise_scheduler
                        seed = np.random.randint(0, 2147483647)
                        generator = torch.Generator(device=next(self.unet.parameters()).device).manual_seed(seed)
                        generator_nocond = torch.Generator(device=next(self.unet.parameters()).device).manual_seed(seed)
                        # 生成有条件图像 (guidance_scale > 1.0)
                        print(f"生成有条件图像 (guidance_scale={self.guidance_scale})...")
                        x_noisy = self.ddim_sample(
                            batch_size=bs,
                            image_shape=latent_shape,
                            generator=generator,
                            num_inference_steps=self.sampling_timesteps,
                            eta=0.0,
                            guidance_scale=self.guidance_scale,  # > 1.0 启用CFG
                            x_self_cond=x_self_cond,
                            output_type="numpy",
                            initial_noise=noise,  # 传入自定义噪声
                        )
                        x_noisy = x_noisy["images"] if isinstance(x_noisy, dict) else x_noisy[0]
                        x_noisy_with_cond_list.append(x_noisy)
                        # x_noisy_with_cond_nonorm_list.append(x_noist_nononrm)
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
                        
                        print("生成无条件图像 (guidance_scale=1.0)...")
                        uncond_result = self.ddim_sample(
                            batch_size=bs,
                            image_shape=latent_shape,
                            generator=generator_nocond,
                            num_inference_steps=self.sampling_timesteps,
                            eta=0.0,
                            guidance_scale=1.0,  # 设为1.0禁用CFG
                            x_self_cond=None,    # 设为None以跳过条件应用
                            output_type="numpy",
                            initial_noise=noise,  # 传入自定义噪声
                        )
                        x_noisy_no_cond = uncond_result["images"] if isinstance(uncond_result, dict) else uncond_result[0]
                        # x_noisy_no_cond = self.diffusion_inference(noise, None, self.sampling_timesteps)
                        x_noisy_no_cond_list.append(x_noisy_no_cond)
                        # x_noisy_no_cond_nonorm_list.append(x_noisy_no_cond_nonorm)
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
                        
                        noise_init = torch.full(noisy_images.shape, -0.35, device=x_start.device)
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
                data_dict['pred_out_inf_with_cond_nonorm'] = x_noisy_with_cond_nonorm_list
                data_dict['pred_out_inf_no_cond'] = x_noisy_no_cond_list
                data_dict['pred_out_inf_no_cond_nonorm'] = x_noisy_no_cond_nonorm_list
                data_dict['noise_init'] =  noise_init_list
                data_dict['noise'] =  noise_list
                data_dict['x'] =  x_list
            
            data_dict['pred_out'] = batch_model_out
            data_dict['gt_noise'] = batch_gt_noise
            data_dict['gt_x0'] = batch_gt_x0
            data_dict['norm_gt_x0'] = batch_norm_gt_x0
            data_dict['t'] = batch_t
            data_dict['target'] = self.parameterization
           
        return data_dict


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
                covmean = np.linalg.sqrtm(sigma1.dot(sigma2))
                
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