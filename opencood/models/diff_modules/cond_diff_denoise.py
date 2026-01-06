import torch
from diffusers import DDPMScheduler,DDIMScheduler
import torch.nn as nn
from opencood.models.diff_modules.unet import DiffusionUNet
from diffusers.utils.torch_utils import randn_tensor
from PIL import Image
from opencood.utils.MDD_utils import normalize_statistical_features,denormalize_statistical_features
from torch.cuda.amp import autocast

def numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


class Config:
	def __init__(self, entries: dict={}):
		for k, v in entries.items():
			if isinstance(v, dict):
				self.__dict__[k] = Config(v)
			else:
				self.__dict__[k] = v
    
class Cond_Diff_Denoise(nn.Module):
    def __init__(self,  model_cfg):
        super().__init__()
        self.config = Config(model_cfg)
        self.parameterization = getattr(self.config.diffusion, 'parameterization', 'x0')
        self.if_normalize = getattr(self.config.diffusion, 'if_normalize', False)
        self.guidance_scale = getattr(self.config.diffusion, 'guidance_scale', 1.0)  
        self.ddim_sampling_eta = getattr(self.config.diffusion, 'ddim_sampling_eta', 0.0)  
        self.sampling_timesteps = getattr(self.config.diffusion, 'sampling_timesteps', 3) 
        self.traing = getattr(self.config.diffusion, 'training', True) 
        self.beta_schedule = getattr(self.config.diffusion, 'beta_schedule', 'linear')  
        if self.parameterization == "eps":
            self.prediction_type = "epsilon"
        elif self.parameterization == "x0":
            self.prediction_type = "sample"
        elif self.parameterization == "v":
            self.prediction_type = "v_prediction"
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule = self.beta_schedule,prediction_type = self.prediction_type) #sample epsilon v_prediction
        self.unet = DiffusionUNet(self.config)
        
        self.condition_adapter = nn.Sequential(
            nn.GroupNorm(32, 512),       
            nn.Linear(512, 256),
            nn.GELU(),
            nn.GroupNorm(16, 256),       
        )
    
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
        initial_noise=None,  
    ):

        device = next(self.unet.parameters()).device
        
        if initial_noise is not None:
            image = initial_noise.to(device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=device)
        
        self.ddim_scheduler.set_timesteps(num_inference_steps)
        do_classifier_free_guidance = guidance_scale > 1.0
        
        for t in self.ddim_scheduler.timesteps:
            timestep = torch.full((batch_size,), t, device=device, dtype=torch.long)
            if do_classifier_free_guidance:
                noise_pred_uncond = self.unet(image, timestep, None)
                noise_pred_cond = self.unet(image, timestep, x_self_cond)
                model_output = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                model_output = self.unet(image, timestep, x_self_cond)
            
            image = self.ddim_scheduler.step(
                model_output, t, image, generator=generator).prev_sample 

        # image = (image / 2 + 0.5).clamp(0, 1)
        image = denormalize_statistical_features(image, self.norm_params)
        # image = image.cpu().permute(0, 2, 3, 1)
        
        if output_type == "pil":
            image = numpy_to_pil(image)
        
        if not return_dict:
            return (image,)
        
        return {"images": image}


    def forward(self, data_dict):
        # Get features for each bounding box in the batch
        batch_gt_spatial_features = data_dict['batch_gt_spatial_features']
        coords = data_dict['voxel_coords']
        box_masks = data_dict['voxel_gt_mask']
        # Using fused features as condition
        cond = data_dict.get('fused_object_factors', None)
        if cond is not None and len(cond) == 0:
            cond = None
        combined_pred = cond

        batch_model_out = []
        batch_gt_noise = []
        batch_gt_x0 = []
        batch_norm_gt_x0 = [] 
        
        if self.traing:
            for batch_idx, gt_features in enumerate(batch_gt_spatial_features):
                batch_mask = coords[:, 0] == batch_idx
                this_coords = coords[batch_mask, :]  # (batch_idx_voxel, 4)
                this_box_mask = box_masks[batch_mask]  # (batch_idx_voxel, )
                unique_values_in_mask = torch.unique(this_box_mask)
                # Get condition
                box_cond = None
                if cond is not None:
                    if isinstance(cond, list) and batch_idx < len(cond):
                        box_cond = cond[batch_idx]
                    else:
                        box_cond = combined_pred
                    keep_indices = []
                    for i in range(len(box_cond)):
                        if i in unique_values_in_mask:
                            keep_indices.append(i)
                    if keep_indices:
                        box_cond = box_cond[keep_indices]
                    else:
                        box_cond = torch.zeros((0,) + box_cond.shape[1:], device=box_cond.device, dtype=box_cond.dtype)
            
                if self.if_normalize:
                    x_start, self.norm_params = normalize_statistical_features(gt_features) 
                else:
                    x_start = gt_features
                batch_gt_x0.append(gt_features)
                batch_norm_gt_x0.append(x_start)
                
                noise = torch.randn(x_start.shape, device=x_start.device)
                batch_gt_noise.append(noise)
                bs = x_start.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=x_start.device,
                    dtype=torch.int64
                )

                # Add noise to the clean images according to the noise magnitude at each timestep
                noisy_images = self.noise_scheduler.add_noise(x_start, noise, timesteps)

                box_cond = self.condition_adapter(box_cond)
                cond_drop_prob = 0.2 
                keep_cond = torch.rand(1).item() >= cond_drop_prob
                x_self_cond = box_cond if keep_cond else None
                
                # Predict the noise residual
                with autocast(enabled=True): 
                    model_out = self.unet(noisy_images, timesteps, x_self_cond).float()
                batch_model_out.append(model_out)
         
            data_dict['pred_out'] = batch_model_out
            data_dict['gt_noise'] = batch_gt_noise
            data_dict['gt_x0'] = batch_gt_x0
            data_dict['norm_gt_x0'] = batch_norm_gt_x0
            data_dict['target'] = self.parameterization
           
        return data_dict
        