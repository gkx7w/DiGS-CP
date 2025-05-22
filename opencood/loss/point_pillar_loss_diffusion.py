import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import reduce
from opencood.loss.ciassd_loss import CiassdLoss, weighted_smooth_l1_loss


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

def weighted_sigmoid_binary_cross_entropy(preds, tgts, weights=None,
                                          class_indices=None):
    if weights is not None:
        weights = weights.unsqueeze(-1)
    if class_indices is not None:
        weights *= (
            indices_to_dense_vector(class_indices, preds.shape[2])
                .view(1, 1, -1)
                .type_as(preds)
        )
    per_entry_cross_ent = nn.functional.binary_cross_entropy_with_logits(preds,
                                                                         tgts,
                                                                         weights)
    return per_entry_cross_ent


def indices_to_dense_vector(
        indices, size, indices_value=1.0, default_value=0, dtype=np.float32
):
    """Creates dense vector with indices set to specific value and rest to zeros.
    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
    Args:
        indices: 1d Tensor with integer indices which are to be set to
            indices_values.
        size: scalar with size (integer) of output Tensor.
        indices_value: values of elements specified by indices in the output vector
        default_value: values of other elements in the output vector.
        dtype: data type.
    Returns:
        dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense

class PointPillarLossDiffusion(nn.Module):
    def __init__(self, args):
        super(PointPillarLossDiffusion, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0
        self.sigma_data = 0.1
        self.num_timesteps = 1000 # 与mdd中的总时间步长对应
        self.total_loss = 0
        self.diff_loss = 0
        self.rcnn_loss = 0
        self.cls_loss = 0
        self.reg_loss = 0
        self.iou_loss = 0
        self.cls = args['stage2']['cls']
        self.reg = args['stage2']['reg']
        self.iou = args['stage2']['iou']
        self.use_focal_cls = True  # 使用Focal Loss
        self.adaptive_cls_weight = True  # 自适应分类权重
        
        # 回归损失优化选项  
        self.use_iou_guided_reg = True  # IoU引导的回归损失
        
        # IoU损失优化选项
        self.use_iou_loss = "adaptive"  # 或 "focal_iou"
        
        # 动态权重调整
        self.dynamic_loss_weights = True
        self.current_epoch = 0
        self.total_epochs = 100
        # 三种类型：naive,naive+,naive+W3
        self.use_iou_loss = "naive"
        # 两种类型：True,Flase
        self.use_reg_loss_focal = True
        self.loss_dict = {}

    
    def sigmoid_weight(self, max_weight, epoch):
        return max_weight / 2 * (- (np.tanh(epoch / 4 - 5)) + 1)


    def stage2_loss(self, rcnn_cls, rcnn_iou, rcnn_reg, tgt_cls, tgt_iou, tgt_reg):
        rcnn_cls = rcnn_cls.view(1, -1, 1)
        rcnn_iou = rcnn_iou.view(1, -1, 1)
        rcnn_reg = rcnn_reg.view(1, -1, 7)

        tgt_cls = tgt_cls.view(1, -1, 1)
        tgt_iou = tgt_iou.view(1, -1, 1)
        tgt_reg = tgt_reg.view(1, -1, 7)

        pos_norm = tgt_cls.sum()
        
        # ========== 分类损失优化 ==========
        # 1. 使用Focal Loss替代简单的BCE
        if hasattr(self, 'use_focal_cls') and self.use_focal_cls:
            loss_cls = self.focal_cls_loss(rcnn_cls, tgt_cls)
        else:
            # 2. 改进的加权BCE，考虑正负样本平衡
            pos_weight = max(1.0, (tgt_cls == 0).sum().float() / max(pos_norm.float(), 1.0))
            loss_cls = F.binary_cross_entropy_with_logits(
                rcnn_cls, tgt_cls, 
                pos_weight=pos_weight,
                reduction='mean'
            )
        
        # 3. 添加分类损失的自适应权重
        if hasattr(self, 'adaptive_cls_weight') and self.adaptive_cls_weight:
            # 根据分类准确率动态调整权重
            with torch.no_grad():
                pred_cls = torch.sigmoid(rcnn_cls) > 0.5
                cls_acc = (pred_cls == tgt_cls).float().mean()
                # 准确率低时增加权重
                adaptive_weight = 2.0 - cls_acc.item()  
                loss_cls = loss_cls * adaptive_weight

        # ========== 回归损失优化 ==========
        weights = torch.ones(tgt_iou.shape, device=tgt_iou.device)
        weights[tgt_cls == 0] = 0
        
        # 4. 改进的focal回归损失策略
        if self.use_reg_loss_focal:
            neg = torch.logical_and(tgt_iou < 0.7, tgt_cls != 0)
            pos = torch.logical_and(tgt_iou >= 0.7, tgt_cls != 0)
            
            num_neg = int(neg.sum(dim=1))
            num_pos = int(pos.sum(dim=1))
            
            # 5. 动态调整正负样本比例
            if num_pos > 0 and num_neg > 0:
                # 根据总体IoU分布调整采样策略
                avg_iou = tgt_iou[tgt_cls != 0].mean()
                if avg_iou < 0.6:  # IoU较低时，保留更多负样本学习
                    num_pos_samples = max(num_neg, 10)
                else:  # IoU较高时，平衡采样
                    num_pos_samples = max(num_neg // 2, 5)
                
                if num_pos > num_pos_samples:
                    pos_indices = torch.where(pos)[1]
                    not_selected = torch.randperm(num_pos)[:num_pos - num_pos_samples]
                    weights[:, pos_indices[not_selected]] = 0
        
        # 6. 使用更稳定的回归损失
        if hasattr(self, 'use_iou_guided_reg') and self.use_iou_guided_reg:
            # IoU引导的回归损失 - 高IoU样本权重更大
            iou_weights = torch.clamp(tgt_iou, 0.1, 1.0)  # 避免权重为0
            weights = weights * iou_weights
        
        loss_reg = weighted_smooth_l1_loss(
            rcnn_reg, tgt_reg,
            weights=weights / max(weights.sum(), 1)
        ).sum()

        # ========== IoU损失优化 ==========
        weights_iou = torch.ones(tgt_iou.shape, device=tgt_iou.device)
        tgt_iou_normalized = 2 * (tgt_iou - 0.5)  # normalize to -1, 1
        
        # 7. 改进的IoU损失策略
        if self.use_iou_loss == "adaptive":
            # 自适应IoU损失权重
            with torch.no_grad():
                iou_error = torch.abs(rcnn_iou - tgt_iou_normalized)
                hard_samples = iou_error > iou_error.median()
                weights_iou[hard_samples] = 2.0  # 难样本加权
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou_normalized, weights=weights_iou).mean()
        
        elif self.use_iou_loss == "focal_iou":
            # Focal IoU Loss - 关注难预测的IoU
            iou_error = torch.abs(rcnn_iou - tgt_iou_normalized)
            focal_weight = torch.pow(iou_error + 1e-8, 2.0)  # gamma=2
            weights_iou = weights_iou * focal_weight
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou_normalized, weights=weights_iou).mean()
        
        elif self.use_iou_loss == "naive+W3":
            weights_iou[tgt_iou_normalized < -0.2] = 3  # 对应原来的0.4
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou_normalized, weights=weights_iou).mean()
        
        else:
            # 默认策略，但加入正样本权重
            weights_iou = tgt_cls.float()  # 只计算正样本的IoU损失
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou_normalized, weights=weights_iou).mean()

        # ========== 动态损失权重调整 ==========
        # 8. 根据训练阶段动态调整各损失权重
        if hasattr(self, 'dynamic_loss_weights') and self.dynamic_loss_weights:
            # 训练初期更关注分类，后期更关注回归精度
            epoch_ratio = getattr(self, 'current_epoch', 0) / getattr(self, 'total_epochs', 100)
            
            cls_weight_factor = 1.5 - 0.5 * epoch_ratio  # 1.5 -> 1.0
            reg_weight_factor = 0.5 + 0.5 * epoch_ratio  # 0.5 -> 1.0
            iou_weight_factor = 0.8 + 0.4 * epoch_ratio  # 0.8 -> 1.2
        else:
            cls_weight_factor = reg_weight_factor = iou_weight_factor = 1.0

        # 应用权重
        loss_cls_reduced = loss_cls * self.cls['weight'] * cls_weight_factor
        loss_iou_reduced = loss_iou * self.iou['weight'] * iou_weight_factor
        loss_reg_reduced = loss_reg * self.reg['weight'] * reg_weight_factor
        
        return loss_cls_reduced, loss_iou_reduced, loss_reg_reduced

    # ========== 辅助函数 ==========
    def focal_cls_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal Loss for classification"""
        pred_sigmoid = torch.sigmoid(pred)
        pt = target * pred_sigmoid + (1 - target) * (1 - pred_sigmoid)
        
        # 防止数值不稳定
        pt = torch.clamp(pt, 1e-8, 1.0 - 1e-8)
        
        alpha_weight = target * alpha + (1 - target) * (1 - alpha)
        focal_weight = alpha_weight * torch.pow(1 - pt, gamma)
        
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

    # def stage2_loss(self, rcnn_cls,rcnn_iou,rcnn_reg,tgt_cls,tgt_iou,tgt_reg):
    #     rcnn_cls = rcnn_cls.view(1, -1, 1)
    #     rcnn_iou = rcnn_iou.view(1, -1, 1)
    #     rcnn_reg = rcnn_reg.view(1, -1, 7)

    #     tgt_cls = tgt_cls.view(1, -1, 1)
    #     tgt_iou = tgt_iou.view(1, -1, 1)
    #     tgt_reg = tgt_reg.view(1, -1, 7)

    #     pos_norm = tgt_cls.sum()
    #     # cls loss
    #     loss_cls = weighted_sigmoid_binary_cross_entropy(rcnn_cls, tgt_cls)

    #     # regression loss
    #     # [deprecated by Yifan Lu] Target resampling : Generate a weights mask to force the regressor concentrate on low iou predictions
    #     # sample 50% with iou>0.7 and 50% < 0.7
    #     weights = torch.ones(tgt_iou.shape, device=tgt_iou.device)
    #     weights[tgt_cls == 0] = 0
    #     # coalign 这边是直接设置为0，没有后续过程
    #     if self.use_reg_loss_focal:
    #         neg = torch.logical_and(tgt_iou < 0.7, tgt_cls != 0)
    #         pos = torch.logical_and(tgt_iou >= 0.7, tgt_cls != 0)
    #         # # 这里不仅仅删除一些正样本，并且对于负样本加量学习！先来个五倍吧
    #         num_neg = int(neg.sum(dim=1))
    #         num_pos = int(pos.sum(dim=1))
    #         num_pos_smps = max(num_neg, 5)
    #         pos_indices = torch.where(pos)[1]
    #         not_selsected = torch.randperm(num_pos)[:num_pos - num_pos_smps]
    #         # not_selsected_indices = pos_indices[not_selsected]
    #         weights[:, pos_indices[not_selsected]] = 0
    #         # print("正样本，负样本数量", num_neg, num_pos, weights[weights != 0].shape)
    #     loss_reg = weighted_smooth_l1_loss(rcnn_reg, tgt_reg,
    #                                     weights=weights / max(weights.sum(),
    #                                                             1)).sum()

    #     # iou loss
    #     weights_iou = torch.ones(tgt_iou.shape, device=tgt_iou.device)
    #     # TODO: also count the negative samples
    #     tgt_iou = 2 * (tgt_iou - 0.5)  # normalize to -1, 1
    #     if self.use_iou_loss == "naive":
    #         # print("use naive iou loss")
    #         loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
    #                                         weights=tgt_cls).mean()
    #     elif self.use_iou_loss == "naive+":
    #         # print("use naive+ iou loss")
    #         loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
    #                                         weights=weights_iou).mean()
    #     elif self.use_iou_loss == "naive+W3":
    #         # print("use naive+W3 iou loss")
    #         # 这个操作有点问题捏，这里不是0.7，而是0.4，这组实验需要重新跑
    #         weights_iou[tgt_iou < 0.4] = 3
    #         loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
    #                                         weights=weights_iou).mean()
    #     else:
    #         print("else use naive iou loss")
    #         loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
    #                                         weights=tgt_cls).mean()

    #     loss_cls_reduced = loss_cls * self.cls['weight']
    #     loss_iou_reduced = loss_iou * self.iou['weight']
    #     loss_reg_reduced = loss_reg * self.reg['weight']
        
    #     return loss_cls_reduced,loss_iou_reduced,loss_reg_reduced
        
    def edge_loss(self, predicted, target):
        # 定义Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(predicted.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(predicted.device)
        
        # 如果输入是多通道的，需要按通道计算梯度
        if len(predicted.shape) == 4:  # [B, C, H, W]
            batch_size, channels = predicted.shape[:2]
            sobel_x = sobel_x.repeat(channels, 1, 1, 1)
            sobel_y = sobel_y.repeat(channels, 1, 1, 1)
            
            # 计算预测图像的梯度
            pred_grad_x = F.conv2d(predicted, sobel_x, padding=1, groups=channels)
            pred_grad_y = F.conv2d(predicted, sobel_y, padding=1, groups=channels)
            pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)  # 添加小值避免0处梯度
            
            # 计算目标图像的梯度
            target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=channels)
            target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=channels)
            target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
            
        else:  # 单通道图像 [B, H, W]
            # 添加通道维度
            pred_reshaped = predicted.unsqueeze(1)
            target_reshaped = target.unsqueeze(1)
            
            # 计算预测图像的梯度
            pred_grad_x = F.conv2d(pred_reshaped, sobel_x, padding=1)
            pred_grad_y = F.conv2d(pred_reshaped, sobel_y, padding=1)
            pred_grad = torch.sqrt(pred_grad_x**2 + pred_grad_y**2 + 1e-8)
            
            # 计算目标图像的梯度
            target_grad_x = F.conv2d(target_reshaped, sobel_x, padding=1)
            target_grad_y = F.conv2d(target_reshaped, sobel_y, padding=1)
            target_grad = torch.sqrt(target_grad_x**2 + target_grad_y**2 + 1e-8)
        
        # 计算梯度差的L2损失
        loss = F.mse_loss(pred_grad, target_grad)
        
        return loss

    
    def gaussian_kernel(self, size, sigma):
        """
        生成高斯卷积核
        """
        # 确保size是正确的奇数值
        if size % 2 == 0:
            size = size + 1
            
        # 创建一维坐标
        coords = torch.arange(size).to(dtype=torch.float)
        coords -= size // 2
        
        # 计算高斯权重
        g = coords ** 2
        g = -(g / (2 * sigma ** 2))
        g = torch.exp(g)
        
        # 归一化
        g /= g.sum()
        kernel_1d = g.view(-1)
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # 外积生成二维核
        
        return kernel_2d  # 直接返回形状为[size, size]的二维核
        
    def ssim_loss(self,predicted, target, window_size=11, sigma=1.5, eps=1e-8, K1=0.01, K2=0.03):
        """
        Args:
            predicted: LayerNorm处理后的预测图像张量
            target: LayerNorm处理后的目标图像张量
        """
        # 检查输入维度
        if len(predicted.shape) != 4 or len(target.shape) != 4:
            raise ValueError("输入必须是4D张量 [B, C, H, W]")
        
        # 获取批次大小和通道数
        batch_size, channels = predicted.shape[:2]
        
        # 估计数据范围
        data_range = torch.max(torch.abs(torch.cat([
            predicted.view(batch_size, -1), 
            target.view(batch_size, -1)
        ], dim=1)), dim=1)[0]
        data_range = data_range.view(batch_size, 1, 1, 1)
        
        # 计算常数 - 基于数据范围动态调整
        C1 = (K1 * data_range) ** 2
        C2 = (K2 * data_range) ** 2
        
        # 生成高斯核
        kernel = self.gaussian_kernel(window_size, sigma).to(predicted.device)
        kernel = kernel.view(1, 1, window_size, window_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        
        # 填充
        pad = window_size // 2
        
        # 计算均值
        mu1 = F.conv2d(predicted, kernel, padding=pad, groups=channels)
        mu2 = F.conv2d(target, kernel, padding=pad, groups=channels)
        
        # 计算平方
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(predicted * predicted, kernel, padding=pad, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=channels) - mu2_sq
        sigma12 = F.conv2d(predicted * target, kernel, padding=pad, groups=channels) - mu1_mu2
        
        # 确保方差非负
        sigma1_sq = F.relu(sigma1_sq + eps)  # 使用relu确保非负
        sigma2_sq = F.relu(sigma2_sq + eps)
        
        # SSIM公式 - 在每个可能为零的项都添加eps
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1 + eps) * (sigma1_sq + sigma2_sq + C2 + eps))
        
        # 确保SSIM值在[-1, 1]范围内
        ssim_value = torch.clamp(ssim_map.mean(), -1, 1)
        return 1.0 - ssim_value
    
    def cal_diff_loss(self, p_e_batch, g_e_batch,t_e_batch):
        # 初始化总损失
        diff_loss = 0.0
        
        # 对每个批次中的每个特征计算损失并累加
        for batch_idx in range(len(p_e_batch)):
            p_noise = p_e_batch[batch_idx]
            g_noise = g_e_batch[batch_idx]
            # 计算当前批次的损失
            pixel_loss = F.mse_loss(p_noise, g_noise, reduction='mean')
            # edge_loss = self.edge_loss(p_noise, g_noise)
            ssim_loss = self.ssim_loss(p_noise, g_noise)
            # 累加总损失
            diff_loss += pixel_loss + ssim_loss
        # 计算平均损失
        diff_loss = diff_loss / len(p_e_batch)
        return diff_loss, pixel_loss, 0, ssim_loss
        
    def forward(self, output_dict, target_dict, epoch = 1, train = True):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        
        if 'rcnn_label_dict' in output_dict.keys():
            rcnn_cls = output_dict['stage2_out']['rcnn_cls']
            rcnn_iou = output_dict['stage2_out']['rcnn_iou']
            rcnn_reg = output_dict['stage2_out']['rcnn_reg']

            tgt_cls = output_dict['rcnn_label_dict']['cls_tgt']
            tgt_iou = output_dict['rcnn_label_dict']['iou_tgt']
            tgt_reg = output_dict['rcnn_label_dict']['reg_tgt']

            loss_cls_reduced, loss_iou_reduced, loss_reg_reduced = self.stage2_loss(rcnn_cls,rcnn_iou,rcnn_reg,tgt_cls,tgt_iou,tgt_reg)
            rcnn_loss = loss_cls_reduced + loss_iou_reduced + loss_reg_reduced
            total_loss = rcnn_loss
            
            self.loss_cls_reduced = loss_cls_reduced.item()
            self.loss_iou_reduced = loss_iou_reduced.item()
            self.loss_reg_reduced = loss_reg_reduced.item()
            self.rcnn_loss = rcnn_loss.item()
            self.total_loss = total_loss.item()   
            self.loss_dict.update({
                'total_loss': self.total_loss,
                'rcnn_loss': self.rcnn_loss,
                'cls_loss': self.loss_cls_reduced,
                'iou_loss': self.loss_iou_reduced,
                'reg_loss': self.loss_reg_reduced,
            })

        if 'pred_out' in output_dict.keys() and 'rcnn_label_dict' in output_dict.keys():
            p_e_batch = output_dict['pred_out']
            if output_dict['target'] == 'eps':
                g_e_batch = output_dict['gt_noise']
            elif output_dict['target'] == 'x0':
                g_e_batch = output_dict["gt_x0"]
            t_e_batch = output_dict['t']
            diff_loss, pixel_loss, edge_loss, ssim_loss = self.cal_diff_loss(p_e_batch, g_e_batch,t_e_batch)
            total_loss = rcnn_loss + diff_loss

            self.total_loss = total_loss.item()
            self.diff_loss = diff_loss.item()
            self.loss_dict.update({'total_loss': self.total_loss,
                                    'diff_loss': self.diff_loss,
                                    'pixel_loss':pixel_loss.item(),
                                    # 'edge_loss':edge_loss.item(),
                                    'ssim_loss':ssim_loss,
                                    })
        elif 'pred_out' in output_dict.keys():
            p_e_batch = output_dict['pred_out']  # 批量列表
            if output_dict['target'] == 'eps':
                g_e_batch = output_dict['gt_noise']
            elif output_dict['target'] == 'x0':
                g_e_batch = output_dict["gt_x0"]
            t_e_batch = output_dict['t']
            diff_loss, pixel_loss, edge_loss, ssim_loss = self.cal_diff_loss(p_e_batch, g_e_batch,t_e_batch)
            total_loss = diff_loss

            self.total_loss = total_loss.item()
            self.diff_loss = diff_loss.item()
            self.loss_dict.update({'total_loss': self.total_loss,
                                   'rcnn_loss': self.rcnn_loss,
                                   'reg_loss': self.reg_loss,
                                   'cls_loss': self.cls_loss,
                                   'iou_loss': self.iou_loss,
                                   'diff_loss': self.diff_loss,
                                   'pixel_loss':pixel_loss.item(),
                                #    'edge_loss':edge_loss.item(),
                                   'ssim_loss':ssim_loss, 
                                    })
    
        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    def diff_loss_func(self, pred_noise: torch.Tensor,
                      gt_noise: torch.Tensor,
                      weights=1):
        return ((gt_noise - pred_noise) ** 2).sum(dim=1).mean()*weights
    
   
    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in    
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None, optimizer=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss'] if 'total_loss' in self.loss_dict.keys() else torch.tensor(0)
        reg_loss = self.loss_dict['reg_loss'] if 'reg_loss' in self.loss_dict.keys() else torch.tensor(0)
        cls_loss = self.loss_dict['cls_loss'] if 'cls_loss' in self.loss_dict.keys() else torch.tensor(0)
        iou_loss = self.loss_dict['iou_loss'] if 'iou_loss' in self.loss_dict.keys() else torch.tensor(0)
        rcnn_loss = self.loss_dict['rcnn_loss'] if 'rcnn_loss' in self.loss_dict.keys() else torch.tensor(0)
        
        # 获取当前学习率
        lr = None
        if optimizer is not None:
            lr = optimizer.param_groups[0]['lr']
        
        if pbar is None:
            if 'diff_loss' in self.loss_dict:
                diff_loss = self.loss_dict['diff_loss']
                # pixel_loss = self.loss_dict['pixel_loss']
                # edge_loss = self.loss_dict['edge_loss']
                ssim_loss = self.loss_dict['ssim_loss']
                print("[epoch %d][%d/%d], LR: %.6f || Loss: %.3f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc: %.3f || Iou: %.3f || Diff: %.4f|| Ssim: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    lr if lr is not None else 0,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss, diff_loss, ssim_loss))
            else:
                print("[epoch %d][%d/%d], LR: %.6f || Loss: %.4f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc Loss: %.4f|| Iou: %.3f" % (
                    epoch, batch_id + 1, batch_len,
                    lr if lr is not None else 0,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss))
        else:
            if 'diff_loss' in self.loss_dict:
                diff_loss = self.loss_dict['diff_loss']
                # pixel_loss = self.loss_dict['pixel_loss']
                # edge_loss = self.loss_dict['edge_loss']
                ssim_loss = self.loss_dict['ssim_loss']
                pbar.set_description("[epoch %d][%d/%d], LR: %.6f || Loss: %.3f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc: %.3f || Iou: %.3f || Diff: %.4f|| Ssim: %.4f" % (
                    epoch, batch_id + 1, batch_len,
                    lr if lr is not None else 0,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss, diff_loss, ssim_loss))
            else:
                pbar.set_description("[epoch %d][%d/%d], LR: %.6f || Loss: %.4f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc Loss: %.4f|| Iou: %.3f" % (
                    epoch, batch_id + 1, batch_len,
                    lr if lr is not None else 0,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss))

        # 记录学习率到tensorboard
        if lr is not None:
            writer.add_scalar('Learning_rate', lr, epoch*batch_len + batch_id)
        
        if 'diff_loss' in self.loss_dict:
            writer.add_scalar('Reconstruction_loss', diff_loss,
                          epoch*batch_len + batch_id)
            # writer.add_scalar('Pixel_loss', pixel_loss,
            #               epoch*batch_len + batch_id)
            # writer.add_scalar('Edge_loss', edge_loss,
            #               epoch*batch_len + batch_id)
            writer.add_scalar('Ssim_loss', ssim_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Regression_loss', reg_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Cls_loss', cls_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Iou_loss', iou_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Rcnn_loss', rcnn_loss,
                          epoch*batch_len + batch_id)