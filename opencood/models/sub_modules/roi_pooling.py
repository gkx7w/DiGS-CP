import torch
import torch.nn as nn
import torch.nn.functional as F


class ROIPooling(nn.Module):
    def __init__(self, pool_size=(7, 7)):
        """
        ROI Pooling层实现
        
        Args:
            pool_size: tuple, 输出特征的空间大小，默认为(7,7)
        """
        super(ROIPooling, self).__init__()
        self.pool_size = pool_size
        
    def forward(self, batch_dict):
        """
        执行ROI Pooling操作
        
        Args:
            batch_dict: 包含特征和边界框信息的字典
            label_dict: 边界框信息在batch_dict中的键
            targets: 目标边界框在label_dict中的键
            
        Returns:
            roi_features: 池化后的ROI特征 [batch_size, num_boxes, channels, pool_height, pool_width]
        """
        # 获取特征图
        features = batch_dict['spatial_features']  # [B, C, H, W]
        
        # 获取边界框
        boxes = batch_dict["gt_boxes"]["targets"] # 应为[B, num_boxes, 4或更多]格式
        
        batch_size = features.shape[0]
        device = features.device
        
        # 创建输出tensor
        pooled_features = []
        
        # 对每个样本进行处理
        for batch_idx in range(batch_size):
            # 获取当前样本的特征和边界框
            current_feat = features[batch_idx]  # [C, H, W]
            current_boxes = boxes[batch_idx]    # [num_boxes, box_dim]
            
            # 确保边界框是有效的（非零，非NaN）
            valid_mask = torch.any(current_boxes != 0, dim=1)
            if torch.sum(valid_mask) == 0:
                # 如果没有有效边界框，创建全零特征
                h, w = self.pool_size
                num_channels = current_feat.shape[0]
                dummy_features = torch.zeros(
                    (current_boxes.shape[0], num_channels, h, w), 
                    device=device
                )
                pooled_features.append(dummy_features)
                continue
                
            current_boxes = current_boxes[valid_mask]
            
            # 提取边界框的坐标（假设格式为[x1, y1, x2, y2, ...])
            # 注意：可能需要根据实际边界框格式调整
            if current_boxes.shape[1] > 4:
                # 如果边界框包含更多信息（如类别，置信度等），只取前4个坐标
                rois = current_boxes[:, :4]
            else:
                rois = current_boxes
                
            # 确保坐标在特征图范围内
            feat_height, feat_width = current_feat.shape[1], current_feat.shape[2]
            rois[:, 0] = torch.clamp(rois[:, 0], min=0, max=feat_width - 1)
            rois[:, 1] = torch.clamp(rois[:, 1], min=0, max=feat_height - 1)
            rois[:, 2] = torch.clamp(rois[:, 2], min=0, max=feat_width - 1)
            rois[:, 3] = torch.clamp(rois[:, 3], min=0, max=feat_height - 1)
            
            # 确保x1 < x2, y1 < y2
            x1, y1, x2, y2 = rois[:, 0], rois[:, 1], rois[:, 2], rois[:, 3]
            x1 = torch.min(x1, x2)
            x2 = torch.max(x1, x2)
            y1 = torch.min(y1, y2)
            y2 = torch.max(y1, y2)
            rois = torch.stack([x1, y1, x2, y2], dim=1)
            
            # 为每个ROI执行池化
            roi_feats = []
            for roi in rois:
                x1, y1, x2, y2 = roi
                
                # 转换为整数坐标
                x1, y1, x2, y2 = map(lambda x: x.int(), [x1, y1, x2, y2])
                
                # 处理边界情况
                if x1 == x2:
                    x2 = x1 + 1
                if y1 == y2:
                    y2 = y1 + 1
                    
                # 提取ROI区域
                roi_feat = current_feat[:, y1:y2, x1:x2]  # [C, roi_h, roi_w]
                
                # 执行自适应池化到指定大小
                roi_feat = F.adaptive_max_pool2d(roi_feat.unsqueeze(0), self.pool_size).squeeze(0)
                
                roi_feats.append(roi_feat)
            
            # 处理无有效ROI的情况
            if len(roi_feats) == 0:
                h, w = self.pool_size
                num_channels = current_feat.shape[0]
                dummy_features = torch.zeros(
                    (current_boxes.shape[0], num_channels, h, w), 
                    device=device
                )
                roi_feats = [dummy_features]
            
            # 将当前样本的所有ROI特征堆叠起来
            stacked_roi_feats = torch.stack(roi_feats)
            
            # 如果有效边界框少于总边界框，填充回原始大小
            if valid_mask.sum() < valid_mask.shape[0]:
                h, w = self.pool_size
                num_channels = current_feat.shape[0]
                full_roi_feats = torch.zeros(
                    (valid_mask.shape[0], num_channels, h, w), 
                    device=device
                )
                full_roi_feats[valid_mask] = stacked_roi_feats
                stacked_roi_feats = full_roi_feats
            
            pooled_features.append(stacked_roi_feats)
            
        # 将所有样本的ROI特征堆叠起来
        roi_features = torch.stack(pooled_features)  # [B, num_boxes, C, pool_h, pool_w]
        
        return roi_features


def roi_pool(batch_dict, pool_size=(7, 7)):
    """
    ROI Pooling的函数式接口
    
    Args:
        batch_dict: 包含特征和边界框信息的字典
        label_dict: 边界框信息在batch_dict中的键
        targets: 目标边界框在label_dict中的键
        pool_size: 输出特征的空间大小
        
    Returns:
        roi_features: 池化后的ROI特征
    """
    roi_pooling = ROIPooling(pool_size=pool_size)
    return roi_pooling(batch_dict)


# 示例用法
if __name__ == "__main__":
    # 创建一个假的batch_dict进行测试
    batch_size = 2
    channels = 64
    height, width = 200, 200
    num_boxes = 5
    box_dim = 4  # [x1, y1, x2, y2, class, score, ...]
    
    batch_dict = {
        'spatial_features': torch.rand(batch_size, channels, height, width),
        'gt_boxes': {
            'targets': torch.rand(batch_size, num_boxes, box_dim)
        }
    }
    
    # 将坐标值缩放到特征图尺寸范围内
    batch_dict['gt_boxes']['targets'][:, :, 0:4] *= torch.tensor([width, height, width, height])
    
    # 测试ROI Pooling
    pool_size = (7, 7)
    roi_features = roi_pool(batch_dict, pool_size=pool_size)
    
    print(f"ROI Features shape: {roi_features.shape}")  # 应为 [batch_size, num_boxes, channels, pool_height, pool_width]