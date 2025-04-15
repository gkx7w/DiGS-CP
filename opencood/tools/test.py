import torch
import numpy as np
from collections import OrderedDict

def compare_backbone_weights(checkpoint1_path, checkpoint2_path, verbose=True, show_identical=True):
    """
    比较两个checkpoint文件中backbone层的权重是否相同，并输出相同的层
    
    参数:
        checkpoint1_path: 第一个checkpoint文件路径
        checkpoint2_path: 第二个checkpoint文件路径
        verbose: 是否打印详细信息，默认为True
        show_identical: 是否显示相同的层，默认为True
    
    返回:
        is_identical: 布尔值，指示backbone层权重是否完全相同
        diff_summary: 字典，包含不同层的差异统计信息
        identical_layers: 列表，完全相同的层名称
    """
    # 加载模型权重
    checkpoint1 = torch.load(checkpoint1_path, map_location='cpu')
    checkpoint2 = torch.load(checkpoint2_path, map_location='cpu')
    
    # 获取state_dict，处理不同的checkpoint格式
    if 'state_dict' in checkpoint1:
        state_dict1 = checkpoint1['state_dict']
    else:
        state_dict1 = checkpoint1
        
    if 'state_dict' in checkpoint2:
        state_dict2 = checkpoint2['state_dict']
    else:
        state_dict2 = checkpoint2
    
    # 筛选backbone相关的权重
    backbone1 = OrderedDict()
    backbone2 = OrderedDict()
    
    for key in state_dict1:
        if 'backbone' in key:
            backbone1[key] = state_dict1[key]
            
    for key in state_dict2:
        if 'backbone' in key:
            backbone2[key] = state_dict2[key]
    
    # 检查两个backbone是否有相同的键
    keys1 = set(backbone1.keys())
    keys2 = set(backbone2.keys())
    
    if keys1 != keys2:
        if verbose:
            print("Backbone层的键不匹配!")
            print(f"仅在checkpoint1中的键: {keys1 - keys2}")
            print(f"仅在checkpoint2中的键: {keys2 - keys1}")
        return False, {"error": "键不匹配"}, []
    
    # 比较每一层的权重
    is_identical = True
    diff_summary = {}
    identical_layers = []
    different_layers = []
    
    for key in backbone1:
        # 获取两个权重张量
        weight1 = backbone1[key]
        weight2 = backbone2[key]
        
        # 检查形状是否相同
        if weight1.shape != weight2.shape:
            is_identical = False
            diff_summary[key] = {
                "shape_match": False,
                "shape1": list(weight1.shape),
                "shape2": list(weight2.shape)
            }
            different_layers.append(key)
            if verbose:
                print(f"层 {key} 的形状不匹配: {weight1.shape} vs {weight2.shape}")
            continue
        
        # 将张量转换为浮点型再计算差异
        weight1_float = weight1.float()
        weight2_float = weight2.float()
        
        # 计算差异
        weight_diff = (weight1_float - weight2_float).abs()
        max_diff = weight_diff.max().item()
        mean_diff = weight_diff.mean().item()
        
        # 检查是否完全相同(考虑到浮点误差)
        if max_diff > 1e-6:  # 允许微小的浮点误差
            is_identical = False
            diff_summary[key] = {
                "shape_match": True,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "shape": list(weight1.shape)
            }
            different_layers.append(key)
            if verbose:
                print(f"层 {key} 的权重不同: 最大差异={max_diff:.8f}, 平均差异={mean_diff:.8f}")
        else:
            identical_layers.append(key)
            if verbose and show_identical:
                print(f"层 {key} 的权重完全相同")
    
    if verbose:
        print("\n統計信息:")
        print(f"共检查 {len(backbone1)} 个层")
        print(f"相同的层: {len(identical_layers)} 个")
        print(f"不同的层: {len(different_layers)} 个")
    
    if is_identical and verbose:
        print("两个checkpoint中的backbone层权重完全相同!")
    
    return is_identical, diff_summary, identical_layers


def visualize_weight_differences(checkpoint1_path, checkpoint2_path, layer_name=None):
    """
    可视化两个checkpoint中特定层或所有backbone层的权重差异分布
    
    参数:
        checkpoint1_path: 第一个checkpoint文件路径
        checkpoint2_path: 第二个checkpoint文件路径
        layer_name: 要可视化的特定层名称，如果为None则可视化所有层的总体差异
    
    返回:
        差异统计信息的字典
    """
    import matplotlib.pyplot as plt
    
    # 加载模型权重
    checkpoint1 = torch.load(checkpoint1_path, map_location='cpu')
    checkpoint2 = torch.load(checkpoint2_path, map_location='cpu')
    
    # 获取state_dict
    if 'state_dict' in checkpoint1:
        state_dict1 = checkpoint1['state_dict']
    else:
        state_dict1 = checkpoint1
        
    if 'state_dict' in checkpoint2:
        state_dict2 = checkpoint2['state_dict']
    else:
        state_dict2 = checkpoint2
    
    # 筛选backbone相关的权重
    backbone1 = OrderedDict()
    backbone2 = OrderedDict()
    
    for key in state_dict1:
        if 'backbone' in key:
            backbone1[key] = state_dict1[key]
            
    for key in state_dict2:
        if 'backbone' in key:
            backbone2[key] = state_dict2[key]
    
    # 确保两个模型有相同的键
    common_keys = set(backbone1.keys()).intersection(set(backbone2.keys()))
    
    # 如果指定了特定层
    if layer_name is not None:
        if layer_name not in common_keys:
            print(f"错误: 指定的层 {layer_name} 不存在于两个checkpoint的backbone中")
            return {}
        
        # 计算指定层的差异
        weight1 = backbone1[layer_name]
        weight2 = backbone2[layer_name]
        
        if weight1.shape != weight2.shape:
            print(f"错误: 层 {layer_name} 的形状不匹配: {weight1.shape} vs {weight2.shape}")
            return {"error": "形状不匹配"}
        
        # 计算并可视化差异
        weight_diff = (weight1 - weight2).flatten().numpy()
        
        plt.figure(figsize=(10, 6))
        plt.hist(weight_diff, bins=100)
        plt.title(f'层 {layer_name} 的权重差异分布')
        plt.xlabel('差异值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return {
            "layer": layer_name,
            "max_diff": np.max(np.abs(weight_diff)),
            "mean_diff": np.mean(np.abs(weight_diff)),
            "std_diff": np.std(weight_diff)
        }
    
    # 如果要可视化所有层
    else:
        all_diffs = []
        layer_stats = {}
        
        for key in common_keys:
            weight1 = backbone1[key]
            weight2 = backbone2[key]
            
            if weight1.shape != weight2.shape:
                print(f"警告: 层 {key} 的形状不匹配，已跳过")
                continue
            
            # 计算差异
            weight_diff = (weight1 - weight2).flatten().numpy()
            all_diffs.append(weight_diff)
            
            # 保存每层的统计信息
            layer_stats[key] = {
                "max_diff": np.max(np.abs(weight_diff)),
                "mean_diff": np.mean(np.abs(weight_diff)),
                "std_diff": np.std(weight_diff)
            }
        
        # 合并所有差异
        all_diffs = np.concatenate(all_diffs)
        
        # 可视化整体差异分布
        plt.figure(figsize=(12, 7))
        plt.hist(all_diffs, bins=100)
        plt.title('所有backbone层的权重差异分布')
        plt.xlabel('差异值')
        plt.ylabel('频率')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 可视化每层的最大和平均差异
        layers = list(layer_stats.keys())
        max_diffs = [layer_stats[layer]["max_diff"] for layer in layers]
        mean_diffs = [layer_stats[layer]["mean_diff"] for layer in layers]
        
        plt.figure(figsize=(14, 7))
        
        # 为了更好的可视化，只显示部分层名称
        n_layers = len(layers)
        display_indices = np.linspace(0, n_layers-1, min(20, n_layers)).astype(int)
        display_layers = [layers[i] for i in display_indices]
        
        # 创建柱状图索引
        indices = np.arange(len(layers))
        
        plt.subplot(1, 2, 1)
        plt.bar(indices, max_diffs)
        plt.title('每层的最大权重差异')
        plt.ylabel('最大差异')
        plt.xticks(display_indices, [l.split('.')[-2:] for l in display_layers], rotation=90)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.bar(indices, mean_diffs)
        plt.title('每层的平均权重差异')
        plt.ylabel('平均差异')
        plt.xticks(display_indices, [l.split('.')[-2:] for l in display_layers], rotation=90)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return {
            "overall": {
                "max_diff": np.max(np.abs(all_diffs)),
                "mean_diff": np.mean(np.abs(all_diffs)),
                "std_diff": np.std(all_diffs)
            },
            "layers": layer_stats
        }

# 使用示例
if __name__ == "__main__":
    # 比较两个checkpoint的backbone权重
    is_identical, diff_summary = compare_backbone_weights(
        "/data/gkx/Code/checkpoints/fisrt_no_fuse_best/net_epoch80.pth", 
        "/data/gkx/Code/checkpoints/1000_step_mdd/net_epoch89.pth"
    )
    
    # 可视化权重差异
    # visualize_weight_differences("path/to/checkpoint1.pth", "path/to/checkpoint2.pth")
    
    # 如果要可视化特定层的差异
    # visualize_weight_differences("path/to/checkpoint1.pth", "path/to/checkpoint2.pth", 
    #                             "backbone.layer1.0.conv1.weight")