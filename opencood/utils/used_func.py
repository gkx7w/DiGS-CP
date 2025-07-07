# 统计最大whl 将gt扩展到相同大小 3:5 hwl
# for i in range(len(batch_dict['gt_boxes'])):
#     if 'max_h' not in gt_box['gt_boxes'] or gt_box['gt_boxes']['max_h'] < torch.max(batch_dict['gt_boxes'][i][:,3]):
#         gt_box['gt_boxes']['max_h'] = torch.max(batch_dict['gt_boxes'][i][:,3])
#         print("max_h: ",gt_box['gt_boxes']['max_h'])
#     if 'max_w' not in gt_box['gt_boxes'] or gt_box['gt_boxes']['max_w'] < torch.max(batch_dict['gt_boxes'][i][:,4]):
#         gt_box['gt_boxes']['max_w'] = torch.max(batch_dict['gt_boxes'][i][:,4])
#         print("max_w: ",gt_box['gt_boxes']['max_w'])
#     if 'max_l' not in gt_box['gt_boxes'] or gt_box['gt_boxes']['max_l'] < torch.max(batch_dict['gt_boxes'][i][:,5]):
#         gt_box['gt_boxes']['max_l'] = torch.max(batch_dict['gt_boxes'][i][:,5])
#         print("max_l: ",gt_box['gt_boxes']['max_l'])
#         print('______________________________________')


# # 这些操作都可以放到数据集处理里面吧？好像不行，我还需要单阶段的bev，这个要用中期融合数据处理
# # 获取gt框
# batch_dict['gt_boxes'] = [b[m][:, [0, 1, 2, 3, 4, 5, 6]].float() for b, m in
#                             zip(data_dict['object_bbx_center'],
#                                 data_dict['object_bbx_mask'].bool())]
# # 将gt扩展到相同大小 3:5 hwl
# for gt_boxes in batch_dict['gt_boxes']:
#     gt_boxes[:, 3:6] = torch.tensor(self.max_hwl, dtype=gt_boxes.dtype, device=gt_boxes.device)

# gt_voxel_batch = []
# for b in range(len(batch_dict['gt_boxes'])): 
#     # 获取gt框中的点云  
#     point_indices = points_in_boxes_cpu(data_dict['row_lidar_points'][b][:, :3], batch_dict['gt_boxes'][b][:,[0, 1, 2, 5, 4, 3, 6]].cpu().numpy()) 
#     gt_voxel_stack = []
#     for car_idx in range(len(batch_dict['gt_boxes'][b])):
#         # 减去box中心
#         gt_point = torch.from_numpy(data_dict['row_lidar_points'][b][:, :3][point_indices[car_idx] > 0]).to(voxel_features.device)
#         gt_point -= batch_dict['gt_boxes'][b][car_idx][0:3]
#         # 旋转点云
#         gt_point = common_utils.rotate_points_along_z(gt_point[np.newaxis, :, :].cpu().numpy(), np.array([float(batch_dict['gt_boxes'][b][car_idx][6].cpu())]))[0]
#         # 体素化
#         gt_point_with_intensity = np.concatenate([gt_point, data_dict['row_lidar_points'][b][point_indices[car_idx] > 0, 3:4]], axis=1)
#         gt_voxel_stack.append(self.voxel_preprocessor.preprocess(gt_point_with_intensity)['voxel_features'])
#     gt_voxel_stack = np.vstack(gt_voxel_stack)
#     gt_voxel_batch.append(gt_voxel_stack)
# # batch_dict['gt_voxel_batch'] = torch.from_numpy(np.array(gt_voxel_batch)).to(voxel_features.device)



# # 5. 将点云数据转换为体素/BEV/降采样点云
# gt_boxes = object_bbx_center[mask.astype(bool)]
"""可视化"""
# # pc_range = [-140.8, -40, -3, 140.8, 40, 1]
# # visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/origin_gt_boxes.png")
# # 将gt扩展到相同大小 3:6 hwl
# gt_boxes[:, 3:6] = np.array(self.max_hwl)
"""可视化"""
# # visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/expend_gt_boxes.png")        
# # 获取gt框中的点云  不能使用gpu版与dataloader多线程有关
# point_indices = points_in_boxes_cpu(projected_lidar_stack[:, :3], gt_boxes[:,[0, 1, 2, 5, 4, 3, 6]]) 
# gt_voxel_stack = []
# gt_coords_stack = []
# gt_num_points_stack = []
# # gt_boxes_stack = [] 用不到gt了
# for car_idx in range(len(gt_boxes)):
#     # 减去box中心
#     gt_point = projected_lidar_stack[point_indices[car_idx] > 0]
#     gt_point[:, :3] -= gt_boxes[car_idx][0:3]
"""可视化"""
#     # pc_range = [-15, -15, -1, 15, 15, 1]
#     # gt_boxes[car_idx][0:3] = [0, 0, 0]
#     # visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_point_{car_idx}.png",scale_bev=10)
#     # 旋转点云 这个好像可以并行 拿外面去
#     gt_point = common_utils.rotate_points_along_z(gt_point[np.newaxis, :, :], np.array([-float(gt_boxes[car_idx][6])]))[0]
"""可视化"""
#     # gt_boxes[car_idx][0:3] = common_utils.rotate_points_along_z(gt_boxes[car_idx][np.newaxis, np.newaxis, 0:3], np.array([-float(gt_boxes[car_idx][6])]))[0,0]
#     # gt_boxes[car_idx][6] -= float(gt_boxes[car_idx][6])
#     # visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_rotate_{car_idx}.png",scale_bev=10)
#     # 体素化 不能并行！！
#     # gt_point_with_intensity = np.concatenate([gt_point, projected_lidar_stack[point_indices[car_idx] > 0, 3:4]], axis=1)
#     processed_lidar_car = self.pre_processor.preprocess(gt_point)
#     gt_voxel_stack.append(processed_lidar_car['voxel_features'])
#     gt_coords_stack.append(processed_lidar_car['voxel_coords'])
#     gt_num_points_stack.append(processed_lidar_car['voxel_num_points'])
#     # gt_boxes_stack.append(gt_boxes[car_idx])
# #还要区分是哪个box的，再加一个mask？ 这个可以在前面的for循环一起做
# gt_masks = []
# for i, arr in enumerate(gt_voxel_stack):
#     mk = np.full(arr.shape[0], i, dtype=np.int32)
#     gt_masks.append(mk)
# # gt_voxel_stack = []
# if len(gt_voxel_stack) == 0:
#     print("gt_voxel_stack is empty")
#     processed_lidar = None
# else:
#     processed_lidar = {
#         'voxel_features': np.concatenate(gt_voxel_stack, axis=0),
#         'voxel_coords': np.concatenate(gt_coords_stack, axis=0),
#         'voxel_num_points': np.concatenate(gt_num_points_stack, axis=0),
#         'gt_masks': np.concatenate(gt_masks, axis=0),
#         # 'gt_boxes': gt_boxes_stack
#         }

# # 5. 将点云数据转换为体素/BEV/降采样点云
# gt_boxes = object_bbx_center[mask.astype(bool)]
# gt_boxes_ori = gt_boxes.copy()
# # pc_range = [-140.8, -40, -3, 140.8, 40, 1]
# # visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/origin_gt_boxes.png")
# # 将gt扩展到相同大小 3:6 hwl
# gt_boxes[:, 3:6] = np.array(self.max_hwl)
# # visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/expend_gt_boxes.png")        
# # 获取gt框中的点云  不能使用gpu版与dataloader多线程有关
# point_indices = points_in_boxes_cpu(projected_lidar_stack[:, :3], gt_boxes[:,[0, 1, 2, 5, 4, 3, 6]])
# point_indices_ori = points_in_boxes_cpu(projected_lidar_stack[:, :3], gt_boxes_ori[:,[0, 1, 2, 5, 4, 3, 6]])
# gt_voxel_stack = []
# gt_coords_stack = []
# gt_num_points_stack = []
# gt_masks = []
# rotation_angles = -gt_boxes[:, 6].astype(float)
# for car_idx in range(len(gt_boxes)):
#     # 获取当前box中的点并平移到以box中心为原点的坐标系
#     gt_point = projected_lidar_stack[point_indices[car_idx] > 0]
#     gt_point_ori = projected_lidar_stack[point_indices_ori[car_idx] > 0]
#     gt_point[:, :3] -= gt_boxes[car_idx][0:3] 
#     gt_point_ori[:, :3] -= gt_boxes_ori[car_idx][0:3]
#     pc_range = [-15, -15, -1, 15, 15, 1]
#     gt_boxes[car_idx][0:3] = [0, 0, 0]
#     gt_boxes_ori[car_idx][0:3] = [0, 0, 0]
#     visualize_gt_boxes(gt_boxes_ori[car_idx][np.newaxis, :], gt_point_ori, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_ori_{car_idx}.png",scale_3d=10)
#     visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_expand_{car_idx}.png",scale_3d=10)
    
    
# def count_trainable_parameters(model):
#     # 只统计requires_grad=True的参数
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     trainable_size_mb = trainable_params * 4 / (1024 * 1024)  # 假设用Float32
    
#     # 统计所有参数（包括冻结的）
#     total_params = sum(p.numel() for p in model.parameters())
#     total_size_mb = total_params * 4 / (1024 * 1024)
    
#     print(f"可训练参数数量: {trainable_params:,}")
#     print(f"可训练参数大小: {trainable_size_mb:.2f} MB")
#     print(f"总参数数量: {total_params:,}")
#     print(f"总参数大小: {total_size_mb:.2f} MB")
#     print(f"可训练参数占比: {trainable_params/total_params*100:.2f}%")
    
#     return trainable_params, total_params

# trainable_params, total_params = count_trainable_parameters(model)
# def check_trainable_params(model, model_name):
#     print(f"\n{model_name} 参数训练状态:")
#     for name, param in model.named_parameters():
#         print(f"{name}: {'可训练' if param.requires_grad else '已冻结'}")
# check_trainable_params(model,"model")    


# class AttentionBlock(nn.Module):
#     def __init__(self, n_head: int, channels: int, d_cond=None):
#         super().__init__()
#         # channels = n_head * n_embd
#         # 有空把这个num_groups改成超参数
#         self.groupnorm = nn.GroupNorm(2, channels, eps=1e-6)
#         self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
#         self.layernorm_1 = nn.LayerNorm(channels)
#         self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
#         # 如果有条件输入d_cond，则使用交叉注意力
#         if d_cond is not None:
#             self.layernorm_2 = nn.LayerNorm(channels)
#             self.attention_2 = CrossAttention(n_head, channels, d_cond, in_proj_bias=False)
#         else:
#             self.layernorm_2 = None
#             self.attention_2 = None
#         self.layernorm_3 = nn.LayerNorm(channels)
#         self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
#         self.linear_geglu_2 = nn.Linear(4 * channels, channels)

#         self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
#     def forward(self, x, cond=None):
#         residue_long = x
#         # 通道分组归一化
#         x = self.groupnorm(x)
#         x = self.conv_input(x)
        
#         n, c, h, w = x.shape
#         x = x.view((n, c, h * w))
#         x = x.transpose(-1, -2)  

#         residue_short = x
#         x = self.layernorm_1(x)
#         x = self.attention_1(x)
#         x += residue_short

#         # 只有当存在条件输入和attention_2时才执行交叉注意力
#         if self.attention_2 is not None and cond is not None:
#             residue_short = x
#             x = self.layernorm_2(x)
#             x = self.attention_2(x, cond)
#             x += residue_short

#         residue_short = x
#         x = self.layernorm_3(x)
#         x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
#         x = x * F.gelu(gate)
#         x = self.linear_geglu_2(x)
#         x += residue_short

#         x = x.transpose(-1, -2)
#         x = x.view((n, c, h, w))    # (n, c, h, w)  

#         return self.conv_output(x) + residue_long



                # box_cond_aspects = encoded_cond.squeeze(-1)  # [B, 8]
                # aspects = []
                # for i, proj in enumerate(self.aspect_projectors):
                #     aspect_input = box_cond_aspects[:, i:i+1]  # [B, 1] 保持batch维度
                #     aspect_output = proj(aspect_input)  # [B, 256]
                #     aspects.append(aspect_output) # (batch, 256)
                # aspects = torch.stack(aspects, dim=1)  # [B, 8, 256]
                # image_norm = F.normalize(image_features, p=2, dim=-1)
                # aspects_norm = F.normalize(aspects, p=2, dim=-1)
                # similarity_matrix = torch.bmm(aspects_norm, aspects_norm.transpose(1, 2))
                # batch_similarity_matrix.append(similarity_matrix)
                
                # image_expanded = image_norm.unsqueeze(1)
                # similarity_scores = torch.sum(image_expanded * aspects_norm, dim=-1)
                
                # if strategy == "gumbel":
                #     # Epoch 0-5: Gumbel Softmax with decreasing tau
                #     tau = max(0.5, 2.0 - (self.current_epoch / 5) * 1.5)
                #     selected_aspect, gumbel_scores = self.gumbel_selection(similarity_scores, aspects, tau=tau)
                # elif strategy == "mixed":
                #     # Epoch 6: 50%概率使用Gumbel, 50%使用硬选择
                #     if torch.rand(1).item() < 0.5:
                #         selected_aspect, gumbel_scores = self.gumbel_selection(similarity_scores, aspects, tau=0.3)
                #     else:
                #         selected_aspect, one_hot_selection = self.hard_selection(similarity_scores, aspects)
                # else:  # strategy == "hard"
                #     # Epoch 7-9: 纯硬选择 后续拓展维度可选top-k个
                #     selected_aspect, one_hot_selection = self.hard_selection(similarity_scores, aspects)
                
                # box_cond = selected_aspect
                
                
                
                # encoded_cond = self.cond_encoder(box_cond.unsqueeze(-1))  # [B, 8, 1]
                # encoded_cond = encoded_cond.squeeze(-1)  # [B, 8]
                
                # # 生成Q, K, V
                # Q = self.query_proj(image_features)      # [B, feature_dim] - GT作为查询
                # K = self.key_proj(encoded_cond)   # [B, feature_dim] - 复杂特征作为键
                # V = self.value_proj(encoded_cond) # [B, feature_dim] - 复杂特征作为值
                
                # b = encoded_cond.size(0)
                # # 重塑为多头 [B, num_heads, head_dim]
                # Q = Q.view(b, self.num_heads, self.head_dim)
                # K = K.view(b, self.num_heads, self.head_dim)
                # V = V.view(b, self.num_heads, self.head_dim)
                
                # # 计算注意力分数
                # attention_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, num_heads, 1]
                # attention_weights = F.softmax(attention_scores, dim=-1)
                # attention_weights = self.dropout(attention_weights)
                
                # # 应用注意力
                # attended_values = torch.matmul(attention_weights, V)  # [B, num_heads, head_dim]
                
                # # 合并多头
                # attended_values = attended_values.view(b, self.feature_dim)
                
                # # 输出投影
                # output = self.output_proj(attended_values)
                
                # # 残差连接 + 层归一化
                # encoded_cond = self.layer_norm(encoded_cond + output)
                
                # batch_box_cond.append(encoded_cond)
                # box_cond = encoded_cond
                # box_cond = image_features
                
                
                
                
                #########提取与输入相关的cond          
                # x_flat = gt_features[:,:-1,:,:] #[:,-1:,:,:]
                # # with torch.no_grad():
                # image_features = self.cond_fc_layers(x_flat.view(x_flat.size(0), -1))  # [B, 256]
                # # batch_image_features.append(image_features)
                # box_cond = image_features
                
                
                ############尝试用输入转化为特征作为cond看看有没有用
                # input_dim = 10 * 24 * 56  # 1344
                # self.cond_fc_layers = nn.Sequential(
                #     nn.Linear(input_dim, 1024),
                #     nn.ReLU(inplace=True),
                #     nn.Dropout(0.3),
                #     nn.Linear(1024, 512),
                #     nn.ReLU(inplace=True),
                #     nn.Dropout(0.2),
                #     nn.Linear(512, 256)
                # )
                
                
        #                 if 'similarity_matrix' in output_dict.keys() and len(output_dict['similarity_matrix']) > 0:
        #     diversity_loss = 0
        #     batch_similarity_matrix = output_dict['similarity_matrix']
        #     for batch_idx in range(len(batch_similarity_matrix)):
        #         similarity_matrix = batch_similarity_matrix[batch_idx]
        #         num_aspects = similarity_matrix.shape[1]
        #         # 鼓励非对角线元素接近0（不同aspect应该不相似）
        #         mask = torch.eye(num_aspects).to(similarity_matrix.device)
        #         off_diagonal = similarity_matrix * (1 - mask)
        #         diversity_loss += torch.mean(torch.abs(off_diagonal))
        #     diversity_loss_avg = diversity_loss / len(batch_similarity_matrix)
        #     total_loss += diversity_loss_avg
        #     self.diversity_loss = diversity_loss_avg.item()
        #     self.total_loss = total_loss.item()
        #     self.loss_dict.update({'total_loss': self.total_loss,
        #                         'diversity_loss': self.diversity_loss})
            
        # if 'image_features' in output_dict.keys() and len(output_dict['image_features']) > 0:
        #     alignment_loss = 0
        #     mse_loss = 0
        #     cosine_loss = 0

        #     batch_image_features = output_dict['image_features']
        #     batch_box_cond = output_dict['box_cond']
        #     for batch_idx in range(len(batch_image_features)):
        #         image_features = batch_image_features[batch_idx]
        #         box_cond = batch_box_cond[batch_idx]

        #         # 1. MSE损失
        #         mse_loss = F.mse_loss(image_features, box_cond)
                
        #         # 2. 余弦相似度损失
        #         image_norm = F.normalize(image_features, p=2, dim=-1)
        #         box_norm = F.normalize(box_cond, p=2, dim=-1)
        #         cos_sim = F.cosine_similarity(image_norm, box_norm, dim=-1)
        #         cosine_loss = (1 - cos_sim.mean())
                
        #         combined_loss = 0.5 * mse_loss + 0.5 * cosine_loss
        #         alignment_loss += combined_loss
                
        #     alignment_loss_avg = alignment_loss / len(batch_image_features)
        #     total_loss += alignment_loss_avg
        #     self.alignment_loss = alignment_loss_avg.item()
        #     self.total_loss = total_loss.item()
        #     self.loss_dict.update({'total_loss': self.total_loss,
        #                         'alignment_loss': self.alignment_loss})
 
 
 
# def normalize_statistical_features(features):
#     """
#         使用语义一致的归一化，确保：
#         1. 零值在所有通道都有一致的表示
#         2. 归一化方式反映数据的实际意义
#     """
#     norm_features = features.clone()
#     params = {}
    
#      # 检查是否是单通道输入
#     if features.shape[1] == 1:
#         # 单通道情况 - 将其视为点数通道
#         points = features[:, 0]  # 形状为 [B,H,W]
#         zero_mask = (points == 0)
        
#         # 点数映射：0 -> -1, max -> +1, 其他按比例
#         if torch.max(points) > 0:
#             log_points = torch.log1p(points)  # log(1+x)
#             max_log = log_points.max()
#             params['max_log_points'] = max_log.item()
            
#             if max_log > 0:
#                 # 归一化到[-1, 1]，确保0点映射到-1
#                 normalized = -1.0 + 2.0 * log_points / max_log
                
#                 # 额外检查确保0点确实映射到-1
#                 normalized[zero_mask] = -1.0
                
#                 # 保存归一化结果
#                 norm_features[:, 0] = normalized
#             else:
#                 # 如果log后最大值为0
#                 norm_features[:, 0] = torch.zeros_like(points)
#                 norm_features[:, 0][zero_mask] = -1.0
#         else:
#             # 所有点数都是0
#             norm_features[:, 0] = torch.full_like(points, -1.0)
        
#         return norm_features, params
    
#     # === 几何特征（前3个通道）===
#     for i in range(3):
#         channel = features[:, i]
        
#         # 找出0和非0值
#         zero_mask = (channel == 0)
#         non_zero = channel[~zero_mask]
        
#         if len(non_zero) > 0:
#             # 对非零值应用标准化
#             min_val = non_zero.min()
#             max_val = non_zero.max()
#             params[f'min_{i}'] = min_val.item()
#             params[f'max_{i}'] = max_val.item()
            
#             # 如果有足够的数值范围，进行归一化
#             if max_val - min_val > 1e-10:
#                 # 创建临时张量存储归一化结果
#                 normalized = torch.zeros_like(channel)
                
#                 # 只对非零值应用归一化，确保0映射到0（中间值）
#                 normalized[~zero_mask] = -1.0 + 2.0 * (non_zero - min_val) / (max_val - min_val)
                
#                 # 确保0值映射到特定值
#                 normalized[zero_mask] = -1
                
#                 norm_features[:, i] = normalized
#             else:
#                 # 如果非零值范围很小，简化处理
#                 normalized = torch.zeros_like(channel)
#                 normalized[~zero_mask] = 0.5  # 非零值映射到一个适中的正值
#                 norm_features[:, i] = normalized
#         else:
#             # 全是0的情况
#             norm_features[:, i] = torch.zeros_like(channel)
    
#     # === 点数通道（第4个通道）===
#     points = features[:, 3]
#     zero_mask = (points == 0)
    
#     # 点数映射：0 -> -1, max -> +1, 其他按比例
#     if torch.max(points) > 0:
#         log_points = torch.log1p(points)  # log(1+x)
#         max_log = log_points.max()
#         params['max_log_points'] = max_log.item()
        
#         if max_log > 0:
#             # 归一化到[-1, 1]，确保0点映射到-1
#             norm_features[:, 3] = -1.0 + 2.0 * log_points / max_log
            
#             # 额外检查确保0点确实映射到-1
#             if not torch.all(norm_features[zero_mask, 3] == -1.0):
#                 norm_features[zero_mask, 3] = -1.0
#     else:
#         # 所有点数都是0
#         norm_features[:, 3] = torch.full_like(points, -1.0)
    
#     return norm_features, params

# def denormalize_statistical_features(normalized_features, normalization_params):
#     """
#     将语义一致的归一化特征反归一化回原始范围
    
#     输入:
#         normalized_features [M, 4] - 归一化到[-1, 1]范围的特征
#         normalization_params - 归一化时保存的参数字典
    
#     输出:
#         反归一化后的特征 [M, 4]
#     """
#     # 复制一份特征，避免修改原始数据
#     denormalized_features = normalized_features.clone()
    
#     # 检查是否是单通道输入
#     if normalized_features.shape[1] == 1:
#         # 单通道情况 - 将其视为点数通道
#         points_normalized = normalized_features[:, 0]  # 形状为 [B,H,W]
        
#         if 'max_log_points' in normalization_params:
#             max_log = normalization_params['max_log_points']
            
#             # 找出映射到-1的零值（原始点数为0的位置）
#             zero_points_mask = (points_normalized <= -0.99)  # 允许一点数值误差
            
#             # 对非零点数进行反归一化
#             if not torch.all(zero_points_mask):
#                 # 将[-1,1]映射回[0,max_log]的log空间
#                 log_points = ((points_normalized + 1.0) / 2.0) * max_log
                
#                 # 从log空间映射回原始点数
#                 denorm_points = torch.expm1(log_points)  # exp(x)-1是log1p的逆运算
                
#                 # 保存反归一化结果
#                 denormalized_features[:, 0] = denorm_points
                
#                 # 确保原始零点数被准确恢复
#                 denormalized_features[:, 0][zero_points_mask] = 0.0
#             else:
#                 # 所有值都是零点
#                 denormalized_features[:, 0] = torch.zeros_like(points_normalized)
#         else:
#             # 如果没有参数，将所有-1值视为0点数
#             denormalized_features[:, 0] = torch.zeros_like(points_normalized)
#             denormalized_features[:, 0][points_normalized > -0.99] = 1.0  # 非-1值设为至少1个点
        
#         return denormalized_features
    
#     # === 几何特征通道（前3个通道）===
#     for i in range(3):
#         channel = normalized_features[:, i]
        
#         # 识别不同的值区域
#         zero_value_mask = (channel == -1.0)  # 被映射到-1.0的原始零值
        
#         # 检查是否有归一化参数
#         if f'min_{i}' in normalization_params and f'max_{i}' in normalization_params:
#             min_val = normalization_params[f'min_{i}']
#             max_val = normalization_params[f'max_{i}']
            
#             # 检查是否有足够的数值范围
#             if max_val - min_val > 1e-10:
#                 # 创建临时张量存储反归一化结果
#                 denorm_channel = torch.zeros_like(channel)
                
#                 # 将表示原始零值的-1.0映射回0
#                 denorm_channel[zero_value_mask] = 0.0
                
#                 # 将非零值（现在在[-1,1]之外的值）映射回原始范围
#                 non_zero_mask = ~zero_value_mask
#                 if non_zero_mask.any():
#                     # 从[-1,1]反归一化到原始范围
#                     denorm_channel[non_zero_mask] = min_val + (
#                         (channel[non_zero_mask] + 1.0) / 2.0) * (max_val - min_val)
                
#                 denormalized_features[:, i] = denorm_channel
#             else:
#                 # 如果原始数据范围很小
#                 denormalized_features[:, i] = torch.zeros_like(channel)
#                 denormalized_features[:, i][~zero_value_mask] = min_val
#         else:
#             # 如果没有提供参数，保持0值
#             denormalized_features[:, i] = torch.zeros_like(channel)
    
#     # === 点数通道（第4个通道）===
#     if 'max_log_points' in normalization_params:
#         max_log = normalization_params['max_log_points']
        
#         # 从[-1,1]映射回原始点数
#         points_normalized = normalized_features[:, 3]
        
#         # 找出映射到-1的零值（原始点数为0的位置）
#         zero_points_mask = (points_normalized <= -0.99)  # 允许一点数值误差
        
#         # 对非零点数进行反归一化
#         if not torch.all(zero_points_mask):
#             # 将[-1,1]映射回[0,max_log]的log空间
#             log_points = ((points_normalized + 1.0) / 2.0) * max_log
            
#             # 从log空间映射回原始点数
#             denormalized_features[:, 3] = torch.expm1(log_points)  # exp(x)-1是log1p的逆运算
        
#         # 确保原始零点数被准确恢复
#         denormalized_features[:, 3][zero_points_mask] = 0.0
#     else:
#         # 如果没有参数，将所有-1值视为0点数
#         denormalized_features[:, 3] = torch.zeros_like(normalized_features[:, 3])
#         denormalized_features[:, 3][normalized_features[:, 3] > -0.99] = 1.0  # 非-1值设为至少1个点
    
#     return denormalized_features


    # # 分别处理xyz坐标 (前3个通道)
    # spatial_range = torch.tensor([5.63,2.36,1.98], device=features.device).view(1, 3, 1, 1)  # [x_range, y_range, z_range]
    # norm_features[:, :3] = features[:, :3] / (spatial_range / 2.0)  # 归一化到[-1, 1]
    # norm_features[:, :3] = torch.clamp(norm_features[:, :3], -1.0, 1.0)
    
    # # 使用相同的spatial_range
    # spatial_range = torch.tensor([5.63, 2.36, 1.98], device=normalized_features.device).view(1, 3, 1, 1)
    # denormalized_features[:, :3] = normalized_features[:, :3] * (spatial_range / 2.0)
    
    
    
            # batch_max_log_std = model(batch_data['ego'])
            # if batch_max_log_std is None:
            #     continue
            # # 更新全局最大值
            # for j in range(3):
            #     global_max_log_std[j] = max(global_max_log_std[j], batch_max_log_std[j])
                
            #  # 可选：定期打印当前全局最大值
            # if i % 1000 == 0:
            #     print(f"当前全局max_log_std: {global_max_log_std}")
            
                        
             # === 在这里计算统计特征 ===
             # 计算当前batch的max_log_std（前3个维度）
            # batch_max_log_std = [0.0, 0.0, 0.0]  # 3个维度的最大值

            # for batch_idx in range(len(batch_dict['batch_gt_spatial_features'])):  # 遍历8个batch
            #     batch_features = batch_dict['batch_gt_spatial_features'][batch_idx]  # [N, C, H, W]
                
            #     for i in range(3):  # 前3个通道
            #         std_channel = batch_features[:, i]  # [N, H, W]
            #         valid_mask = std_channel > 0
                    
            #         if valid_mask.any():
            #             valid_std = std_channel[valid_mask]
            #             log_std = torch.log1p(valid_std)
            #             current_max = log_std.max().item()
            #             batch_max_log_std[i] = max(batch_max_log_std[i], current_max)
            
            
            
#             def normalize_statistical_features(features):
#     """
#         使用语义一致的归一化，确保：
#         1. 零值在所有通道都有一致的表示
#         2. 归一化方式反映数据的实际意义
#     """
#     norm_features = features.clone()
#     params = {
#         'max_log_std_0': 0.0488,
#         'max_log_std_1': 0.0488, 
#         'max_log_std_2': 0.6762,
#         'max_log_points': 2.8332
#     }
    
#      # 检查是否是单通道输入
#     if features.shape[1] == 1:
#         # 单通道情况 - 将其视为点数通道
#         points = features[:, 0]  # 形状为 [B,H,W]
#         zero_mask = (points == 0)
        
#         # 点数映射：0 -> -1, max -> +1, 其他按比例
#         if torch.max(points) > 0:
#             log_points = torch.log1p(points)  # log(1+x)
#             max_log = log_points.max()
#             params['max_log_points'] = max_log.item()
            
#             if max_log > 0:
#                 # 归一化到[-1, 1]，确保0点映射到-1
#                 normalized = -1.0 + 2.0 * log_points / max_log
                
#                 # 额外检查确保0点确实映射到-1
#                 normalized[zero_mask] = -1.0
                
#                 # 保存归一化结果
#                 norm_features[:, 0] = normalized
#             else:
#                 # 如果log后最大值为0
#                 norm_features[:, 0] = torch.zeros_like(points)
#                 norm_features[:, 0][zero_mask] = -1.0
#         else:
#             # 所有点数都是0
#             norm_features[:, 0] = torch.full_like(points, -1.0)
        
#         return norm_features, params
    

#     # === 几何特征（前3个通道）===
#     for i in range(3):
#         std_channel = features[:, i]
#         points_channel = features[:, 3]
        
#         # 只区分两种情况：
#         invalid_mask = (points_channel == 0) | (std_channel == -1.0)  # 无效情况（点数0或单点）
#         zero_std_mask = (std_channel == 0.0) & (points_channel > 1)   # 真实方差0（多点重合）
#         positive_mask = (std_channel > 0.0)                           # 正常正值
        
#         if positive_mask.any():
#             log_std = torch.log1p(std_channel[positive_mask])
#             max_log_std = params[f'max_log_std_{i}']
            
#             if max_log_std > 0:
#                 normalized_vals = -0.5 + 1.5 * log_std / max_log_std  # 映射到(-0.5, 1]
#                  # 确保不会超出范围
#                 normalized_vals = torch.clamp(normalized_vals, max=1.0)
#                 norm_features[positive_mask, i] = normalized_vals
        
#         # 简化的特殊值映射：
#         norm_features[invalid_mask, i] = -1.0      # 所有无效情况 -> -1
#         norm_features[zero_std_mask, i] = -0.5     # 真实方差0 -> -0.5
    
#     # === 点数通道（第4个通道）===
#     points = features[:, 3]
#     zero_mask = (points == 0)
    
#     # 点数映射：1 -> -1, max -> +1, 其他按比例
#     if torch.max(points) > 0:
#         log_points = torch.log1p(points)  # log(1+x)
#         max_log = params['max_log_points']
        
#         if max_log > 0:
#             # 归一化到[-1, 1]，确保0点映射到-1
#             norm_features[:, 3] = -1.0 + 2.0 * log_points / max_log
            
#             # 额外检查确保0点确实映射到-1
#             if not torch.all(norm_features[zero_mask, 3] == -1.0):
#                 norm_features[zero_mask, 3] = -1.0
#     else:
#         # 所有点数都是0
#         norm_features[:, 3] = torch.full_like(points, -1.0)
    
#     return norm_features, params

# def denormalize_statistical_features(normalized_features, normalization_params):
#     """
#     将语义一致的归一化特征反归一化回原始范围
    
#     输入:
#         normalized_features [M, 4] - 归一化到[-1, 1]范围的特征
#         normalization_params - 归一化时保存的参数字典
    
#     输出:
#         反归一化后的特征 [M, 4]
#     """
#     # 复制一份特征，避免修改原始数据
#     denormalized_features = normalized_features.clone()
    
#     # 检查是否是单通道输入
#     if normalized_features.shape[1] == 1:
#         # 单通道情况 - 将其视为点数通道
#         points_normalized = normalized_features[:, 0]  # 形状为 [B,H,W]
        
#         if 'max_log_points' in normalization_params:
#             max_log = normalization_params['max_log_points']
            
#             # 找出映射到-1的零值（原始点数为0的位置）
#             zero_points_mask = (points_normalized <= -0.99)  # 允许一点数值误差
            
#             # 对非零点数进行反归一化
#             if not torch.all(zero_points_mask):
#                 # 将[-1,1]映射回[0,max_log]的log空间
#                 log_points = ((points_normalized + 1.0) / 2.0) * max_log
                
#                 # 从log空间映射回原始点数
#                 denorm_points = torch.expm1(log_points)  # exp(x)-1是log1p的逆运算
                
#                 # 保存反归一化结果
#                 denormalized_features[:, 0] = denorm_points
                
#                 # 确保原始零点数被准确恢复
#                 denormalized_features[:, 0][zero_points_mask] = 0.0
#             else:
#                 # 所有值都是零点
#                 denormalized_features[:, 0] = torch.zeros_like(points_normalized)
#         else:
#             # 如果没有参数，将所有-1值视为0点数
#             denormalized_features[:, 0] = torch.zeros_like(points_normalized)
#             denormalized_features[:, 0][points_normalized > -0.99] = 1.0  # 非-1值设为至少1个点
        
#         return denormalized_features
    
#      # === 反归一化标准差特征（前3个通道）===
    
#     # === 反归一化前3个通道（标准差）===
#     for i in range(3):
#         normalized = normalized_features[:, i]
#         max_log_std = normalization_params[f'max_log_std_{i}']
        
#         # 识别特殊值(使用适当容差)
#         invalid_mask = normalized < -0.9           # 小于-0.9的为无效标记
#         zero_mask = torch.abs(normalized + 0.5) < 0.05  # 接近-0.5的为真实0
#         normal_mask = normalized > -0.45           # 大于-0.45的为正常值
        
#         # 初始化输出
#         denormalized_features[:, i] = 0.0
        
#         # 反归一化正常值
#         if normal_mask.any() and max_log_std > 0:
#             # 逆变换: normalized = -0.5 + 1.5 * log_std / max_log_std
#             # 所以: log_std = (normalized + 0.5) * max_log_std / 1.5
#             log_std = (normalized[normal_mask] + 0.5) * max_log_std / 1.5
#             log_std = torch.clamp(log_std, min=0.0)  # 确保非负
#             std_vals = torch.expm1(log_std)
#             std_vals = torch.clamp(std_vals, min=0.0)  # 确保非负
#             denormalized_features[normal_mask, i] = std_vals
        
#     # === 点数通道（第4个通道）===
#     if 'max_log_points' in normalization_params:
#         max_log = normalization_params['max_log_points']
        
#         # 从[-1,1]映射回原始点数
#         points_normalized = normalized_features[:, 3]
        
#         # 找出映射到-1的零值（原始点数为0的位置）
#         zero_points_mask = (points_normalized <= -0.99)  # 允许一点数值误差
        
#         # 对非零点数进行反归一化
#         if not torch.all(zero_points_mask):
#             # 将[-1,1]映射回[0,max_log]的log空间
#             log_points = ((points_normalized + 1.0) / 2.0) * max_log
            
#             # 从log空间映射回原始点数
#             denormalized_features[:, 3] = torch.expm1(log_points)  # exp(x)-1是log1p的逆运算
        
#         # 确保原始零点数被准确恢复
#         denormalized_features[:, 3][zero_points_mask] = 0.0
#     else:
#         # 如果没有参数，将所有-1值视为0点数
#         denormalized_features[:, 3] = torch.zeros_like(normalized_features[:, 3])
#         denormalized_features[:, 3][normalized_features[:, 3] > -0.99] = 1.0  # 非-1值设为至少1个点
    
#     return denormalized_features



#  # === 在这里计算统计特征 ===
#             # 计算当前batch的max_log值
#             current_batch_max_absmean = [0.0, 0.0, 0.0]  # 当前batch绝对偏移平均值3个维度的最大值
#             current_batch_max_dist = 0.0  # 当前batch最大距离的最大值

#             # 遍历batch中的每个样本
#             for batch_idx in range(len(batch_dict['batch_gt_spatial_features'])):
#                 batch_features = batch_dict['batch_gt_spatial_features'][batch_idx]  # [N, C, H, W]
                
#                 # 统计绝对偏移平均值（前3个通道，索引0-2）
#                 for channel_idx in range(3):
#                     absmean_channel = batch_features[:, channel_idx]  # [N, H, W]
#                     valid_mask = absmean_channel > 0
                    
#                     if valid_mask.any():
#                         valid_absmean = absmean_channel[valid_mask]
#                         log_absmean = torch.log1p(valid_absmean)  # log(1+x)
#                         current_max = log_absmean.max().item()
#                         current_batch_max_absmean[channel_idx] = max(current_batch_max_absmean[channel_idx], current_max)
                
#                 # 统计最大距离（第6个通道，索引6）
#                 max_dist_channel = batch_features[:, 6]  # [N, H, W]
#                 valid_mask = max_dist_channel > 0
                
#                 if valid_mask.any():
#                     valid_max_dist = max_dist_channel[valid_mask]
#                     log_max_dist = torch.log1p(valid_max_dist)  # log(1+x)
#                     current_max = log_max_dist.max().item()
#                     current_batch_max_dist = max(current_batch_max_dist, current_max)


    
    # batch_max_log_absmean = [0, 0, 0]  # 用于记录每个通道的最大值
    # batch_max_log_max_dist = 0  # 用于记录每个batch的最大值
    
    # current_batch_max_absmean, current_batch_max_dist= model(batch_data['ego'])
    #         if current_batch_max_absmean is None or current_batch_max_dist is None:
    #             continue
    #         # 更新全局最大值
    #         for j in range(3):
    #             batch_max_log_absmean[j] = max(batch_max_log_absmean[j], current_batch_max_absmean[j])
            
    #         batch_max_log_max_dist = max(batch_max_log_max_dist, current_batch_max_dist)
                        
    #         # 可选：定期打印当前全局最大值
    #         if i % 1000 == 0:
    #             print(f"Step {i}:")
    #             print(f"  当前全局max_log_absmean: {batch_max_log_absmean}")
    #             print(f"  当前全局max_log_max_dist: {batch_max_log_max_dist}")
    
    
    # # 训练结束后打印最终结果
    # print("="*50)
    # print("最终统计结果:")
    # print(f"max_log_mean_offset_0: {batch_max_log_absmean[0]:.6f}")
    # print(f"max_log_mean_offset_1: {batch_max_log_absmean[1]:.6f}")
    # print(f"max_log_mean_offset_2: {batch_max_log_absmean[2]:.6f}")
    # print(f"max_log_max_dist: {batch_max_log_max_dist:.6f}")
    # print("="*50)
    
    # self.dete_convertor = nn.Sequential(
    #                 nn.LayerNorm(hidden_dim),
    #                 nn.Linear(hidden_dim, hidden_dim * 2),  
    #                 nn.GELU(),
    #                 nn.Dropout(0.1),
    #                 nn.Linear(hidden_dim * 2, hidden_dim),  
    #                 nn.GELU(),
    #                 nn.Linear(hidden_dim, hidden_dim)      
    #             )
    
    # 没什么用
    # self.dete_convertor = nn.Sequential(
    #                 nn.LayerNorm(hidden_dim),
    #                 # 大容量变换，充分重构特征
    #                 nn.Linear(hidden_dim, hidden_dim * 8),
    #                 nn.GELU(),
    #                 nn.Dropout(0.15),  # 稍高的dropout防止过拟合
    #                 nn.Linear(hidden_dim * 8, hidden_dim * 8),
    #                 nn.GELU(),
    #                 nn.Dropout(0.15),
    #                 nn.Linear(hidden_dim * 8, hidden_dim * 4),
    #                 nn.GELU(),
    #                 nn.Dropout(0.1),
    #                 nn.Linear(hidden_dim * 4, hidden_dim * 2),
    #                 nn.GELU(),
    #                 nn.Dropout(0.1),
    #                 nn.Linear(hidden_dim * 2, hidden_dim),
    #                 nn.GELU(),
    #                 nn.Linear(hidden_dim, hidden_dim)
    #             )
    
    # if mask.sum() == 0:
            #     print("no object!!!")
            #     processed_lidar = None
            # else:    
            #     # 5. 将点云数据转换为体素/BEV/降采样点云
            #     gt_boxes = object_bbx_center[mask.astype(bool)]
            #     pc_range = [-140.8, -40, -3, 140.8, 40, 1]
            #     visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/origin_gt_boxes_project.png")
            #     # 将gt扩展到相同大小 3:6 hwl
            #     gt_boxes[:, 3:6] = np.array(self.max_hwl)
            #     visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/expend_gt_boxes_project.png")        
            #     # 获取gt框中的点云  不能使用gpu版与dataloader多线程有关
            #     point_indices = points_in_boxes_cpu(projected_lidar_stack[:, :3], gt_boxes[:,[0, 1, 2, 5, 4, 3, 6]]) 
            #     gt_voxel_stack = []
            #     gt_coords_stack = []
            #     gt_num_points_stack = []
            #     gt_masks = []
            #     rotation_angles = -gt_boxes[:, 6].astype(float)
            #     for car_idx in range(len(gt_boxes)):
            #         # 获取当前box中的点并平移到以box中心为原点的坐标系
            #         gt_point = projected_lidar_stack[point_indices[car_idx] > 0]
            #         gt_point[:, :3] -= gt_boxes[car_idx][0:3]
            #         gt_boxes[car_idx][0:3] = [0, 0, 0]
            #         pc_range = [-15, -15, -1, 15, 15, 1]
            #         visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_expand_{car_idx}.png",scale_bev=10)
            #         # 旋转点云 
            #         gt_point = common_utils.rotate_points_along_z(gt_point[np.newaxis, :, :], np.array([rotation_angles[car_idx]]))[0]
            #         gt_boxes[car_idx][0:3] = common_utils.rotate_points_along_z(gt_boxes[car_idx][np.newaxis, np.newaxis, 0:3], np.array([-float(gt_boxes[car_idx][6])]))[0,0]
            #         gt_boxes[car_idx][6] -= float(gt_boxes[car_idx][6])
            #         visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_rotate_{car_idx}.png",scale_bev=10)
            #         # 体素化 不能并行！！
            #         processed_lidar_car = self.pre_processor.preprocess(gt_point, is_car=True)
            #         gt_voxel_stack.append(processed_lidar_car['voxel_features'])
            #         gt_coords_stack.append(processed_lidar_car['voxel_coords'])
            #         gt_num_points_stack.append(processed_lidar_car['voxel_num_points'])
            #         gt_masks.append(np.full(processed_lidar_car['voxel_features'].shape[0], car_idx, dtype=np.int32))
            #     processed_lidar = {
            #         'voxel_features': np.concatenate(gt_voxel_stack, axis=0),
            #         'voxel_coords': np.concatenate(gt_coords_stack, axis=0),
            #         'voxel_num_points': np.concatenate(gt_num_points_stack, axis=0),
            #         'gt_masks': np.concatenate(gt_masks, axis=0),
            #         }
                