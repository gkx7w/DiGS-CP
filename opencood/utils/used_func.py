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
 