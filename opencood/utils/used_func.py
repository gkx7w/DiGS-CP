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