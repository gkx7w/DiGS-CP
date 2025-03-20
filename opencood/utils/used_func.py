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
    