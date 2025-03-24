# early fusion dataset
import torch
import numpy as np
from opencood.utils.pcd_utils import downsample_lidar_minimum
import math
from collections import OrderedDict
import copy
from opencood.utils import box_utils
from opencood.utils.common_utils import merge_features_to_dict
from opencood.data_utils.post_processor import build_postprocessor
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.heter_utils import AgentSelector
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_cpu
from opencood.utils import common_utils
# from opencood.visualization.my_vis import visualize
from opencood.visualization.simple_vis import visualize
from opencood.utils.box_utils import boxes_to_corners_3d

def getDecdiffusionFusionDataset(cls):
    class DecdiffusionFusionDataset(cls):
        """
        This dataset is used for early fusion, where each CAV transmit the raw
        point cloud to the ego vehicle.
        """
        def __init__(self, params, visualize, train=True):
            super(DecdiffusionFusionDataset, self).__init__(params, visualize, train)
            self.supervise_single = True if ('supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                                        else False
            assert self.supervise_single is False
            self.proj_first = False if 'proj_first' not in params['fusion']['args']\
                                         else params['fusion']['args']['proj_first']
            self.max_hwl = params['model']['args']['max_hwl']
            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

            self.heterogeneous = False
            if 'heter' in params:
                self.heterogeneous = True
                self.selector = AgentSelector(params['heter'], self.max_cav)

        def __getitem__(self, idx):
            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict,self.params['noise_setting'])


            processed_data_dict = OrderedDict()
            processed_data_dict['ego'] = {}

            ego_id = -1
            ego_lidar_pose = []
            ego_cav_base = None

            # first find the ego vehicle's lidar pose
            for cav_id, cav_content in base_data_dict.items():
                if cav_content['ego']:
                    ego_id = cav_id
                    ego_lidar_pose = cav_content['params']['lidar_pose']
                    ego_cav_base = cav_content
                    break

            assert cav_id == list(base_data_dict.keys())[
                0], "The first element in the OrderedDict must be ego"
            assert ego_id != -1
            assert len(ego_lidar_pose) > 0

            projected_lidar_stack = []
            object_stack = []
            object_id_stack = []
            dec_processed_features = []
            vsa_lidar_stack = []
            vsa_lidar_stack_project = []
            vsa_lidar_stack_noproject = []
            single_label_list = []
            too_far = []
            cav_id_list = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            # 1. 收集所有车辆的点云数据
            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # 2. 检查车辆是否在通信范围内
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue
                # 3. 处理单个车辆的点云数据
                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)
                # all these lidar and object coordinates are projected to ego
                # already. 
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']
                single_label_list.append(selected_cav_processed['single_label_dict'])
                cav_id_list.append(cav_id)
                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                dec_processed_features.append(
                    selected_cav_processed['processed_features'])
                if self.proj_first:
                    vsa_lidar_stack.append(selected_cav_processed['projected_lidar'])
                else:
                    vsa_lidar_stack.append(selected_cav_processed['no_projected_lidar'])

                vsa_lidar_stack_project.append(selected_cav_processed['projected_vsa_lidar'])
                vsa_lidar_stack_noproject.append(selected_cav_processed['no_projected_lidar'])

            for cav_id in too_far:
                base_data_dict.pop(cav_id)
            
            pairwise_t_matrix = \
                get_pairwise_transformation(base_data_dict,
                                                self.max_cav,
                                                self.proj_first)

            lidar_poses = np.array(lidar_pose_list).reshape(-1, 6)  # [N_cav, 6]
            lidar_poses_clean = np.array(lidar_pose_clean_list).reshape(-1, 6)  # [N_cav, 6]
            
            # merge preprocessed features from different cavs into the same dict
            cav_num = len(cav_id_list)
            
            # generate single view label (no coop) label
            label_dict_no_coop = single_label_list # [{cav1_label}, {cav2_label}...]

            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]
            object_stack = np.vstack(object_stack)
            object_stack = object_stack[unique_indices]
            
            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1
            # decople
            merged_feature_dict = merge_features_to_dict(dec_processed_features)
            processed_data_dict['ego'].update({'dec_processed_lidar': merged_feature_dict,
                                                'vsa_lidar': vsa_lidar_stack,
                                                'vsa_lidar_project': vsa_lidar_stack_project,
                                                'vsa_lidar_noproject': vsa_lidar_stack_noproject})

            # 4. 将所有车辆的点云数据堆叠在一起（完成点云融合）
            # convert list to numpy array, (N, 4)
            projected_lidar_stack = np.vstack(projected_lidar_stack)

            # data augmentation 我应该不能要数据增强，不然对点云有影响，不过gt也一起增强，应该就对了
            projected_lidar_stack, object_bbx_center, mask = \
                self.augment(projected_lidar_stack, object_bbx_center, mask)

            # we do lidar filtering in the stacked lidar
            projected_lidar_stack = mask_points_by_range(projected_lidar_stack,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'])
            # augmentation may remove some of the bbx out of range
            object_bbx_center_valid = object_bbx_center[mask == 1]
            object_bbx_center_valid, range_mask = \
                box_utils.mask_boxes_outside_range_numpy(object_bbx_center_valid,
                                                        self.params['preprocess'][
                                                            'cav_lidar_range'],
                                                        self.params['postprocess'][
                                                            'order'],
                                                        return_mask=True
                                                        )
            mask[object_bbx_center_valid.shape[0]:] = 0
            object_bbx_center[:object_bbx_center_valid.shape[0]] = \
                object_bbx_center_valid
            object_bbx_center[object_bbx_center_valid.shape[0]:] = 0
            unique_indices = list(np.array(unique_indices)[range_mask])

            if mask.sum() == 0:
                print("no object!!!")
                processed_lidar = None
            else:    
                # 5. 将点云数据转换为体素/BEV/降采样点云
                gt_boxes = object_bbx_center[mask.astype(bool)]
                # pc_range = [-140.8, -40, -3, 140.8, 40, 1]
                # visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/origin_gt_boxes.png")
                # 将gt扩展到相同大小 3:6 hwl
                gt_boxes[:, 3:6] = np.array(self.max_hwl)
                # visualize_gt_boxes(gt_boxes, projected_lidar_stack, pc_range, "/home/ubuntu/Code2/opencood/vis_output/expend_gt_boxes.png")        
                # 获取gt框中的点云  不能使用gpu版与dataloader多线程有关
                point_indices = points_in_boxes_cpu(projected_lidar_stack[:, :3], gt_boxes[:,[0, 1, 2, 5, 4, 3, 6]]) 
                gt_voxel_stack = []
                gt_coords_stack = []
                gt_num_points_stack = []
                gt_masks = []
                rotation_angles = -gt_boxes[:, 6].astype(float)
                for car_idx in range(len(gt_boxes)):
                    # 获取当前box中的点并平移到以box中心为原点的坐标系
                    gt_point = projected_lidar_stack[point_indices[car_idx] > 0]
                    gt_point[:, :3] -= gt_boxes[car_idx][0:3]
                    # gt_boxes[car_idx][0:3] = [0, 0, 0]
                    # pc_range = [-15, -15, -1, 15, 15, 1]
                    # visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_expand_{car_idx}.png",scale_bev=10)
                    # 旋转点云 
                    gt_point = common_utils.rotate_points_along_z(gt_point[np.newaxis, :, :], np.array([rotation_angles[car_idx]]))[0]
                    gt_boxes[car_idx][0:3] = common_utils.rotate_points_along_z(gt_boxes[car_idx][np.newaxis, np.newaxis, 0:3], np.array([-float(gt_boxes[car_idx][6])]))[0,0]
                    gt_boxes[car_idx][6] -= float(gt_boxes[car_idx][6])
                    # visualize_gt_boxes(gt_boxes[car_idx][np.newaxis, :], gt_point, pc_range, f"/home/ubuntu/Code2/opencood/vis_output/gt_rotate_{car_idx}.png",scale_bev=10)
                    # 体素化 不能并行！！
                    processed_lidar_car = self.pre_processor.preprocess(gt_point, is_car=True)
                    gt_voxel_stack.append(processed_lidar_car['voxel_features'])
                    gt_coords_stack.append(processed_lidar_car['voxel_coords'])
                    gt_num_points_stack.append(processed_lidar_car['voxel_num_points'])
                    gt_masks.append(np.full(processed_lidar_car['voxel_features'].shape[0], car_idx, dtype=np.int32))
                processed_lidar = {
                    'voxel_features': np.concatenate(gt_voxel_stack, axis=0),
                    'voxel_coords': np.concatenate(gt_coords_stack, axis=0),
                    'voxel_num_points': np.concatenate(gt_num_points_stack, axis=0),
                    'gt_masks': np.concatenate(gt_masks, axis=0),
                    }
                
            # generate the anchor boxes
            anchor_box = self.post_processor.generate_anchor_box()

            # generate targets label
            # 有点奇怪，怎么获得的标签
            label_dict_coop = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=anchor_box,
                    mask=mask)
            label_dict = {
                'stage1': label_dict_no_coop, # list 
                'stage2': label_dict_coop # dict
            }
            
            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                'object_bbx_mask': mask,
                'object_ids': [object_id_stack[i] for i in unique_indices],
                'anchor_box': anchor_box,
                'processed_lidar': processed_lidar,
                'label_dict': label_dict,
                'cav_num': cav_num,
                'pairwise_t_matrix': pairwise_t_matrix,
                'lidar_poses_clean': lidar_poses_clean,
                'lidar_poses': lidar_poses})

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                                                    projected_lidar_stack})
                
            processed_data_dict['ego'].update({'sample_idx': idx,
                                                'cav_id_list': cav_id_list})

            return processed_data_dict

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Project the lidar and bbx to ego space first, and then do clipping.

            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
            ego_pose : list
                The ego vehicle lidar pose under world coordinate.

            Returns
            -------
            selected_cav_processed : dict
                The dictionary contains the cav's processed information.
            """
            selected_cav_processed = {}
            ego_pose, ego_pose_clean = ego_cav_base['params']['lidar_pose'], ego_cav_base['params']['lidar_pose_clean']

            # calculate the transformation matrix
            transformation_matrix = \
                x1_to_x2(selected_cav_base['params']['lidar_pose'],
                        ego_pose)
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                        ego_pose_clean)

            # retrieve objects under ego coordinates
            object_bbx_center, object_bbx_mask, object_ids = \
                self.generate_object_center([selected_cav_base],
                                                        ego_pose)

            # filter lidar
            lidar_np = selected_cav_base['lidar_np']
            lidar_np = shuffle_points(lidar_np)
            # remove points that hit itself
            lidar_np = mask_ego_points(lidar_np)
            # no projected lidar
            no_project_lidar = copy.deepcopy(lidar_np)

            # project the lidar to ego space
            # x,y,z in ego space
            projected_lidar = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],transformation_matrix)
            
            vsa_project_lidar = copy.deepcopy(lidar_np)
            vsa_project_lidar[:, :3] = projected_lidar


            if self.proj_first: # 
                lidar_np[:, :3] = projected_lidar
            processed_lidar = self.pre_processor.preprocess(lidar_np, is_car=False)
            selected_cav_processed.update({'projected_lidar': projected_lidar,
                                            'projected_vsa_lidar': vsa_project_lidar,
                                            'no_projected_lidar': no_project_lidar,
                                            'processed_features': processed_lidar})
            
            # generate targets label single GT, note the reference pose is itself.
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center(
                [selected_cav_base], selected_cav_base['params']['lidar_pose']
            )
            label_dict = self.post_processor.generate_label(
                gt_box_center=object_bbx_center, anchors=self.anchor_box, mask=object_bbx_mask
            )
            selected_cav_processed.update({"object_bbx_center_no_coop": object_bbx_center[object_bbx_mask==1],
                                           "single_label_dict": label_dict})
            
            selected_cav_processed.update(
                {'object_bbx_center': object_bbx_center[object_bbx_mask == 1],
                'object_bbx_mask': object_bbx_mask,
                'object_ids': object_ids,
                'projected_lidar': lidar_np,
                'transformation_matrix': transformation_matrix,
                'transformation_matrix_clean': transformation_matrix_clean})

            return selected_cav_processed

        def collate_batch_test(self, batch):
            """
            Customized collate function for pytorch dataloader during testing
            for late fusion dataset.

            Parameters
            ----------
            batch : dict

            Returns
            -------
            batch : dict
                Reformatted batch.
            """
            # currently, we only support batch size of 1 during testing
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            batch = batch[0] # only ego

            output_dict = {}

            for cav_id, cav_content in batch.items():
                output_dict.update({cav_id: {}})
                # shape: (1, max_num, 7)
                object_bbx_center = \
                    torch.from_numpy(np.array([cav_content['object_bbx_center']]))
                object_bbx_mask = \
                    torch.from_numpy(np.array([cav_content['object_bbx_mask']]))
                object_ids = cav_content['object_ids']

                # the anchor box is the same for all bounding boxes usually, thus
                # we don't need the batch dimension.
                if cav_content['anchor_box'] is not None:
                    output_dict[cav_id].update({'anchor_box':
                        torch.from_numpy(np.array(
                            cav_content[
                                'anchor_box']))})
                if self.visualize:
                    origin_lidar = [cav_content['origin_lidar']]

                # processed lidar dictionary
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(
                        [cav_content['processed_lidar']])
                # label dictionary
                label_torch_dict = \
                    self.post_processor.collate_batch([cav_content['label_dict']])

                # save the transformation matrix (4, 4) to ego vehicle
                transformation_matrix_torch = \
                    torch.from_numpy(np.identity(4)).float()
                transformation_matrix_clean_torch = \
                    torch.from_numpy(np.identity(4)).float()

                output_dict[cav_id].update({'object_bbx_center': object_bbx_center,
                                            'object_bbx_mask': object_bbx_mask,
                                            'processed_lidar': processed_lidar_torch_dict,
                                            'label_dict': label_torch_dict,
                                            'object_ids': object_ids,
                                            'transformation_matrix': transformation_matrix_torch,
                                            'transformation_matrix_clean': transformation_matrix_clean_torch})

                if self.visualize:
                    origin_lidar = \
                        np.array(
                            downsample_lidar_minimum(pcd_np_list=origin_lidar))
                    origin_lidar = torch.from_numpy(origin_lidar)
                    output_dict[cav_id].update({'origin_lidar': origin_lidar})

            return output_dict
        
        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}
            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
            dec_processed_lidar_list = []
            image_inputs_list = []
            # used to record different scenario
            record_len = []
            label_dict_no_coop_batch_list = []
            label_dict_list = []
            lidar_pose_list = []
            origin_lidar = []
            vsa_lidar = []
            # 增加2个
            vsa_lidar_project = []
            vsa_lidar_noproject = []

            lidar_pose_clean_list = []
            # heterogeneous
            lidar_agent_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []
            
            ### 2022.10.10 single gt ####
            if self.supervise_single:
                pos_equal_one_single = []
                neg_equal_one_single = []
                targets_single = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                    dec_processed_lidar_list.append(ego_dict['dec_processed_lidar'])
                    vsa_lidar.append(ego_dict['vsa_lidar'])

                    vsa_lidar_project.append(ego_dict['vsa_lidar_project'])
                    vsa_lidar_noproject.append(ego_dict['vsa_lidar_noproject'])
                
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.
                
                record_len.append(ego_dict['cav_num'])
                label_dict_no_coop_batch_list.append(ego_dict['label_dict']['stage1'])
                label_dict_list.append(ego_dict['label_dict']['stage2'])
                
                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

                ### 2022.10.10 single gt ####
                if self.supervise_single:
                    pos_equal_one_single.append(ego_dict['single_label_dict_torch']['pos_equal_one'])
                    neg_equal_one_single.append(ego_dict['single_label_dict_torch']['neg_equal_one'])
                    targets_single.append(ego_dict['single_label_dict_torch']['targets'])

                # heterogeneous
                if self.heterogeneous:
                    lidar_agent_list.append(ego_dict['lidar_agent'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            if self.load_lidar_file:
                if processed_lidar_list[0] is None:
                    processed_lidar_torch_dict = {}
                    output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})
                else:
                    merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                    dec_merged_feature_dict = merge_features_to_dict(dec_processed_lidar_list)
                    if self.heterogeneous:
                        lidar_agent = np.concatenate(lidar_agent_list)
                        lidar_agent_idx = lidar_agent.nonzero()[0].tolist()
                        for k, v in merged_feature_dict.items(): # 'voxel_features' 'voxel_num_points' 'voxel_coords'
                            merged_feature_dict[k] = [v[index] for index in lidar_agent_idx]

                    if not self.heterogeneous or (self.heterogeneous and sum(lidar_agent) != 0):
                        processed_lidar_torch_dict = \
                            self.pre_processor.collate_batch(merged_feature_dict)
                        output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})
                        dec_processed_lidar_torch_dict = \
                            self.pre_processor.collate_batch(dec_merged_feature_dict)
                        output_dict['ego'].update({'dec_processed_lidar': dec_processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')

                if self.heterogeneous:
                    camera_agent = 1 - lidar_agent
                    camera_agent_idx = camera_agent.nonzero()[0].tolist()
                    if sum(camera_agent) != 0:
                        for k, v in merged_image_inputs_dict.items(): # 'imgs' 'rots' 'trans' ...
                            merged_image_inputs_dict[k] = torch.stack([v[index] for index in camera_agent_idx])
                            
                if not self.heterogeneous or (self.heterogeneous and sum(camera_agent) != 0):
                    output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})
            
            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            label_dict_no_coop_cavs_batch_list = [label_dict for label_dict_cavs_list in
                                    label_dict_no_coop_batch_list for label_dict in
                                    label_dict_cavs_list]
            label_no_coop_torch_dict = \
                                    self.post_processor.collate_batch(label_dict_no_coop_cavs_batch_list)
            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

            # for centerpoint
            label_torch_dict.update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask})

            # (B, max_cav)
            pairwise_t_matrix = torch.from_numpy(np.array(pairwise_t_matrix_list))

            # add pairwise_t_matrix to label dict
            label_torch_dict['pairwise_t_matrix'] = pairwise_t_matrix
            label_torch_dict['record_len'] = record_len

            # object id is only used during inference, where batch size is 1.
            # so here we only get the first element.
            output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                    'object_bbx_mask': object_bbx_mask,
                                    'record_len': record_len,
                                        'label_dict': {
                                            'stage1': label_no_coop_torch_dict,
                                            'stage2': label_torch_dict,
                                        },
                                    'object_ids': object_ids[0],
                                    'pairwise_t_matrix': pairwise_t_matrix,
                                    'lidar_pose_clean': lidar_pose_clean,
                                    'lidar_pose': lidar_pose,
                                    'proj_first': self.proj_first,
                                    'anchor_box': self.anchor_box_torch})

            if self.load_lidar_file:
                coords = []
                idx = 0
                for b in range(len(batch)):
                    for points in vsa_lidar[b]:
                        assert len(points) != 0
                        coor_pad = np.pad(points, ((0, 0), (1, 0)),
                                          mode="constant", constant_values=idx)
                        coords.append(coor_pad)
                        idx += 1
                origin_lidar_for_vsa = np.concatenate(coords, axis=0)
                origin_lidar_for_vsa = torch.from_numpy(origin_lidar_for_vsa)
                output_dict['ego'].update({'origin_lidar_for_vsa': origin_lidar_for_vsa})

                coords_project = []
                idx = 0
                for b in range(len(batch)):
                    for points in vsa_lidar_project[b]:
                        assert len(points) != 0
                        coor_pad = np.pad(points, ((0, 0), (1, 0)),
                                          mode="constant", constant_values=idx)
                        coords_project.append(coor_pad)
                        idx += 1

                # print("project",[t.shape for t in coords_project])
                origin_lidar_for_vsa_project = np.concatenate(coords_project, axis=0)
                origin_lidar_for_vsa_project = torch.from_numpy(origin_lidar_for_vsa_project)
                output_dict['ego'].update({'origin_lidar_for_vsa_project': origin_lidar_for_vsa_project})

                coords = []
                idx = 0
                for b in range(len(batch)):
                    for points in vsa_lidar_noproject[b]:
                        assert len(points) != 0
                        coor_pad = np.pad(points, ((0, 0), (1, 0)),
                                          mode="constant", constant_values=idx)
                        coords.append(coor_pad)
                        idx += 1
                # print("noproject",[t.shape for t in coords])

                origin_lidar_for_vsa_noproject = np.concatenate(coords, axis=0)
                origin_lidar_for_vsa_noproject = torch.from_numpy(origin_lidar_for_vsa_noproject)
                output_dict['ego'].update({'origin_lidar_for_vsa_noproject': origin_lidar_for_vsa_noproject})
            
            
            if self.visualize:
                origin_lidar = \
                    np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
                origin_lidar = torch.from_numpy(origin_lidar)
                output_dict['ego'].update({'origin_lidar': origin_lidar})

            if self.supervise_single:
                output_dict['ego'].update({
                    "label_dict_single" : 
                        {"pos_equal_one": torch.cat(pos_equal_one_single, dim=0),
                        "neg_equal_one": torch.cat(neg_equal_one_single, dim=0),
                        "targets": torch.cat(targets_single, dim=0)}
                })

            if self.heterogeneous:
                output_dict['ego'].update({
                    "lidar_agent_record": torch.from_numpy(np.concatenate(lidar_agent_list)) # [0,1,1,0,1...]
                })

            return output_dict

        def post_process(self, data_dict, output_dict):
            """
            Process the outputs of the model to 2D/3D bounding box.

            Parameters
            ----------
            data_dict : dict
                The dictionary containing the origin input data of model.

            output_dict :dict
                The dictionary containing the output of the model.

            Returns
            -------
            pred_box_tensor : torch.Tensor
                The tensor of prediction bounding box after NMS.
            gt_box_tensor : torch.Tensor
                The tensor of gt bounding box.
            """
            pred_box_tensor, pred_score = \
                self.post_processor.post_process(data_dict, output_dict)
            gt_box_tensor = self.post_processor.generate_gt_bbx(data_dict)

            return pred_box_tensor, pred_score, gt_box_tensor
    
    return DecdiffusionFusionDataset

def visualize_gt_boxes(gt_boxes, pcd_array, pc_range, save_path, scale_3d=40, scale_bev=10):
    pcd_tensor = torch.from_numpy(pcd_array)
    if gt_boxes is not None:
        ex_gt_box_corners = boxes_to_corners_3d(gt_boxes,'hwl')
        ex_gt_box_tensor = torch.from_numpy(ex_gt_box_corners).float() if isinstance(ex_gt_box_corners, np.ndarray) else ex_gt_box_corners
        infer_result = {
            "gt_box_tensor": ex_gt_box_tensor,
            # "pred_box_tensor": None  # 如果你有预测框，可以添加这个
        }
    else:
        infer_result = None
    visualize(
        infer_result=infer_result, 
        pcd=pcd_tensor, 
        pc_range=pc_range, 
        save_path=save_path, 
        scale_3d=scale_3d,
        scale_bev=scale_bev,
        method='bev',  # 3d或者使用'bev'来获取鸟瞰图
        left_hand=False  # 如果你的坐标系是左手系，设置为True
    )    
            

