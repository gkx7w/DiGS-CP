# intermediate fusion dataset
import random
import math
from collections import OrderedDict
import numpy as np
import torch
import copy
from icecream import ic
from PIL import Image
import pickle as pkl
from opencood.utils import box_utils as box_utils
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.data_utils.post_processor import build_postprocessor
from opencood.utils.camera_utils import (
    sample_augmentation,
    img_transform,
    normalize_img,
    img_to_tensor,
)
from opencood.utils.common_utils import merge_features_to_dict
from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation
from opencood.utils.pose_utils import add_noise_data_dict
from opencood.utils.pcd_utils import (
    mask_points_by_range,
    mask_ego_points,
    shuffle_points,
    downsample_lidar_minimum,
)


def getIntermediate2stageFusionDataset(cls):
    """
    cls: the Basedataset.
    """

    class Intermediate2stageFusionDataset(cls):
        def __init__(self, params, visualize, train=True):
            super().__init__(params, visualize, train)
            # intermediate and supervise single
            self.supervise_single = True if (
                        'supervise_single' in params['model']['args'] and params['model']['args']['supervise_single']) \
                else False
            # it is assert to be False but by default it will load single label for 1-stage training.
            assert self.supervise_single is False

            self.proj_first = False if 'proj_first' not in params['fusion']['args'] \
                else params['fusion']['args']['proj_first']

            self.anchor_box = self.post_processor.generate_anchor_box()
            self.anchor_box_torch = torch.from_numpy(self.anchor_box)

        def get_item_single_car(self, selected_cav_base, ego_cav_base):
            """
            Process a single CAV's information for the train/test pipeline.


            Parameters
            ----------
            selected_cav_base : dict
                The dictionary contains a single CAV's raw information.
                including 'params', 'camera_data'
            ego_pose : list, length 6
                The ego vehicle lidar pose under world coordinate.
            ego_pose_clean : list, length 6
                only used for gt box generation

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
                         ego_pose)  # T_ego_cav
            transformation_matrix_clean = \
                x1_to_x2(selected_cav_base['params']['lidar_pose_clean'],
                         ego_pose_clean)

            # lidar
            if self.load_lidar_file or self.visualize:
                # process lidar
                lidar_np = selected_cav_base['lidar_np']
                lidar_np = shuffle_points(lidar_np)
                # remove points that hit itself
                lidar_np = mask_ego_points(lidar_np)

                # no projected lidar
                no_project_lidar = copy.deepcopy(lidar_np)

                # project the lidar to ego space
                # x,y,z in ego space
                projected_lidar = \
                    box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                             transformation_matrix)

                vsa_project_lidar = copy.deepcopy(lidar_np)
                vsa_project_lidar[:, :3] = projected_lidar

                if self.proj_first:  #
                    lidar_np[:, :3] = projected_lidar

                if self.visualize:
                    # filter lidar
                    selected_cav_processed.update({'projected_lidar': projected_lidar})

                processed_lidar = self.pre_processor.preprocess(lidar_np,is_car=False)
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
            selected_cav_processed.update({"object_bbx_center_no_coop": object_bbx_center[object_bbx_mask == 1],
                                           "single_label_dict": label_dict})

            # camera
            if self.load_camera_file:
                camera_data_list = selected_cav_base["camera_data"]

                params = selected_cav_base["params"]
                imgs = []
                rots = []
                trans = []
                intrins = []
                post_rots = []
                post_trans = []

                for idx, img in enumerate(camera_data_list):
                    camera_to_lidar, camera_intrinsic = self.get_ext_int(params, idx)

                    intrin = torch.from_numpy(camera_intrinsic)
                    rot = torch.from_numpy(
                        camera_to_lidar[:3, :3]
                    )  # R_wc, we consider world-coord is the lidar-coord
                    tran = torch.from_numpy(camera_to_lidar[:3, 3])  # T_wc

                    post_rot = torch.eye(2)
                    post_tran = torch.zeros(2)

                    img_src = [img]

                    # depth
                    if self.load_depth_file:
                        depth_img = selected_cav_base["depth_data"][idx]
                        img_src.append(depth_img)
                    else:
                        depth_img = None

                    # data augmentation
                    resize, resize_dims, crop, flip, rotate = sample_augmentation(
                        self.data_aug_conf, self.train
                    )
                    img_src, post_rot2, post_tran2 = img_transform(
                        img_src,
                        post_rot,
                        post_tran,
                        resize=resize,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate,
                    )
                    # for convenience, make augmentation matrices 3x3
                    post_tran = torch.zeros(3)
                    post_rot = torch.eye(3)
                    post_tran[:2] = post_tran2
                    post_rot[:2, :2] = post_rot2

                    # decouple RGB and Depth

                    img_src[0] = normalize_img(img_src[0])
                    if self.load_depth_file:
                        img_src[1] = img_to_tensor(img_src[1]) * 255

                    imgs.append(torch.cat(img_src, dim=0))
                    intrins.append(intrin)
                    rots.append(rot)
                    trans.append(tran)
                    post_rots.append(post_rot)
                    post_trans.append(post_tran)

                selected_cav_processed.update(
                    {
                        "image_inputs":
                            {
                                "imgs": torch.stack(imgs),  # [Ncam, 3or4, H, W]
                                "intrins": torch.stack(intrins),
                                "rots": torch.stack(rots),
                                "trans": torch.stack(trans),
                                "post_rots": torch.stack(post_rots),
                                "post_trans": torch.stack(post_trans),
                            }
                    }
                )

            # anchor box
            selected_cav_processed.update({"anchor_box": self.anchor_box})

            # note the reference pose ego
            object_bbx_center, object_bbx_mask, object_ids = self.generate_object_center([selected_cav_base],
                                                                                         ego_pose_clean)

            selected_cav_processed.update(
                {
                    "object_bbx_center": object_bbx_center[object_bbx_mask == 1],
                    "object_bbx_mask": object_bbx_mask,
                    "object_ids": object_ids,
                    'transformation_matrix': transformation_matrix,
                    'transformation_matrix_clean': transformation_matrix_clean
                }
            )

            return selected_cav_processed

        def __getitem__(self, idx):

            # put here to avoid initialization error
            from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils \
                import points_in_boxes_cpu


            base_data_dict = self.retrieve_base_data(idx)
            base_data_dict = add_noise_data_dict(base_data_dict, self.params['noise_setting'])

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

            agents_image_inputs = []
            processed_features = []
            object_stack = []
            object_id_stack = []
            single_label_list = []
            too_far = []
            lidar_pose_list = []
            lidar_pose_clean_list = []
            cav_id_list = []

            projected_lidar_stack = []
            no_projected_lidar_stack = []

            vsa_lidar_stack = []

            vsa_lidar_stack_project = []
            vsa_lidar_stack_noproject = []

            Multi_view_object_id_stack = []

            if self.visualize:
                projected_lidar_stack = []

            # loop over all CAVs to process information
            for cav_id, selected_cav_base in base_data_dict.items():
                # check if the cav is within the communication range with ego
                distance = \
                    math.sqrt((selected_cav_base['params']['lidar_pose'][0] -
                            ego_lidar_pose[0]) ** 2 + (
                                    selected_cav_base['params'][
                                        'lidar_pose'][1] - ego_lidar_pose[
                                        1]) ** 2)

                # if distance is too far, we will just skip this agent
                if distance > self.params['comm_range']:
                    too_far.append(cav_id)
                    continue

                lidar_pose_clean_list.append(selected_cav_base['params']['lidar_pose_clean'])
                lidar_pose_list.append(selected_cav_base['params']['lidar_pose']) # 6dof pose
                cav_id_list.append(cav_id)

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

            for _i, cav_id in enumerate(cav_id_list):
                selected_cav_base = base_data_dict[cav_id]

                selected_cav_processed = self.get_item_single_car(
                    selected_cav_base,
                    ego_cav_base)

                object_stack.append(selected_cav_processed['object_bbx_center'])
                object_id_stack += selected_cav_processed['object_ids']

                Multi_view_object_id_stack.append(selected_cav_processed['object_ids'])

                if self.load_lidar_file:
                    processed_features.append(
                        selected_cav_processed['processed_features'])
                    if self.proj_first:
                        vsa_lidar_stack.append(selected_cav_processed['projected_lidar'])
                    else:
                        vsa_lidar_stack.append(selected_cav_processed['no_projected_lidar'])

                    vsa_lidar_stack_project.append(selected_cav_processed['projected_vsa_lidar'])
                    vsa_lidar_stack_noproject.append(selected_cav_processed['no_projected_lidar'])

                if self.load_camera_file:
                    agents_image_inputs.append(
                        selected_cav_processed['image_inputs'])

                # if self.visualize:
                #     projected_lidar_stack.append(
                #         selected_cav_processed['projected_lidar'])
                projected_lidar_stack.append(
                    selected_cav_processed['projected_lidar'])
       
                single_label_list.append(selected_cav_processed['single_label_dict'])

            # generate single view label (no coop) label
            label_dict_no_coop = single_label_list # [{cav1_label}, {cav2_label}...]
            
            
            # exclude all repetitive objects
            unique_indices = \
                [object_id_stack.index(x) for x in set(object_id_stack)]

            unique_indices_order = []
            unique_data_order = []
            tmp_set = set()
            for index, id in enumerate(object_id_stack):
                if id not in tmp_set:
                    unique_indices_order.append(index)
                    unique_data_order.append(id)
                    tmp_set.add(id)

            Multi_view_object_id_index = [[unique_data_order.index(b) for b in a] for a in
                                                  Multi_view_object_id_stack]
            tmp_object_stack = object_stack


            object_stack = np.vstack(object_stack)
            # object_stack = object_stack[unique_indices]

            object_stack = object_stack[unique_indices_order]

            # make sure bounding boxes across all frames have the same number
            object_bbx_center = \
                np.zeros((self.params['postprocess']['max_num'], 7))
            mask = np.zeros(self.params['postprocess']['max_num'])
            object_bbx_center[:object_stack.shape[0], :] = object_stack
            mask[:object_stack.shape[0]] = 1

            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_features)
                processed_data_dict['ego'].update({'processed_lidar': merged_feature_dict,
                                                   'vsa_lidar': vsa_lidar_stack,
                                                   'vsa_lidar_project': vsa_lidar_stack_project,
                                                   'vsa_lidar_noproject': vsa_lidar_stack_noproject})
            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(agents_image_inputs, merge='stack')
                processed_data_dict['ego'].update({'image_inputs': merged_image_inputs_dict})

            # generate targets label
            label_dict_coop = \
                self.post_processor.generate_label(
                    gt_box_center=object_bbx_center,
                    anchors=self.anchor_box,
                    mask=mask)



            object_stack_filtered = []
            label_dict_no_coop = []
            Multi_view_object_id_index_filtered = []
            Multi_view_point_in_box_filtered = []
            for boxes, points, id_index in zip(tmp_object_stack, projected_lidar_stack, Multi_view_object_id_index):

                point_indices = points_in_boxes_cpu(points[:, :3], boxes[:,
                                                                   [0, 1, 2, 5, 4,
                                                                    3, 6]])
                cur_num = point_indices.sum(axis=1)
                cur_mask = cur_num > 0


                tmp_list = np.zeros((self.params['postprocess']['max_num']))
                tmp_num = np.zeros((self.params['postprocess']['max_num']))
                for i, v in enumerate(cur_mask):
                    if v:
                        tmp_list[id_index[i]] = 1
                        tmp_num[id_index[i]] = cur_num[i]
                Multi_view_object_id_index_filtered.append(tmp_list)
                Multi_view_point_in_box_filtered.append(tmp_num)

                # if cur_mask.sum() == 0:
                #     label_dict_no_coop.append({
                #         'pos_equal_one': np.zeros((*anchor_box.shape[:2],
                #                                    self.post_processor.anchor_num)),
                #         'neg_equal_one': np.ones((*anchor_box.shape[:2],
                #                                   self.post_processor.anchor_num)),
                #         'targets': np.zeros((*anchor_box.shape[:2],
                #                              self.post_processor.anchor_num * 7))
                #     })
                #     continue
                #
                # object_stack_filtered.append(boxes[cur_mask])
                # bbx_center = \
                #     np.zeros((self.params['postprocess']['max_num'], 7))
                # bbx_mask = np.zeros(self.params['postprocess']['max_num'])
                # bbx_center[:boxes[cur_mask].shape[0], :] = boxes[cur_mask]
                # bbx_mask[:boxes[cur_mask].shape[0]] = 1
                # label_dict_no_coop.append(
                #     self.post_processor.generate_label_naive(
                #         gt_box_center=bbx_center,  # hwl
                #         anchors=anchor_box,
                #         mask=bbx_mask)
                # )
            Multi_view_object_id_index_filtered = np.array(Multi_view_object_id_index_filtered)
            Multi_view_point_in_box_filtered = np.array(Multi_view_point_in_box_filtered)

            # print()
            # print()
            # print("object_id_stack", object_id_stack)
            # print("Multi_view_object_id_stack", Multi_view_object_id_stack)
            # print("unique_indices", unique_indices)
            # print("unique_indices_order", unique_indices_order)
            # print("unique_data_order", unique_data_order)
            # print("Multi_view_object_id_index", Multi_view_object_id_index)
            # print("转化格式后",Multi_view_object_id_index_filtered[:,:10])
            # # Multi_view_point_in_box_filtered
            # print("num:",Multi_view_point_in_box_filtered[:,:10])
            # print()




            # label_dict = {
            #     'stage1': label_dict_no_coop, # list
            #     'stage2': label_dict_coop # dict
            # }

            label_dict = {
                'stage1': label_dict_coop,  # list
                'stage2': label_dict_coop  # dict
            }

            processed_data_dict['ego'].update(
                {'object_bbx_center': object_bbx_center,
                 'object_bbx_mask': mask,
                 'object_ids': [object_id_stack[i] for i in unique_indices],
                 'anchor_box': self.anchor_box,
                 'label_dict': label_dict,
                 'cav_num': cav_num,
                 'pairwise_t_matrix': pairwise_t_matrix,
                 'lidar_poses_clean': lidar_poses_clean,
                 'lidar_poses': lidar_poses,

                 'Multi_view_box_index': Multi_view_object_id_index_filtered,
                 'Multi_view_box_points': Multi_view_point_in_box_filtered

                 })

            if self.visualize:
                processed_data_dict['ego'].update({'origin_lidar':
                    np.vstack(
                        projected_lidar_stack)})

            processed_data_dict['ego'].update({'sample_idx': idx,
                                               'cav_id_list': cav_id_list})

            return processed_data_dict

        def collate_batch_train(self, batch):
            # Intermediate fusion is different the other two
            output_dict = {'ego': {}}

            object_bbx_center = []
            object_bbx_mask = []
            object_ids = []
            processed_lidar_list = []
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

            # 新增内容
            Multi_view_box_index = []
            Multi_view_box_points = []

            lidar_pose_clean_list = []

            # pairwise transformation matrix
            pairwise_t_matrix_list = []

            for i in range(len(batch)):
                ego_dict = batch[i]['ego']
                object_bbx_center.append(ego_dict['object_bbx_center'])
                object_bbx_mask.append(ego_dict['object_bbx_mask'])
                object_ids.append(ego_dict['object_ids'])
                lidar_pose_list.append(ego_dict['lidar_poses']) # ego_dict['lidar_pose'] is np.ndarray [N,6]
                lidar_pose_clean_list.append(ego_dict['lidar_poses_clean'])
                if self.load_lidar_file:
                    processed_lidar_list.append(ego_dict['processed_lidar'])
                    vsa_lidar.append(ego_dict['vsa_lidar'])

                    vsa_lidar_project.append(ego_dict['vsa_lidar_project'])
                    vsa_lidar_noproject.append(ego_dict['vsa_lidar_noproject'])
                if self.load_camera_file:
                    image_inputs_list.append(ego_dict['image_inputs']) # different cav_num, ego_dict['image_inputs'] is dict.

                record_len.append(ego_dict['cav_num'])
                label_dict_no_coop_batch_list.append(ego_dict['label_dict']['stage1'])
                label_dict_list.append(ego_dict['label_dict']['stage2'])

                pairwise_t_matrix_list.append(ego_dict['pairwise_t_matrix'])

                Multi_view_box_index.append(ego_dict['Multi_view_box_index'])
                Multi_view_box_points.append(ego_dict['Multi_view_box_points'])

                if self.visualize:
                    origin_lidar.append(ego_dict['origin_lidar'])

            # convert to numpy, (B, max_num, 7)
            object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
            object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

            Multi_view_box_index = torch.from_numpy(np.vstack(Multi_view_box_index))
            Multi_view_box_points = torch.from_numpy(np.vstack(Multi_view_box_points))

            # example: {'voxel_features':[np.array([1,2,3]]),
            # np.array([3,5,6]), ...]}
            if self.load_lidar_file:
                merged_feature_dict = merge_features_to_dict(processed_lidar_list)
                # [sum(record_len), C, H, W]
                processed_lidar_torch_dict = \
                    self.pre_processor.collate_batch(merged_feature_dict)
                output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict})

            if self.load_camera_file:
                merged_image_inputs_dict = merge_features_to_dict(image_inputs_list, merge='cat')
                output_dict['ego'].update({'image_inputs': merged_image_inputs_dict})

            record_len = torch.from_numpy(np.array(record_len, dtype=int))
            lidar_pose = torch.from_numpy(np.concatenate(lidar_pose_list, axis=0))
            lidar_pose_clean = torch.from_numpy(np.concatenate(lidar_pose_clean_list, axis=0))
            # label_dict_no_coop_cavs_batch_list = [label_dict for label_dict_cavs_list in
            #                         label_dict_no_coop_batch_list for label_dict in
            #                         label_dict_cavs_list]
            # label_no_coop_torch_dict = \
            #                         self.post_processor.collate_batch(label_dict_no_coop_cavs_batch_list)

            label_torch_dict = \
                self.post_processor.collate_batch(label_dict_list)

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
                                           'stage1': label_torch_dict,
                                           'stage2': label_torch_dict,
                                       },
                                       'object_ids': object_ids[0],
                                       'pairwise_t_matrix': pairwise_t_matrix,
                                       'lidar_pose_clean': lidar_pose_clean,
                                       'lidar_pose': lidar_pose,
                                       'proj_first': self.proj_first,
                                       'anchor_box': self.anchor_box_torch,

                                       'Multi_view_box_index': Multi_view_box_index,
                                       'Multi_view_box_points': Multi_view_box_points
                                       })

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

            return output_dict

        def collate_batch_test(self, batch):
            assert len(batch) <= 1, "Batch size 1 is required during testing!"
            output_dict = self.collate_batch_train(batch)
            if output_dict is None:
                return None

            # check if anchor box in the batch
            output_dict['ego'].update({'anchor_box': self.anchor_box_torch})

            # save the transformation matrix (4, 4) to ego vehicle
            # transformation is only used in post process (no use.)
            # we all predict boxes in ego coord.
            transformation_matrix_torch = \
                torch.from_numpy(np.identity(4)).float()
            transformation_matrix_clean_torch = \
                torch.from_numpy(np.identity(4)).float()

            output_dict['ego'].update({'transformation_matrix':
                                        transformation_matrix_torch,
                                       'transformation_matrix_clean':
                                        transformation_matrix_clean_torch,})

            output_dict['ego'].update({
                "sample_idx": batch[0]['ego']['sample_idx'],
                "cav_id_list": batch[0]['ego']['cav_id_list']
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

    return Intermediate2stageFusionDataset