# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, OpenPCDet
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Transform points to voxels using sparse conv library
"""
import sys

import numpy as np
import torch
from icecream import ic

from opencood.data_utils.pre_processor.base_preprocessor import \
    BasePreprocessor


class SpVoxelPreprocessor(BasePreprocessor):
    def __init__(self, preprocess_params, train):
        super(SpVoxelPreprocessor, self).__init__(preprocess_params,
                                                  train)
        self.spconv = 1
        try:
            # spconv v1.x
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
        except:
            # spconv v2.x
            from cumm import tensorview as tv
            from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
            self.tv = tv
            self.spconv = 2
        self.lidar_range = self.params['cav_lidar_range']
        self.voxel_size = self.params['args']['voxel_size']
        self.max_points_per_voxel = self.params['args']['max_points_per_voxel']
        self.max_hwl = self.params['args']['max_hwl']
        self.use_hwl = self.params['args']['use_hwl']
        max_h, max_w, max_l = self.max_hwl
        self.gt_range = [-max_l/2, -max_w/2, -1, max_l/2, max_w/2, 1] #z轴不能为0
        if train:
            self.max_voxels = self.params['args']['max_voxel_train']
        else:
            self.max_voxels = self.params['args']['max_voxel_test']
        if self.use_hwl:
            grid_size = (np.array(self.gt_range[3:6]) -
                     np.array(self.gt_range[0:3])) / np.array(self.voxel_size)
        else:
            grid_size = (np.array(self.lidar_range[3:6]) -
                     np.array(self.lidar_range[0:3])) / np.array(self.voxel_size)
        self.grid_size = np.maximum(np.round(grid_size).astype(np.int64), 1)

        # use sparse conv library to generate voxel
        if self.spconv == 1:
            if self.use_hwl:
                self.voxel_generator_car = VoxelGenerator(
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.gt_range,
                    max_num_points=self.max_points_per_voxel,
                    max_voxels=self.max_voxels
                )
            
            self.voxel_generator_lidar = VoxelGenerator(
                voxel_size=self.voxel_size,
                point_cloud_range=self.lidar_range,
                max_num_points=self.max_points_per_voxel,
                max_voxels=self.max_voxels)
        else:
            if self.use_hwl:
                self.voxel_generator_car = VoxelGenerator(
                    vsize_xyz=self.voxel_size,
                    coors_range_xyz=self.gt_range,
                    max_num_points_per_voxel=self.max_points_per_voxel,
                    num_point_features=4,
                    max_num_voxels=self.max_voxels
                    )
            
            self.voxel_generator_lidar = VoxelGenerator(
                vsize_xyz=self.voxel_size,
                coors_range_xyz=self.lidar_range,
                max_num_points_per_voxel=self.max_points_per_voxel,
                num_point_features=4,
                max_num_voxels=self.max_voxels)
    def preprocess(self, pcd_np, is_car):
        data_dict = {}
        if is_car:
            self.voxel_generator = self.voxel_generator_car
        else:
            self.voxel_generator = self.voxel_generator_lidar
            
        if self.spconv == 1:
            voxel_output = self.voxel_generator.generate(pcd_np)
        else:
            # 注意voxel_generator中的雷达范围
            pcd_tv = self.tv.from_numpy(pcd_np)
            voxel_output = self.voxel_generator.point_to_voxel(pcd_tv)
        if isinstance(voxel_output, dict):
            voxels, coordinates, num_points = \
                voxel_output['voxels'], voxel_output['coordinates'], \
                voxel_output['num_points_per_voxel']
        else:
            voxels, coordinates, num_points = voxel_output

        if self.spconv == 2:
            voxels = voxels.numpy()
            coordinates = coordinates.numpy()
            num_points = num_points.numpy()
        
        data_dict['voxel_features'] = voxels #[非空体素的数量(num_voxels),每个体素最多保存的点数(max_points_per_voxel),每个点的特征维度(x,y,z,强度)]
        data_dict['voxel_coords'] = coordinates  # [num_voxels, 3(z, y, x: 体素在3D空间中的坐标索引)]
        data_dict['voxel_num_points'] = num_points # [num_voxels]记录每个体素内实际包含多少个点

        return data_dict

    def collate_batch(self, batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list or dict
            List or dictionary.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """

        if isinstance(batch, list):
            return self.collate_batch_list(batch)
        elif isinstance(batch, dict):
            return self.collate_batch_dict(batch)
        else:
            sys.exit('Batch has too be a list or a dictionarn')

    @staticmethod
    def collate_batch_list(batch):
        """
        Customized pytorch data loader collate function.

        Parameters
        ----------
        batch : list
            List of dictionary. Each dictionary represent a single frame.

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = []
        voxel_num_points = []
        voxel_coords = []

        for i in range(len(batch)):
            voxel_features.append(batch[i]['voxel_features'])
            voxel_num_points.append(batch[i]['voxel_num_points'])
            coords = batch[i]['voxel_coords']
            voxel_coords.append(
                np.pad(coords, ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))

        voxel_num_points = torch.from_numpy(np.concatenate(voxel_num_points))
        voxel_features = torch.from_numpy(np.concatenate(voxel_features))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        return {'voxel_features': voxel_features,
                'voxel_coords': voxel_coords,
                'voxel_num_points': voxel_num_points}

    @staticmethod
    def collate_batch_dict(batch: dict):
        """
        Collate batch if the batch is a dictionary,
        eg: {'voxel_features': [feature1, feature2...., feature n]}

        Parameters
        ----------
        batch : dict

        Returns
        -------
        processed_batch : dict
            Updated lidar batch.
        """
        voxel_features = \
            torch.from_numpy(np.concatenate(batch['voxel_features']))
        voxel_num_points = \
            torch.from_numpy(np.concatenate(batch['voxel_num_points']))
        coords = batch['voxel_coords']
        voxel_coords = []
        
        # 添加batch_idx
        for i in range(len(coords)):
            voxel_coords.append(
                np.pad(coords[i], ((0, 0), (1, 0)),
                       mode='constant', constant_values=i))
        voxel_coords = torch.from_numpy(np.concatenate(voxel_coords))

        output_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        if 'gt_masks' in batch:
            output_dict['gt_masks'] = \
                torch.from_numpy(np.concatenate(batch['gt_masks']))
        if 'gt_boxes' in batch:
            output_dict['gt_boxes'] = \
                torch.from_numpy(np.concatenate(batch['gt_boxes']))

        return output_dict