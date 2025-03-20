# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os.path
from typing import Union

import numpy as np
import torch
import einops
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.data_utils.datasets import build_dataset
from opencood.pcdet_utils.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu, points_in_boxes_cpu
from opencood.visualization import simple_vis


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument(
        "--hypes_yaml",
        "-y",
        type=str,
        required=True,
        help="data generation yaml file needed ",
    )
    parser.add_argument("--model_dir", default="", help="Continued training path")
    parser.add_argument("--fusion_method", "-f", default="intermediate", help="passed to inference.")
    opt = parser.parse_args()
    return opt


def pad_point_cloud(original_points: torch.Tensor, target_num: int, device: Union[str, torch.device] = "cpu") -> tuple[
    Tensor, Tensor]:
    """
    :param original_points: (N, 4)
    """
    # 获取原始点云的数量
    original_num = original_points.shape[0]

    # 检查输入是否小于目标数量
    # TODO: 这里直接返回是否可以呢? 还说是应该对多余的点云进行裁剪
    if original_num >= target_num:
        mask = torch.ones(original_num, dtype=torch.bool, device=device)
        return original_points.to(device=device), mask

    # 计算需要填充的点数
    pad_num = target_num - original_num

    # 随机选择需要重复的点（这里使用简单重复填充，你可以根据需要修改填充策略）
    original_points = original_points.to(device=device)
    pad_points = torch.zeros((pad_num, original_points.shape[1]), dtype=original_points.dtype, device=device)
    # pad_indices = torch.randint(0, original_num, size=(pad_num,), device=device)
    # pad_points = original_points[pad_indices]

    # 使用torch.cat合并原始点云和填充点
    padded_points = torch.cat([original_points, pad_points], dim=0)

    # 创建掩码：原始点为 True，填充点为 False
    mask = torch.zeros(target_num, dtype=torch.bool, device=device)
    mask[:original_num] = True

    return padded_points, mask


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)

    print("Dataset Building")
    opencood_train_dataset = build_dataset(hypes, visualize=True, train=True)
    opencood_validate_dataset = build_dataset(hypes, visualize=True, train=False)

    train_loader = DataLoader(
        opencood_train_dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=4,
        collate_fn=opencood_train_dataset.collate_batch_train,
        shuffle=False,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        opencood_validate_dataset,
        batch_size=hypes["train_params"]["batch_size"],
        num_workers=4,
        collate_fn=opencood_train_dataset.collate_batch_train,
        shuffle=True,
        pin_memory=False,
        drop_last=True,
        prefetch_factor=2,
    )
    device = torch.device("cuda:1")
    dtype = torch.float32

    for step, batch_data in enumerate(train_loader):
        if batch_data is None or batch_data["ego"]["object_bbx_mask"].sum() == 0:
            continue
        ego_data = batch_data["ego"]
        lidar_np_list_list = ego_data["lidar_np_list_list"]  # list shape 后两个维度是 numpy 的: (B, CAV_NUM, N, 4)
        object_bbx_center = ego_data["object_bbx_center"]
        object_bbx_mask = ego_data["object_bbx_mask"]
        # object_bbx_center_valid = object_bbx_center[object_bbx_mask == 1]
        ego_data["transformation_matrix_clean"] = torch.from_numpy(np.identity(4)).to(dtype=dtype)
        ego_data["transformation_matrix"] = torch.from_numpy(np.identity(4)).to(dtype=dtype)
        infer_result = {
            "gt_box_tensor": opencood_train_dataset.post_processor.generate_gt_bbx(batch_data)
        }

        """用 cpu 试试看呢?"""
        lidar_np = ego_data["origin_lidar"][0]
        point_indices = points_in_boxes_cpu(lidar_np, object_bbx_center[object_bbx_mask == 1])
        point_in_boxes = np.empty((1, 3))
        for i in range(point_indices.shape[0]):
            point_in_boxes = np.concatenate((point_in_boxes, lidar_np[point_indices[i] == 1]), axis=0)
        # point_in_boxes = np.stack(point_in_boxes)
        simple_vis.visualize(
            infer_result,
            torch.tensor(point_in_boxes),
            hypes["postprocess"]["gt_range"],
            os.path.join("/home/ubuntu/Code2/opencood/logs/semantic_segmentation/vis", f"bev_{step:05d}_cpu.png"),
            # method="bev"
        )

        N = 58000  # 为了并行化处理, 把每个 batch 里面的 lidar 数量增加到 58000
        batched_lidar_np = []
        batched_original_point_mask = []
        for lidar_np_list in lidar_np_list_list:
            for lidar_np in lidar_np_list:
                padded_lidar_np, original_point_mask = pad_point_cloud(torch.from_numpy(lidar_np), N, device)
                batched_lidar_np.append(padded_lidar_np)
                batched_original_point_mask.append(original_point_mask)
        # x y z h w l yaw -> x y z l w h yaw
        object_bbx_center_with_order_lwh = object_bbx_center[..., [0, 1, 2, 5, 4, 3, 6]]
        batched_lidar_np = torch.stack(batched_lidar_np, dim=0)
        # batched_lidar_np = batched_lidar_np[:, :, :3]
        batched_lidar_np_without_reflect = batched_lidar_np[:, :, :3]
        batched_original_point_mask = torch.stack(batched_original_point_mask, dim=0)
        batched_original_point_mask = batched_original_point_mask.to(device=device)  # 这里是 bool 类型, 可千万别 to 的时候转成其他类型

        # TODO: 注意设备的选择
        points_in_boxes_mask = points_in_boxes_gpu(batched_lidar_np_without_reflect.to(device=device, dtype=dtype),
                                                   object_bbx_center_with_order_lwh.to(device=device, dtype=dtype))
        # print(points_in_boxes_mask)
        points_in_boxes_list = []
        batch_size = batched_lidar_np.shape[0]

        for batch_num in range(batch_size):
            lidar_np = batched_lidar_np[batch_num]
            # 记得加括号, 不然会有运算顺序的问题
            final_mask = batched_original_point_mask[batch_num] & (points_in_boxes_mask[batch_num] != -1)
            lidar_np = lidar_np[final_mask]
            logger.info(f"{step=} {batch_num=} {lidar_np.shape=}")
            points_in_boxes_list.append(lidar_np)

        simple_vis.visualize(
            infer_result,
            points_in_boxes_list[0],
            hypes["postprocess"]["gt_range"],
            os.path.join("/home/ubuntu/Code2/opencood/logs/semantic_segmentation/vis", f"bev_{step:05d}_gpu.png"),
            # method="bev"
        )
        simple_vis.visualize(
            infer_result,
            torch.tensor(ego_data["lidar_np_list_list"][0][0]),
            hypes["postprocess"]["gt_range"],
            os.path.join("/home/ubuntu/Code2/opencood/logs/semantic_segmentation/vis", f"bev_{step:05d}.png"),
            # method="bev"
        )

        break


if __name__ == "__main__":
    main()
