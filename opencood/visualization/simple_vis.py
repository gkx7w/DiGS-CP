# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


from matplotlib import pyplot as plt
import numpy as np
import copy
import torch
from matplotlib.colors import Normalize
from opencood.tools.inference_utils import get_cav_box
import opencood.visualization.simple_plot3d.canvas_3d as canvas_3d
import opencood.visualization.simple_plot3d.canvas_bev as canvas_bev
import os
import math
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import torch
import numpy as np
from matplotlib.gridspec import GridSpec


def visualize(infer_result, pcd, pc_range, save_path, scale_3d=40, scale_bev=10, method='3d', left_hand=False):
        """
        Visualize the prediction, ground truth with point cloud together.
        They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

        Parameters
        ----------
        infer_result:
            pred_box_tensor : torch.Tensor
                (N, 8, 3) prediction.

            gt_tensor : torch.Tensor
                (N, 8, 3) groundtruth bbx
            
            uncertainty_tensor : optional, torch.Tensor
                (N, ?)

            lidar_agent_record: optional, torch.Tensor
                (N_agnet, )


        pcd : torch.Tensor
            PointCloud, (N, 4).

        pc_range : list
            [xmin, ymin, zmin, xmax, ymax, zmax]

        save_path : str
            Save the visualization results to given path.

        dataset : BaseDataset
            opencood dataset object.

        method: str, 'bev' or '3d'

        """
        plt.figure(figsize=[(pc_range[3]-pc_range[0])/scale_3d, (pc_range[4]-pc_range[1])/scale_3d]) #40
        pc_range = [int(i) for i in pc_range]
        pcd_np = pcd.cpu().numpy()

        # 为了展示不同视角差异，只保留绿色框
        # pred_box_tensor = infer_result.get("pred_box_tensor", None)
        pred_box_tensor = None
        gt_box_tensor = infer_result.get("gt_box_tensor", None)

        if pred_box_tensor is not None:
            pred_box_np = pred_box_tensor.cpu().numpy()
            pred_name = ['pred'] * pred_box_np.shape[0]

            score = infer_result.get("score_tensor", None)
            if score is not None:
                score_np = score.cpu().numpy()
                pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

            uncertainty = infer_result.get("uncertainty_tensor", None)
            if uncertainty is not None:
                uncertainty_np = uncertainty.cpu().numpy()
                uncertainty_np = np.exp(uncertainty_np)
                d_a_square = 1.6**2 + 3.9**2
                
                if uncertainty_np.shape[1] == 3:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) 
                    # yaw angle is in radian, it's the same in g2o SE2's setting.

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:.3f} a_u:{uncertainty_np[i,2]:.3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 2:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]

                elif uncertainty_np.shape[1] == 7:
                    uncertainty_np[:,:2] *= d_a_square
                    uncertainty_np = np.sqrt(uncertainty_np) # yaw angle is in radian

                    pred_name = [f'x_u:{uncertainty_np[i,0]:.3f} y_u:{uncertainty_np[i,1]:3f} a_u:{uncertainty_np[i,6]:3f}' \
                                    for i in range(uncertainty_np.shape[0])]                    

        if gt_box_tensor is not None:
            gt_box_np = gt_box_tensor.cpu().numpy()
            gt_name = ['gt'] * gt_box_np.shape[0]

        if method == 'bev':
            canvas = canvas_bev.Canvas_BEV_heading_right(canvas_shape=((pc_range[4]-pc_range[1])*scale_bev, (pc_range[3]-pc_range[0])*scale_bev),
                                            canvas_x_range=(pc_range[0], pc_range[3]), 
                                            canvas_y_range=(pc_range[1], pc_range[4]),
                                            left_hand=left_hand) 

            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np) # Get Canvas Coords
            canvas.draw_canvas_points(canvas_xy[valid_mask]) # Only draw valid points
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)



        elif method == '3d':
            canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
            canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)

            # 这个points能够分成不同颜色！观察pcd-np
            canvas.draw_canvas_points(canvas_xy[valid_mask], colors=(0, 0, 255))
            if gt_box_tensor is not None:
                canvas.draw_boxes(gt_box_np,colors=(0,255,0), texts=gt_name)
            if pred_box_tensor is not None:
                canvas.draw_boxes(pred_box_np, colors=(255,0,0), texts=pred_name)

            # heterogeneous
            lidar_agent_record = infer_result.get("lidar_agent_record", None)
            cav_box_np = infer_result.get("cav_box_np", None)
            if lidar_agent_record is not None:
                cav_box_np = copy.deepcopy(cav_box_np)
                for i, islidar in enumerate(lidar_agent_record):
                    text = ['lidar'] if islidar else ['camera']
                    color = (0,191,255) if islidar else (255,185,15)
                    canvas.draw_boxes(cav_box_np[i:i+1], colors=color, texts=text)

        else:
            raise(f"Not Completed for f{method} visualization.")

        plt.axis("off")

        plt.imshow(canvas.canvas)
        plt.tight_layout()
        plt.savefig(save_path, transparent=False, dpi=900)
        plt.clf()
        plt.show()
        plt.close()


def visualize_color(infer_result, pcd, batchsize, pc_range, save_path, method='3d', left_hand=False):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    infer_result:
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        uncertainty_tensor : optional, torch.Tensor
            (N, ?)

        lidar_agent_record: optional, torch.Tensor
            (N_agnet, )


    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """
    plt.figure(figsize=[(pc_range[3] - pc_range[0]) / 40, (pc_range[4] - pc_range[1]) / 40])
    pc_range = [int(i) for i in pc_range]
    # fig, ax = plt.subplots()
    # fig.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0)

    color_index = []
    for i in range(batchsize):
        index = pcd[:,0].long() == i
        color_index.append(index.cpu().numpy())

    # 这个pcd需要剪裁
    pcd_np = pcd[:,1:4].cpu().numpy()



    # 为了展示不同视角差异，只保留绿色框
    # pred_box_tensor = infer_result.get("pred_box_tensor", None)
    pred_box_tensor = None
    gt_box_tensor = infer_result.get("gt_box_tensor", None)

    if pred_box_tensor is not None:
        pred_box_np = pred_box_tensor.cpu().numpy()
        pred_name = ['pred'] * pred_box_np.shape[0]

        score = infer_result.get("score_tensor", None)
        if score is not None:
            score_np = score.cpu().numpy()
            pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

        uncertainty = infer_result.get("uncertainty_tensor", None)
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6 ** 2 + 3.9 ** 2

            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}' \
                    for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}' \
                             for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}' \
                    for i in range(uncertainty_np.shape[0])]

    if gt_box_tensor is not None:
        gt_box_np = gt_box_tensor.cpu().numpy()
        gt_name = ['gt'] * gt_box_np.shape[0]

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand)

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords

        # 原版 注释掉
        # canvas.draw_canvas_points(canvas_xy[valid_mask])  # Only draw valid points

        canvas.draw_canvas_points_color(canvas_xy, valid_mask, color_index=color_index)

        if gt_box_tensor is not None:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name ,box_text_size=0)
        if pred_box_tensor is not None:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name , box_text_size=0)

        # heterogeneous
        lidar_agent_record = infer_result.get("lidar_agent_record", None)
        cav_box_np = infer_result.get("cav_box_np", None)
        if lidar_agent_record is not None:
            cav_box_np = copy.deepcopy(cav_box_np)
            for i, islidar in enumerate(lidar_agent_record):
                text = ['lidar'] if islidar else ['camera']
                color = (0, 191, 255) if islidar else (255, 185, 15)
                canvas.draw_boxes(cav_box_np[i:i + 1], colors=color, texts=text)



    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        # 剪裁操作是认可的，这个不会影响

        # 这个points能够分成不同颜色！观察pcd-np
        # canvas.draw_canvas_points(canvas_xy[valid_mask])
        print(color_index[0].shape,valid_mask.shape)
        canvas.draw_canvas_points_color(canvas_xy,valid_mask,color_index=color_index)

        if gt_box_tensor is not None:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name,box_text_size=0)
        if pred_box_tensor is not None:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name,box_text_size=0)

        # heterogeneous
        lidar_agent_record = infer_result.get("lidar_agent_record", None)
        cav_box_np = infer_result.get("cav_box_np", None)
        if lidar_agent_record is not None:
            cav_box_np = copy.deepcopy(cav_box_np)
            for i, islidar in enumerate(lidar_agent_record):
                text = ['lidar'] if islidar else ['camera']
                color = (0, 191, 255) if islidar else (255, 185, 15)
                canvas.draw_boxes(cav_box_np[i:i + 1], colors=color, texts=text)

    else:
        raise (f"Not Completed for f{method} visualization.")

    plt.axis("off")

    plt.imshow(canvas.canvas)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=1024)

    # plt.show()
    plt.clf()
    plt.close()

# 车内车外倒好说，mask都能解决，这个one_mask出现的是，只有车内的数据，但是还是要区分车，one_mask是车内*第几个车
def visualize_one_color(infer_result, pcd, color_select, pc_range, save_path, method='3d', left_hand=False):
    """
    Visualize the prediction, ground truth with point cloud together.
    They may be flipped in y axis. Since carla is left hand coordinate, while kitti is right hand.

    Parameters
    ----------
    infer_result:
        pred_box_tensor : torch.Tensor
            (N, 8, 3) prediction.

        gt_tensor : torch.Tensor
            (N, 8, 3) groundtruth bbx

        uncertainty_tensor : optional, torch.Tensor
            (N, ?)

        lidar_agent_record: optional, torch.Tensor
            (N_agnet, )


    pcd : torch.Tensor
        PointCloud, (N, 4).

    pc_range : list
        [xmin, ymin, zmin, xmax, ymax, zmax]

    save_path : str
        Save the visualization results to given path.

    dataset : BaseDataset
        opencood dataset object.

    method: str, 'bev' or '3d'

    """
    plt.figure(figsize=[(pc_range[3] - pc_range[0]) / 40, (pc_range[4] - pc_range[1]) / 40])
    pc_range = [int(i) for i in pc_range]

    # 透明背景
    # fig, ax = plt.subplots()
    # fig.patch.set_alpha(0.0)
    # ax.patch.set_alpha(0.0)

    # 这个pcd需要剪裁,有一种是这样剪裁
    pcd_np = pcd[:,:3].cpu().numpy()

    # 还有一种需要这样剪裁
    # pcd_np = pcd[:,1:4].cpu().numpy()


    # 为了展示不同视角差异，只保留绿色框
    # pred_box_tensor = infer_result.get("pred_box_tensor", None)
    # gt_box_tensor = infer_result.get("gt_box_tensor", None)
    pred_box_tensor = None
    gt_box_tensor = None

    if pred_box_tensor is not None:
        pred_box_np = pred_box_tensor.cpu().numpy()
        pred_name = ['pred'] * pred_box_np.shape[0]

        score = infer_result.get("score_tensor", None)
        if score is not None:
            score_np = score.cpu().numpy()
            pred_name = [f'score:{score_np[i]:.3f}' for i in range(score_np.shape[0])]

        uncertainty = infer_result.get("uncertainty_tensor", None)
        if uncertainty is not None:
            uncertainty_np = uncertainty.cpu().numpy()
            uncertainty_np = np.exp(uncertainty_np)
            d_a_square = 1.6 ** 2 + 3.9 ** 2

            if uncertainty_np.shape[1] == 3:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)
                # yaw angle is in radian, it's the same in g2o SE2's setting.

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:.3f} a_u:{uncertainty_np[i, 2]:.3f}' \
                    for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 2:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f}' \
                             for i in range(uncertainty_np.shape[0])]

            elif uncertainty_np.shape[1] == 7:
                uncertainty_np[:, :2] *= d_a_square
                uncertainty_np = np.sqrt(uncertainty_np)  # yaw angle is in radian

                pred_name = [
                    f'x_u:{uncertainty_np[i, 0]:.3f} y_u:{uncertainty_np[i, 1]:3f} a_u:{uncertainty_np[i, 6]:3f}' \
                    for i in range(uncertainty_np.shape[0])]

    if gt_box_tensor is not None:
        gt_box_np = gt_box_tensor.cpu().numpy()
        gt_name = ['gt'] * gt_box_np.shape[0]

    if method == 'bev':
        canvas = canvas_bev.Canvas_BEV_heading_right(
            canvas_shape=((pc_range[4] - pc_range[1]) * 10, (pc_range[3] - pc_range[0]) * 10),
            canvas_x_range=(pc_range[0], pc_range[3]),
            canvas_y_range=(pc_range[1], pc_range[4]),
            left_hand=left_hand)

        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)  # Get Canvas Coords

        # 原版 注释掉
        # canvas.draw_canvas_points(canvas_xy[valid_mask])  # Only draw valid points

        # canvas.draw_canvas_points_color(canvas_xy, valid_mask, color_index=color_index)

        canvas.draw_canvas_points_color_one(canvas_xy[valid_mask], color_select=color_select)

        if gt_box_tensor is not None:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name,box_text_size=0)
        if pred_box_tensor is not None:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name,box_text_size=0)

        # heterogeneous
        lidar_agent_record = infer_result.get("lidar_agent_record", None)
        cav_box_np = infer_result.get("cav_box_np", None)
        if lidar_agent_record is not None:
            cav_box_np = copy.deepcopy(cav_box_np)
            for i, islidar in enumerate(lidar_agent_record):
                text = ['lidar'] if islidar else ['camera']
                color = (0, 191, 255) if islidar else (255, 185, 15)
                canvas.draw_boxes(cav_box_np[i:i + 1], colors=color, texts=text)



    elif method == '3d':
        canvas = canvas_3d.Canvas_3D(left_hand=left_hand)
        canvas_xy, valid_mask = canvas.get_canvas_coords(pcd_np)
        # 剪裁操作是认可的，这个不会影响

        # 这个points能够分成不同颜色！观察pcd-np
        # canvas.draw_canvas_points(canvas_xy[valid_mask])
        # print(color_index[0].shape,valid_mask.shape)
        # canvas.draw_canvas_points_color(canvas_xy,valid_mask,color_index=color_index)
        #
        canvas.draw_canvas_points_color_one(canvas_xy[valid_mask],color_select=color_select)

        if gt_box_tensor is not None:
            canvas.draw_boxes(gt_box_np, colors=(0, 255, 0), texts=gt_name,box_text_size=0)
        if pred_box_tensor is not None:
            canvas.draw_boxes(pred_box_np, colors=(255, 0, 0), texts=pred_name,box_text_size=0)

        # heterogeneous
        lidar_agent_record = infer_result.get("lidar_agent_record", None)
        cav_box_np = infer_result.get("cav_box_np", None)
        if lidar_agent_record is not None:
            cav_box_np = copy.deepcopy(cav_box_np)
            for i, islidar in enumerate(lidar_agent_record):
                text = ['lidar'] if islidar else ['camera']
                color = (0, 191, 255) if islidar else (255, 185, 15)
                canvas.draw_boxes(cav_box_np[i:i + 1], colors=color, texts=text)

    else:
        raise (f"Not Completed for f{method} visualization.")

    plt.axis("off")

    plt.imshow(canvas.canvas)
    plt.tight_layout()
    plt.savefig(save_path, transparent=False, dpi=4096)
    plt.clf()
    plt.close()
    # plt.show()



def visualize_bev_features(bev_feature, save_dir='./bev_visualizations', n_cols=5):
    """
    Visualize Bird's Eye View (BEV) features.
    
    Args:
        bev_feature: Tensor of shape [N, C, H, W]
        save_dir: Directory to save visualizations
        n_cols: Number of columns in the grid plot
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    N, C, H, W = bev_feature.shape
    
    # Create a figure for each of the N BEV features
    for n in range(N):
        # Get the current BEV feature
        feature = bev_feature[n]  # Shape: [C, H, W]
        
        # Calculate how many rows we need
        n_rows = (C + n_cols - 1) // n_cols
        
        # Create a figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
        fig.suptitle(f'BEV Feature {n+1}/{N}', fontsize=16)
        
        # Flatten axes for easy indexing if multiple rows
        if n_rows > 1:
            axes = axes.flatten()
        
        # Loop through each channel
        for c in range(C):
            # Get the current channel
            channel_data = feature[c].detach().cpu().numpy()  # Shape: [H, W]
            
            # Normalize data for better visualization
            norm = Normalize(vmin=channel_data.min(), vmax=channel_data.max())
            
            # Get the corresponding axis
            if C <= n_cols and n_rows == 1:
                ax = axes[c] if n_cols > 1 else axes
            else:
                ax = axes[c]
            
            # Plot the feature as a heatmap
            im = ax.imshow(channel_data, cmap='viridis', norm=norm)
            ax.set_title(f'Channel {c}')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for c in range(C, n_rows * n_cols):
            if C <= n_cols and n_rows == 1:
                if c < n_cols:  # Check if we're still within the valid range
                    if n_cols > 1:
                        axes[c].axis('off')
            else:
                axes[c].axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f'{save_dir}/bev_feature_{n+1}.png', dpi=200)
        plt.close(fig)
        
def visualize_averaged_channels_individual(bev_feature, save_dir='./bev_avg_viz', vmin=None, vmax=None):
    os.makedirs(save_dir, exist_ok=True)
    
    N, C, H, W = bev_feature.shape

    # 确定网格布局（尽量接近正方形）
    grid_size = math.ceil(math.sqrt(N))
    rows = grid_size
    cols = math.ceil(N / rows)
    
    # 创建一个大图，包含所有BEV特征
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    gs = GridSpec(rows, cols, figure=fig)
    
    # 找到所有特征的全局最小值和最大值，用于统一颜色范围
    all_features = torch.mean(bev_feature, dim=1).detach().cpu().numpy()
    current_vmin = np.min(all_features)
    current_vmax = np.max(all_features)
    
    for n in range(N):
        # 计算该特征在网格中的位置
        row = n // cols
        col = n % cols
        
        # 在特定位置创建子图
        ax = fig.add_subplot(gs[row, col])
        
        # 获取通道平均特征
        feature_avg = all_features[n]
        
        # 使用统一的颜色范围
        if vmin is not None and vmax is not None:
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=current_vmin, vmax=current_vmax)
        im = ax.imshow(feature_avg, cmap='viridis', norm=norm)
        ax.set_title(f'BEV {n+1}')
        ax.axis('off')
    
    # 添加全局颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    # 设置整体标题
    fig.suptitle(f'Averaged BEV Features (N={N}, C={C}, H={W}, W={H})', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_dir, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Saved grid visualization to {save_dir}")