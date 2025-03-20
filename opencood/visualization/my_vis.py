from operator import gt
import numpy as np
import pickle
from pyquaternion import Quaternion
from matplotlib import pyplot as plt
from icecream import ic
from torch import margin_ranking_loss
from opencood.utils.box_utils import boxes_to_corners2d



def visualize(gt_tensor, pcd, save_path, dataset=None):
    """
    Visualize the ground truth with point cloud together.

    Parameters
    ----------
    gt_tensor : torch.Tensor
        (N, 7) groundtruth bbx, where 7 = [x, y, z, dx, dy, dz, heading]

    pcd : torch.Tensor
        PointCloud, (N, 4).

    save_path : str
        Save the visualization results to given path.

    """
    pcd_np = pcd
    gt_box_np = gt_tensor

    plt.figure(dpi=400)
    # draw point cloud. It's in lidar coordinate
    plt.scatter(pcd_np[:,0], pcd_np[:,1], s=0.5)

    N = gt_tensor.shape[0]
    corners = boxes_to_corners2d(gt_box_np,'hwl')
    plt.plot(corners[:, 0], corners[:, 1], c="r", linewidth=1, markersize=1.5)
    
    plt.savefig(save_path)
    plt.clf()
