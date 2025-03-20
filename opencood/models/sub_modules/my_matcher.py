import torch
from torch import nn

from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils.box_utils import corner_to_center_torch, boxes_to_corners_3d, project_box3d, \
    project_points_by_matrix_torch, get_mask_for_boxes_within_range_torch
from opencood.utils.transformation_utils import x1_to_x2
from icecream import ic
import copy

pi = 3.141592653


def limit_period(val, offset=0.5, period=2 * pi):
    return val - torch.floor(val / period + offset) * period


class Matcher(nn.Module):
    """Correct localization error and use Algorithm 1:
     BBox matching with scores to fuse the proposal BBoxes"""

    def __init__(self, cfg, pc_range):
        super(Matcher, self).__init__()
        self.pc_range = pc_range

    @torch.no_grad()
    def forward(self, data_dict):
        data_dict['boxes_fused'], data_dict[
            'scores_fused'] = data_dict['det_boxes_fused'], data_dict['det_scores_fused']
        self.merge_keypoints(data_dict)
        return data_dict

    def merge_keypoints(self, data_dict):
        # merge keypoints
        kpts_feat_out = []
        kpts_coor_out = []
        kpts_coor_out_ego = []
        keypoints_features = data_dict['point_features']  # sum(record_len)
        keypoints_coords = data_dict['point_coords']  # [[N,3],...]
        idx = 0
        record_len = data_dict['record_len']
        lidar_poses = data_dict['lidar_pose'].cpu().numpy()

        # print("data_dict['proj_first']",data_dict['proj_first'])

        data_dict['proj_first'] = True
        for l in data_dict['record_len']:
            # Added by Yifan Lu
            # if not project first, first transform the keypoints coords
            if data_dict['proj_first'] is False:
                kpts_coor_cur = []
                for agent_id in range(0, l):
                    tfm = x1_to_x2(lidar_poses[idx + agent_id], lidar_poses[idx])
                    tfm = torch.from_numpy(tfm).to(keypoints_coords[0].device).float()
                    keypoints_coords[idx + agent_id][:, :3] = project_points_by_matrix_torch(
                        keypoints_coords[idx + agent_id][:, :3], tfm)

                kpts_coor_out_ego.append(
                    torch.cat(keypoints_coords[idx:l + idx], dim=0)
                )

            kpts_coor_out.append(
                torch.cat(keypoints_coords[idx:l + idx], dim=0))
            kpts_feat_out.append(
                torch.cat(keypoints_features[idx:l + idx], dim=0))
            idx += l
        data_dict['point_features'] = kpts_feat_out
        data_dict['point_coords'] = kpts_coor_out

        if data_dict['proj_first'] is False:
            # print("matcher 还是投影了")
            data_dict['point_coords'] = kpts_coor_out_ego
        # else:
        #     print("matcher 没有投影了")

        # 防止影响后面，所以变回去
        data_dict['proj_first'] = False

