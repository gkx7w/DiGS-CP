import torch
from torch import nn

from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils.box_utils import corner_to_center_torch, boxes_to_corners_3d, project_box3d, project_points_by_matrix_torch, get_mask_for_boxes_within_range_torch
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
        clusters, scores,cluster_indices,cur_cluster_id = self.clustering(data_dict)
        data_dict['boxes_fused'], data_dict['scores_fused'], data_dict['cluster_indices'], data_dict['cur_cluster_id']= \
            self.cluster_fusion(clusters, scores,cluster_indices,cur_cluster_id)
        self.merge_keypoints(data_dict)
        return data_dict

    
    def clustering(self, data_dict):
        """
        Assign predicted boxes to clusters according to their ious with each other
        """
        clusters_batch = []
        scores_batch = []
        cluster_indices_batch = []
        cur_cluster_id_batch = []
        record_len = [int(l) for l in data_dict['record_len']]
        lidar_poses = data_dict['lidar_pose'].cpu().numpy()
        for i, l in enumerate(record_len):         
            cur_boxes_list = data_dict['det_boxes'][sum(record_len[:i]):sum(record_len[:i])+l]
                
            # Added by Yifan Lu 
            if data_dict['proj_first'] is False:
                cur_boxes_list_ego = []
                # project bounding box to ego coordinate. [x,y,z,l,w,h,yaw]
                cur_boxes_list_ego.append(cur_boxes_list[0])
                for agent_id in range(1, l):
                    tfm = x1_to_x2(lidar_poses[sum(record_len[:i])+agent_id], 
                                   lidar_poses[sum(record_len[:i])])
                    tfm = torch.from_numpy(tfm).to(cur_boxes_list[0].device).float()
                    cur_boxes = cur_boxes_list[agent_id]
                    cur_corners = boxes_to_corners_3d(cur_boxes, order='hwl')
                    cur_corners_ego = project_box3d(cur_corners, tfm)
                    cur_boxes_ego = corner_to_center_torch(cur_corners_ego, order='hwl')
                    cur_boxes_list_ego.append(cur_boxes_ego)
                cur_boxes_list = cur_boxes_list_ego

            cur_scores_list = data_dict['det_scores'][sum(record_len[:i]):sum(record_len[:i])+l]
            cur_boxes_list = [b for b in cur_boxes_list if len(b) > 0]
            cur_scores_list = [s for s in cur_scores_list if len(s) > 0]         
            if len(cur_scores_list) == 0:
                clusters_batch.append([torch.Tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.57]).
                                      to(torch.device('cuda')).view(1, 7)])
                scores_batch.append([torch.Tensor([0.01]).to(torch.device('cuda')).view(-1)])
                cluster_indices_batch.append(torch.ones(1, device=torch.device('cuda'), dtype=torch.int))
                # 因为没有形成任何有效簇 (ID >= 1)，下一个可用 ID 是 1
                cur_cluster_id_batch.append(1)
                continue
      
            pred_boxes_cat = torch.cat(cur_boxes_list, dim=0)
            pred_boxes_cat[:, -1] = limit_period(pred_boxes_cat[:, -1])
            pred_scores_cat = torch.cat(cur_scores_list, dim=0)

             # 在调用boxes_iou3d_gpu前确保数据类型是float32
            pred_boxes_cat = pred_boxes_cat.float()  # 显式转换为Float类型
            ious = boxes_iou3d_gpu(pred_boxes_cat, pred_boxes_cat)
            
            
            cluster_indices = torch.zeros(len(ious)).int() # gt assignments of preds
            cur_cluster_id = 1
            while torch.any(cluster_indices == 0):
                # 找到第一个0出现的位置
                cur_idx = torch.where(cluster_indices == 0)[0][0] # find the idx of the first pred which is not assigned yet
                # 找到和它iou>0.1的所有box,给这些box索引上分，标注它属于第几个集合
                cluster_indices[torch.where(ious[cur_idx] > 0.1)[0]] = cur_cluster_id
                cur_cluster_id += 1
            clusters = []
            scores = []
            if cur_cluster_id == 1:  # 没有找到任何聚类
                # 可以选择默认将所有未分配的点分配到一个聚类
                cluster_indices[cluster_indices == 0] = 1
                cur_cluster_id = 2
            for j in range(1, cur_cluster_id):
                clusters.append(pred_boxes_cat[cluster_indices==j])
                scores.append(pred_scores_cat[cluster_indices==j])
            # 在这个上面可以添加一个feature  
            clusters_batch.append(clusters)
            scores_batch.append(scores)
            cluster_indices_batch.append(cluster_indices)
            cur_cluster_id_batch.append(cur_cluster_id)

        return clusters_batch, scores_batch,cluster_indices_batch,cur_cluster_id_batch

    def cluster_fusion(self, clusters, scores, cluster_indices_batch=None, cur_cluster_id_batch=None):
        """
        Merge boxes in each cluster with scores as weights for merging
        """
        boxes_fused = []
        scores_fused = []
        for cl, sl in zip(clusters, scores): # each frame
            for c, s in zip(cl, sl): # frame's cluster
                # reverse direction for non-dominant direction of boxes
                dirs = c[:, -1]
                if s.numel() == 0:
                    print("Empty tensor encountered, skipping fusion for this cluster.")
                    # 处理空张量情况
                    continue  # 直接返回，跳过这个聚类
                max_score_idx = torch.argmax(s)
                dirs_diff = torch.abs(dirs - dirs[max_score_idx].item())
                lt_pi = (dirs_diff > pi).int()
                dirs_diff = dirs_diff * (1 - lt_pi) + (
                            2 * pi - dirs_diff) * lt_pi
                score_lt_half_pi = s[dirs_diff > pi / 2].sum()  # larger than
                score_set_half_pi = s[
                    dirs_diff <= pi / 2].sum()  # small equal than
                # select larger scored direction as final direction
                if score_lt_half_pi <= score_set_half_pi:
                    dirs[dirs_diff > pi / 2] += pi
                else:
                    dirs[dirs_diff <= pi / 2] += pi
                dirs = limit_period(dirs)
                s_normalized = s / s.sum()
                sint = torch.sin(dirs) * s_normalized
                cost = torch.cos(dirs) * s_normalized
                theta = torch.atan2(sint.sum(), cost.sum()).view(1, )
                center_dim = c[:, :-1] * s_normalized[:, None]
                
                boxes_fused.append(torch.cat([center_dim.sum(dim=0), theta]))
                s_sorted = torch.sort(s, descending=True).values
                s_fused = 0
                for i, ss in enumerate(s_sorted):
                    s_fused += ss ** (i + 1)
                s_fused = torch.tensor([min(s_fused, 1.0)], device=s.device)
                scores_fused.append(s_fused)

        assert len(boxes_fused) > 0
        boxes_fused = torch.stack(boxes_fused, dim=0)
        len_records = [len(c) for c in clusters] # each frame
        boxes_fused = [
            boxes_fused[sum(len_records[:i]):sum(len_records[:i]) + l] for i, l
            in enumerate(len_records)]
        scores_fused = torch.stack(scores_fused, dim=0)
        scores_fused = [
            scores_fused[sum(len_records[:i]):sum(len_records[:i]) + l] for
            i, l in enumerate(len_records)]

        new_cluster_indices_batch = []
        new_cur_cluster_id_batch = []
        for i in range(len(boxes_fused)):
            # Get the mask for boxes within range
            corners3d = boxes_to_corners_3d(boxes_fused[i], order='hwl')
            mask = get_mask_for_boxes_within_range_torch(corners3d, self.pc_range)
            
            # Apply mask to boxes_fused, scores_fused, and clusters
            boxes_fused[i] = boxes_fused[i][mask]
            scores_fused[i] = scores_fused[i][mask]
            
            # Get the cluster IDs that correspond to valid boxes
            valid_cluster_ids = []
            for j, valid in enumerate(mask):
                if valid:
                    valid_cluster_ids.append(j + 1)  # Cluster IDs start from 1
            
            # Create a mapping from old cluster IDs to new ones
            old_to_new_id = {}
            new_id = 1
            for old_id in valid_cluster_ids:
                old_to_new_id[old_id] = new_id
                new_id += 1
            
            # Update cluster indices
            original_indices = cluster_indices_batch[i] #索引越界了
            new_indices = torch.zeros_like(original_indices)
            
            for old_id, new_id in old_to_new_id.items():
                new_indices[original_indices == old_id] = new_id
            
            new_cluster_indices_batch.append(new_indices)
            new_cur_cluster_id_batch.append(len(valid_cluster_ids) + 1)  # New maximum cluster ID
            # corners3d = boxes_to_corners_3d(boxes_fused[i], order='hwl')
            # mask = get_mask_for_boxes_within_range_torch(corners3d, self.pc_range)
            # boxes_fused[i] = boxes_fused[i][mask]
            # scores_fused[i] = scores_fused[i][mask]
            # clusters[i] = clusters[i][mask]
        

        return boxes_fused, scores_fused, new_cluster_indices_batch, new_cur_cluster_id_batch

    def merge_keypoints(self, data_dict):
        # merge keypoints
        kpts_feat_out = []
        kpts_coor_out = []
        kpts_coor_out_ego = []
        keypoints_features = data_dict['point_features'] # sum(record_len)
        keypoints_coords = data_dict['point_coords'] # [[N,3],...]
        idx = 0
        record_len = data_dict['record_len']
        lidar_poses = data_dict['lidar_pose'].cpu().numpy()
        for l in data_dict['record_len']:
            # if not project first, first transform the keypoints coords
            if data_dict['proj_first'] is False:
                kpts_coor_cur = []
                for agent_id in range(0, l):
                    tfm = x1_to_x2(lidar_poses[idx+agent_id], lidar_poses[idx])
                    tfm = torch.from_numpy(tfm).to(keypoints_coords[0].device).float()
                    keypoints_coords[idx+agent_id][:, :3] = project_points_by_matrix_torch(keypoints_coords[idx+agent_id][:,:3], tfm)

                kpts_coor_out_ego.append(
                    torch.cat(keypoints_coords[idx:l + idx], dim=0)
                )
                
            kpts_coor_out.append(
                torch.cat(keypoints_coords[idx:l + idx], dim=0))
            kpts_feat_out.append(
                torch.cat(keypoints_features[idx:l + idx], dim=0))
            idx += l
        # data_dict['point_features'] = kpts_feat_out
        # data_dict['point_coords'] = kpts_coor_out

        data_dict['merge_point_features'] = kpts_feat_out
        data_dict['merge_point_coords'] = kpts_coor_out

        if data_dict['proj_first'] is False:
            data_dict['merge_point_coords'] = kpts_coor_out_ego
