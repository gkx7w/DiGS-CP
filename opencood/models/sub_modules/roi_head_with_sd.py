import copy
from icecream import ic
import torch.nn as nn
import torch
import numpy as np
from opencood.pcdet_utils.pointnet2.pointnet2_stack import \
    pointnet2_modules as pointnet2_stack_modules
from opencood.utils import common_utils
from opencood.pcdet_utils.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
from opencood.utils import box_utils

import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_grid_with_points_all_256_iou_score(x, y, boxes1, boxes2, iou, score, label, title):
    fig, axes = plt.subplots(1, 1, dpi=300)
    # plt.figure(dpi=100)
    plt.scatter(x, y, s=0.1, c='gray', alpha=0.7)
    # 绘制点云
    for index, box in enumerate(boxes1):

        # if iou[index]<0.7:

        x1, y1, width, height = box[0], box[1], box[3], box[4]

        if label[index] >= 1.0:

            rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="red",
                                     facecolor='none')
        else:
            rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="yellow",
                                     facecolor='none')

        plt.text(x1, y1, str(iou[index])[:5], ha='center', va='center', fontsize=3)

        plt.text(x1 - width, y1, str(score[index])[:5], ha='center', va='center', fontsize=3)
        # plt.text(x1+width, y1+height, str(label[index])[:5], ha='center', va='center')
        axes.add_patch(rect)

    for index, box in enumerate(boxes2):
        x1, y1, width, height = box[0], box[1], box[3], box[4]
        rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="green",
                                 facecolor='none')
        axes.add_patch(rect)

    plt.title("two box  256   " + title)
    plt.show()


def plot_grid_with_points_all_256(x, y, boxes1, boxes2, iou, label, title):
    fig, axes = plt.subplots(1, 1)
    # plt.figure(dpi=100)
    plt.scatter(x, y, s=0.1, c='gray', alpha=0.7)
    # 绘制点云
    for index, box in enumerate(boxes1):

        # if iou[index]<0.7:

        x1, y1, width, height = box[0], box[1], box[3], box[4]

        if label[index] >= 1.0:

            rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="red",
                                     facecolor='none')
        else:
            rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="yellow",
                                     facecolor='none')

        plt.text(x1, y1, str(iou[index])[:5], ha='center', va='center', fontsize=3)

        # plt.text(x1+width, y1+height, str(label[index])[:5], ha='center', va='center')
        axes.add_patch(rect)

    for index, box in enumerate(boxes2):
        x1, y1, width, height = box[0], box[1], box[3], box[4]
        rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="green",
                                 facecolor='none')
        axes.add_patch(rect)

    plt.title("two box  256   " + title)
    plt.show()


import torch
import numpy as np

pi = 3.141592653


def random_translation(box, max_translation=0.5):
    """ 对box进行随机平移 """
    translation = (torch.rand((box.shape[0], 3)).to(
        box.device) * 2 - 1) * max_translation  # -max_translation 到 +max_translation
    box[:, :3] += translation
    return box


def random_scaling(box, min_scale=0.8, max_scale=1.2):
    """ 对box进行随机缩放 """
    scale_factor = torch.empty((box.shape[0], 3)).uniform_(min_scale, max_scale).to(box.device)
    box[:, 3:6] *= scale_factor  # 只缩放尺寸
    return box


def random_rotation(box, max_angle=45.0):
    """ 对box进行随机旋转 """
    angle = (torch.rand((box.shape[0])).to(box.device) * 2 - 1) * max_angle  # -max_angle 到 +max_angle 度
    angle_rad = angle * (pi / 180)
    box[:, 6] += angle_rad  # 更新旋转角度
    # 注意：实际应用中可能还需要考虑旋转后的位置调整
    return box


# 数据增强函数
def augment_box(box):
    augmented_box = box.clone()
    augmented_box = random_translation(augmented_box)
    augmented_box = random_scaling(augmented_box)
    augmented_box = random_rotation(augmented_box)
    return augmented_box


# # 使用数据增强
# augmented_box = augment_box(box)
#
# print("Original Box:", box)
# print("Augmented Box:", augmented_box)


class RoIHead(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        input_channels = model_cfg['in_channels']
        self.code_size = 7

        # 调整分类头
        self.big_head = True
        # 对象可以选择 iou reg cls
        self.mv_loss_object = "all"
        # 计算loss方式，当前是 l1 计划加入 kl 散度
        self.mv_loss_compute = "l1"
        self.mv_loss_sigmod = False

        # test not need sd
        self.isTest = True

        self.use_convertor = True
        
        self.test_use_convertor = False

        # 确定是否需要添加额外监督信号,这个是全部正样本
        self.add_more_pos_supervision = True

        # 这个是添加增强的gt信息
        self.add_more_augmented_supervision = True

        # don't need, false is ok
        self.limit_mv_to_fused = False

        # s->f
        self.add_s_f = False

        # 统计方差,true,false
        self.compute_var = False
        self.write_cnt = 0
        self.feature_var_path = "/home/ypy/projects/CoAlign/opencood/save_score/compare_feature.txt"
        self.feature_var_list = []

        # s->s
        self.s_s_weight = 1
        
        # don't need, 0 is ok
        self.cls_box_weight = 0

        # 正负样本阈值
        self.score_threshold = 0.3

        mlps = copy.copy(self.model_cfg['roi_grid_pool']['mlps'])
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg['roi_grid_pool']['pool_radius'],
            nsamples=self.model_cfg['roi_grid_pool']['n_sample'],
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg['roi_grid_pool']['pool_method'],
        )

        grid_size = self.model_cfg['roi_grid_pool']['grid_size']
        self.grid_size = grid_size
        c_out = sum([x[-1] for x in mlps])
        pre_channel = grid_size * grid_size * grid_size * c_out
        # fc_layers = [self.model_cfg['n_fc_neurons']] * 2

        # big 分类头
        if self.big_head:
            fc_layers = [self.model_cfg['n_fc_neurons'] * 4, self.model_cfg['n_fc_neurons'] * 2]
        else:
            fc_layers = [self.model_cfg['n_fc_neurons']] * 2

        # 还是用伪分类头和公共提取模块
        pre_channel1 = grid_size * grid_size * grid_size * c_out
        self.fake_shared_fc_layers, pre_channel1 = self._make_fc_layers(pre_channel1,
                                                                        fc_layers)
        self.fake_cls_head, pre_channel1 = self._make_fc_layers(pre_channel1,
                                                                fc_layers,
                                                                output_channels=
                                                                self.model_cfg[
                                                                    'num_cls'])
        self.fake_reg_layers, _ = self._make_fc_layers(pre_channel1, fc_layers,
                                                       output_channels=
                                                       self.model_cfg[
                                                           'num_cls'] * 7)

        pre_channel = grid_size * grid_size * grid_size * c_out

        self.shared_fc_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                                  fc_layers)

        # 新增 mv->fv converter
        # self.convertor = None
        self.convertor, pre_channel = self._make_fc_layers(pre_channel, fc_layers)

        self.cls_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                            fc_layers,
                                                            output_channels=
                                                            self.model_cfg[
                                                                'num_cls'])
        self.iou_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=
                                                  self.model_cfg['num_cls'])
        self.reg_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=
                                                  self.model_cfg[
                                                      'num_cls'] * 7)

        self._init_weights(weight_init='xavier')

    def _init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def _make_fc_layers(self, input_channels, fc_list, output_channels=None):
        fc_layers = []
        pre_channel = input_channels
        for k in range(len(fc_list)):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                # nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg['dp_ratio'] > 0:
                fc_layers.append(nn.Dropout(self.model_cfg['dp_ratio']))
        if output_channels is not None:
            fc_layers.append(
                nn.Conv1d(pre_channel, output_channels, kernel_size=1,
                          bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers, pre_channel

    def get_global_grid_points_of_roi(self, rois):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        # (B, 6x6x6, 3)
        local_roi_grid_points = self.get_dense_grid_points(rois,
                                                           batch_size_rcnn,
                                                           self.grid_size)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        """
        Get the local coordinates of each grid point of a roi in the coordinate
        system of the roi(origin lies in the center of this roi.
        """
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = torch.stack(torch.where(faked_features),
                                dim=1)  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1,
                                     1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (
                                  dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(
            dim=1) \
                          - (local_roi_size.unsqueeze(
            dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def assign_targets(self, batch_dict):
        batch_dict['rcnn_label_dict'] = {
            'rois': [],
            'gt_of_rois': [],
            'gt_of_rois_src': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
            'rois_anchor': [],
            'record_len': [],
            'rois_scores_stage1': [],
            # 新增
            'gt_box_show': []
        }
        pred_boxes = batch_dict['boxes_fused']
        pred_scores = batch_dict['scores_fused']
        gt_boxes = [b[m][:, [0, 1, 2, 5, 4, 3, 6]].float() for b, m in
                    zip(batch_dict['object_bbx_center'],
                        batch_dict['object_bbx_mask'].bool())]  # hwl -> lwh order

        batch_dict['gt_boxes_len'] = [t.shape[0] for t in gt_boxes]
        batch_dict['gt_boxes_list'] = gt_boxes
        batch_dict['gt_boxes'] = torch.cat(gt_boxes, dim=0).detach()

        for rois, scores, gts in zip(pred_boxes, pred_scores, gt_boxes):  # each frame
            rois = rois[:, [0, 1, 2, 5, 4, 3, 6]]  # hwl -> lwh
            if rois.shape[0] == 0:
                rois = torch.tensor([[0, 0, 0, 4, 2, 2, 0.0]], device=gts.device)
            if gts.shape[0] == 0:
                # 这句话有点意思，emm，因为这样会让这一帧全部变成1
                gts = rois.clone()

            gt_box_show = gts.clone().detach()

            # print(rois)
            # print(gts)
            ious = boxes_iou3d_gpu(rois, gts)
            # print(ious)
            max_ious, gt_inds = ious.max(dim=1)
            gt_of_rois = gts[gt_inds]
            # rcnn_labels = (max_ious > 0.3).float()

            rcnn_labels = (max_ious > self.score_threshold).float()

            mask = torch.logical_not(rcnn_labels.bool())

            # set negative samples back to rois, no correction in stage2 for them
            gt_of_rois[mask] = rois[mask]
            gt_of_rois_src = gt_of_rois.clone().detach()

            # canoical transformation
            roi_center = rois[:, 0:3]
            # TODO: roi_ry > 0 in pcdet
            roi_ry = rois[:, 6] % (2 * np.pi)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]),
                angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = (gt_of_rois[:, 6] + (
                    torch.div(torch.abs(gt_of_rois[:, 6].min()),
                              (2 * np.pi), rounding_mode='trunc')
                    + 1) * 2 * np.pi) % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                    heading_label < np.pi * 1.5)

            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            heading_label[opposite_flag] = (heading_label[
                                                opposite_flag] + np.pi) % (
                                                   2 * np.pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[
                                      flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2,
                                        max=np.pi / 2)
            gt_of_rois[:, 6] = heading_label

            # generate regression target
            rois_anchor = rois.clone().detach().view(-1, self.code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            reg_targets = box_utils.box_encode(
                gt_of_rois.view(-1, self.code_size), rois_anchor
            )

            batch_dict['rcnn_label_dict']['rois'].append(rois)
            batch_dict['rcnn_label_dict']['rois_scores_stage1'].append(scores)
            batch_dict['rcnn_label_dict']['gt_of_rois'].append(gt_of_rois)
            batch_dict['rcnn_label_dict']['gt_of_rois_src'].append(
                gt_of_rois_src)
            batch_dict['rcnn_label_dict']['cls_tgt'].append(rcnn_labels)
            batch_dict['rcnn_label_dict']['reg_tgt'].append(reg_targets)
            batch_dict['rcnn_label_dict']['iou_tgt'].append(max_ious)
            batch_dict['rcnn_label_dict']['rois_anchor'].append(rois_anchor)
            batch_dict['rcnn_label_dict']['record_len'].append(rois.shape[0])

            batch_dict['rcnn_label_dict']['gt_box_show'].append(gt_box_show)

        # cat list to tensor
        for k, v in batch_dict['rcnn_label_dict'].items():
            if k == 'record_len':
                continue
            batch_dict['rcnn_label_dict'][k] = torch.cat(v, dim=0)

        return batch_dict

    def roi_grid_pool(self, batch_dict):
        batch_size = len(batch_dict['record_len'])
        rois = batch_dict['rcnn_label_dict']['rois']
        point_coords = batch_dict['merge_point_coords']
        point_features = batch_dict['merge_point_features']
        label_record_len = batch_dict['rcnn_label_dict']['record_len']

        point_features = torch.cat(point_features, dim=0)
        # (BxN, 6x6x6, 3)
        global_roi_grid_points, local_roi_grid_points = \
            self.get_global_grid_points_of_roi(rois)
        # (B, Nx6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)

        xyz = torch.cat(point_coords, dim=0)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = len(point_coords[bs_idx])
        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            new_xyz_batch_cnt[bs_idx] = label_record_len[
                                            bs_idx] * self.grid_size ** 3

        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz[:, :3].contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz[:, :3].contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),  # weighted point features
        )  # (M1 + M2 ..., C)
        # (BxN, 6x6x6, C)
        # print("pooled_feature1:", pooled_features.shape)
        # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(-1, self.grid_size ** 3,
                                               pooled_features.shape[-1])
        # print("pooled_feature2:", pooled_features.shape)

        return pooled_features

    def gt_roi_grid_pool(self, batch_dict, rois):
        batch_size = len(batch_dict['gt_boxes_len'])


        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        label_record_len = batch_dict['gt_boxes_len']
        Multi_view_box_index = batch_dict['Multi_view_box_index']
        mv_len = batch_dict['record_len']

        tmp_rois, start_index, start_rois, tmp_rois_len = [], 0, 0, []
        for i, v in enumerate(mv_len):
            for j in range(v):
                tmp = []
                for k in range(label_record_len[i]):
                    if Multi_view_box_index[start_index + j][k] == 1:
                        tmp.append(rois[start_rois + k].unsqueeze(0))
                if len(tmp) > 0:
                    tmp_rois.append(torch.cat(tmp, dim=0))
                    tmp_rois_len.append(len(tmp))
                else:
                    print("为啥会是0？", Multi_view_box_index[start_index + j].sum())
                    tmp_rois_len.append(0)
            start_index += v
            start_rois += label_record_len[i]

        rois = torch.cat(tmp_rois, dim=0)
        label_record_len = tmp_rois_len
        batch_size = len(label_record_len)
        batch_dict['gt_box_record_len'] = label_record_len


        point_features = torch.cat(point_features, dim=0)
        # (BxN, 6x6x6, 3)
        global_roi_grid_points, local_roi_grid_points = \
            self.get_global_grid_points_of_roi(rois)

        xyz = torch.cat(point_coords, dim=0)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()


        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = len(point_coords[bs_idx])
        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            new_xyz_batch_cnt[bs_idx] = label_record_len[
                                            bs_idx] * self.grid_size ** 3

        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz[:, :3].contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz[:, :3].contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),  # weighted point features
        )  # (M1 + M2 ..., C)

        # print("gt_pooled_feature1:",pooled_features.shape)
        # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(-1, self.grid_size ** 3,
                                               pooled_features.shape[-1])
        # print("gt_pooled_feature2:",pooled_features.shape)

        return pooled_features

    def gt_fused_roi_grid_pool(self, batch_dict, rois):
        batch_size = len(batch_dict['record_len'])
        # 加不加
        # rois = batch_dict['gt_boxes']
        point_coords = batch_dict['merge_point_coords']
        point_features = batch_dict['merge_point_features']
        label_record_len = batch_dict['gt_boxes_len']

        point_features = torch.cat(point_features, dim=0)
        # (BxN, 6x6x6, 3)
        global_roi_grid_points, local_roi_grid_points = \
            self.get_global_grid_points_of_roi(rois)
        # (B, Nx6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)

        xyz = torch.cat(point_coords, dim=0)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            xyz_batch_cnt[bs_idx] = len(point_coords[bs_idx])
        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            new_xyz_batch_cnt[bs_idx] = label_record_len[
                                            bs_idx] * self.grid_size ** 3

        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz[:, :3].contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz[:, :3].contiguous(),
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),  # weighted point features
        )  # (M1 + M2 ..., C)
        # (BxN, 6x6x6, C)
        # print("pooled_feature1:", pooled_features.shape)
        # (BxN, 6x6x6, C)
        pooled_features = pooled_features.view(-1, self.grid_size ** 3,
                                               pooled_features.shape[-1])
        # print("pooled_feature_fused:", pooled_features.shape)

        return pooled_features

    def assign_targets_mv(self, batch_dict):
        batch_dict['rcnn_label_dict_mv'] = {}

        label_cnt = batch_dict['Multi_view_box_index'].sum().int().item()
        cls_tgt_mv = torch.ones((label_cnt, 1)).to(batch_dict['Multi_view_box_index'].device)
        batch_dict['rcnn_label_dict_mv']["cls_tgt_mv"] = cls_tgt_mv

        cls_tgt_fv = torch.ones((sum(batch_dict['gt_boxes_len']), 1)).to(batch_dict['Multi_view_box_index'].device)
        batch_dict['rcnn_label_dict_mv']["cls_tgt_fv"] = cls_tgt_fv

        return batch_dict

    def assign_targets_fv_more(self, batch_dict):
        batch_dict['rcnn_label_dict_fv_more'] = {
            'rois': [],
            'gt_of_rois': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
        }
        gt_boxes_len = batch_dict['gt_boxes_len']
        gt_boxes = batch_dict['gt_boxes_list']
        # print([t.shape for t in gt_boxes])
        for gts in gt_boxes:  # each frame


            rois = augment_box(gts)
            ious = boxes_iou3d_gpu(rois, gts)
            max_ious = torch.diag(ious)

            mask = torch.logical_not(rcnn_labels.bool())

            # set negative samples back to rois, no correction in stage2 for them
            gt_of_rois = gts
            gt_of_rois[mask] = rois[mask]
            gt_of_rois_src = gt_of_rois.clone().detach()

            # canoical transformation
            roi_center = rois[:, 0:3]
            # TODO: roi_ry > 0 in pcdet
            roi_ry = rois[:, 6] % (2 * np.pi)
            gt_of_rois[:, 0:3] = gt_of_rois[:, 0:3] - roi_center
            gt_of_rois[:, 6] = gt_of_rois[:, 6] - roi_ry

            # transfer LiDAR coords to local coords
            gt_of_rois = common_utils.rotate_points_along_z(
                points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]),
                angle=-roi_ry.view(-1)
            ).view(-1, gt_of_rois.shape[-1])

            # flip orientation if rois have opposite orientation
            heading_label = (gt_of_rois[:, 6] + (
                    torch.div(torch.abs(gt_of_rois[:, 6].min()),
                              (2 * np.pi), rounding_mode='trunc')
                    + 1) * 2 * np.pi) % (2 * np.pi)  # 0 ~ 2pi
            opposite_flag = (heading_label > np.pi * 0.5) & (
                    heading_label < np.pi * 1.5)

            # (0 ~ pi/2, 3pi/2 ~ 2pi)
            heading_label[opposite_flag] = (heading_label[
                                                opposite_flag] + np.pi) % (
                                                   2 * np.pi)
            flag = heading_label > np.pi
            heading_label[flag] = heading_label[
                                      flag] - np.pi * 2  # (-pi/2, pi/2)
            heading_label = torch.clamp(heading_label, min=-np.pi / 2,
                                        max=np.pi / 2)
            gt_of_rois[:, 6] = heading_label

            # generate regression target
            rois_anchor = rois.clone().detach().view(-1, self.code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0

            reg_targets = box_utils.box_encode(
                gt_of_rois.view(-1, self.code_size), rois_anchor
            )

            batch_dict['rcnn_label_dict_fv_more']['rois'].append(rois)
            batch_dict['rcnn_label_dict_fv_more']['gt_of_rois'].append(gt_of_rois)
            batch_dict['rcnn_label_dict_fv_more']['cls_tgt'].append(rcnn_labels)
            batch_dict['rcnn_label_dict_fv_more']['reg_tgt'].append(reg_targets)
            batch_dict['rcnn_label_dict_fv_more']['iou_tgt'].append(max_ious)

        # cat list to tensor
        for k, v in batch_dict['rcnn_label_dict_fv_more'].items():
            if k == 'record_len':
                continue
            batch_dict['rcnn_label_dict_fv_more'][k] = torch.cat(v, dim=0)

        return batch_dict

    def assign_targets_mv_more(self, batch_dict):
        batch_dict['rcnn_label_dict_mv_more'] = {
            'rois': [],
            'gt_of_rois': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
        }

        rcnn_label_fv_more = batch_dict['rcnn_label_dict_fv_more']
        # print([[k, v.shape] for k, v in rcnn_label_fv_more.items()])
        label_record_len = batch_dict['gt_boxes_len']
        Multi_view_box_index = batch_dict['Multi_view_box_index']
        mv_len = batch_dict['record_len']


        gt_boxes_len = batch_dict['gt_boxes_len']
        gt_boxes = batch_dict['gt_boxes']
        start_index, start_rois, tmp_rois_len = 0, 0, []
        for i, v in enumerate(mv_len):
            for j in range(v):
                tmp_rois_now, tmp_gt_of_rois_now, tmp_cls_tgt_now, tmp_reg_tgt_now, tmp_iou_tgt_now = [], [], [], [], []
                for k in range(label_record_len[i]):
                    if Multi_view_box_index[start_index + j][k] == 1:
                        tmp_rois_now.append(rcnn_label_fv_more['rois'][start_rois + k].unsqueeze(0))
                        tmp_gt_of_rois_now.append(rcnn_label_fv_more['gt_of_rois'][start_rois + k].unsqueeze(0))
                        tmp_cls_tgt_now.append(rcnn_label_fv_more['cls_tgt'][start_rois + k].unsqueeze(0))
                        tmp_reg_tgt_now.append(rcnn_label_fv_more['reg_tgt'][start_rois + k].unsqueeze(0))
                        tmp_iou_tgt_now.append(rcnn_label_fv_more['iou_tgt'][start_rois + k].unsqueeze(0))
                if len(tmp_rois_now) > 0:
                    # tmp_rois.append(torch.cat(tmp_rois_now,dim=0))
                    # tmp_gt_of_rois.append(torch.cat(tmp_gt_of_rois_now,dim=0))
                    # tmp_cls_tgt.append(torch.cat(tmp_cls_tgt_now,dim=0))
                    # tmp_reg_tgt.append(torch.cat(tmp_reg_tgt_now,dim=0))
                    # tmp_iou_tgt.append(torch.cat(tmp_iou_tgt_now,dim=0))

                    batch_dict['rcnn_label_dict_mv_more']['rois'].append(torch.cat(tmp_rois_now, dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['gt_of_rois'].append(torch.cat(tmp_gt_of_rois_now, dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['cls_tgt'].append(torch.cat(tmp_cls_tgt_now, dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['reg_tgt'].append(torch.cat(tmp_reg_tgt_now, dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['iou_tgt'].append(torch.cat(tmp_iou_tgt_now, dim=0))

                    tmp_rois_len.append(len(tmp_rois_now))
                else:
                    tmp_rois_len.append(0)
            start_index += v
            start_rois += label_record_len[i]

        # cat list to tensor
        for k, v in batch_dict['rcnn_label_dict_mv_more'].items():
            if k == 'record_len':
                continue
            batch_dict['rcnn_label_dict_mv_more'][k] = torch.cat(v, dim=0)
        # print(batch_dict['rcnn_label_dict_mv_more']['iou_tgt'])
        # print(batch_dict['rcnn_label_dict_mv_more']['iou_tgt'].shape)
        return batch_dict

    def forward(self, batch_dict):


        batch_dict = self.assign_targets(batch_dict)


        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, self.grid_size,
                              self.grid_size,
                              self.grid_size)  # (BxN, C, 6, 6, 6)
        shared_features = self.shared_fc_layers(
            pooled_features.view(batch_size_rcnn, -1, 1))

        if self.test_use_convertor:
            shared_features = self.convertor(shared_features)

        rcnn_cls = self.cls_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, 1 or 2)
        rcnn_iou = self.iou_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, 1)
        rcnn_reg = self.reg_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, C)


        batch_dict['stage2_out'] = {
            'rcnn_cls': rcnn_cls,
            'rcnn_iou': rcnn_iou,
            'rcnn_reg': rcnn_reg,
        }

        if self.isTest:
            return batch_dict

        mv_box_loss = torch.tensor(0.).to(rcnn_cls.device)
        cls_box_loss = torch.tensor(0.).to(rcnn_cls.device)
        loss_cls_gt_mv = torch.tensor(0.).to(rcnn_cls.device)
        loss_cls_gt_fused = torch.tensor(0.).to(rcnn_cls.device)

        judge = True
        for t in batch_dict['gt_boxes_len']:
            if t == 0:
                judge = False
        if batch_dict['Multi_view_box_index'].sum() > 0 and sum(batch_dict['gt_boxes_len']) > 0 and judge:
            # if sum(batch_dict['gt_boxes_len']) > 0 and (
            #             self.s_s_weight > 0 or self.cls_box_weight > 0):
            # gt fused
            pooled_features_gt_fused = self.gt_fused_roi_grid_pool(batch_dict, batch_dict['gt_boxes'])
            batch_size_gt = pooled_features_gt_fused.shape[0]
            pooled_features_gt_fused = pooled_features_gt_fused.permute(0, 2, 1). \
                contiguous().view(batch_size_gt, -1, self.grid_size,
                                  self.grid_size,
                                  self.grid_size)  # (BxN, C, 6, 6, 6)

            shared_features_gt_fused = self.shared_fc_layers(
                pooled_features_gt_fused.view(batch_size_gt, -1, 1))

            rcnn_cls_gt_fused = self.cls_layers(shared_features_gt_fused).transpose(1,
                                                                                    2).contiguous().squeeze(
                dim=1)
            rcnn_reg_gt_fused = self.reg_layers(shared_features_gt_fused).transpose(1,
                                                                                    2).contiguous().squeeze(
                dim=1)
            rcnn_iou_gt_fused = self.iou_layers(shared_features_gt_fused).transpose(1,
                                                                                    2).contiguous().squeeze(
                dim=1)


            # gt mv
            pooled_features_gt = self.gt_roi_grid_pool(batch_dict, batch_dict['gt_boxes'])

            batch_size_gt = pooled_features_gt.shape[0]
            pooled_features_gt = pooled_features_gt.permute(0, 2, 1). \
                contiguous().view(batch_size_gt, -1, self.grid_size,
                                  self.grid_size,
                                  self.grid_size)  # (BxN, C, 6, 6, 6)
            # true cls
            shared_features_gt = self.shared_fc_layers(
                pooled_features_gt.view(batch_size_gt, -1, 1))

            # 添加转换器
            if self.use_convertor:
                shared_features_gt = self.convertor(shared_features_gt)

            rcnn_cls_gt = self.cls_layers(shared_features_gt).transpose(1,
                                                                        2).contiguous().squeeze(
                dim=1)  # (B, 1 or 2)


            rcnn_reg_gt = self.reg_layers(shared_features_gt).transpose(1,
                                                                        2).contiguous().squeeze(
                dim=1)  # (B, C)
            rcnn_iou_gt = self.iou_layers(shared_features_gt).transpose(1,
                                                                        2).contiguous().squeeze(
                dim=1)  # (B, C)

            batch_dict['rcnn_iou_gt'] = rcnn_iou_gt
            batch_dict['rcnn_cls_gt'] = rcnn_cls_gt


            if self.add_more_augmented_supervision:
                batch_dict = self.assign_targets_fv_more(batch_dict)
                pooled_features_gt_fused_more = self.gt_fused_roi_grid_pool(batch_dict,
                                                                            batch_dict['rcnn_label_dict_fv_more'][
                                                                                'rois'])
                batch_size_gt = pooled_features_gt_fused_more.shape[0]
                pooled_features_gt_fused_more = pooled_features_gt_fused_more.permute(0, 2, 1). \
                    contiguous().view(batch_size_gt, -1, self.grid_size,
                                      self.grid_size,
                                      self.grid_size)  # (BxN, C, 6, 6, 6)

                shared_features_gt_fused_more = self.shared_fc_layers(
                    pooled_features_gt_fused_more.view(batch_size_gt, -1, 1))
                # 这是fused_gt对应的分数，如果需要用的话，还要把一些信息加上，因为融合后和融合前要一一对齐
                rcnn_cls_gt_fused_more = self.cls_layers(shared_features_gt_fused_more).transpose(1,
                                                                                                  2).contiguous().squeeze(
                    dim=1)
                rcnn_reg_gt_fused_more = self.reg_layers(shared_features_gt_fused_more).transpose(1,
                                                                                                  2).contiguous().squeeze(
                    dim=1)
                rcnn_iou_gt_fused_more = self.iou_layers(shared_features_gt_fused_more).transpose(1,
                                                                                                  2).contiguous().squeeze(
                    dim=1)
                batch_dict['stage2_out']['rcnn_cls_fv_more'] = rcnn_cls_gt_fused_more
                batch_dict['stage2_out']['rcnn_iou_fv_more'] = rcnn_iou_gt_fused_more
                batch_dict['stage2_out']['rcnn_reg_fv_more'] = rcnn_reg_gt_fused_more

                # 单视角
                batch_dict = self.assign_targets_mv_more(batch_dict)
                pooled_features_gt_more = self.gt_roi_grid_pool(batch_dict,
                                                                batch_dict['rcnn_label_dict_fv_more']['rois'])
                batch_size_gt = pooled_features_gt_more.shape[0]
                pooled_features_gt_more = pooled_features_gt_more.permute(0, 2, 1). \
                    contiguous().view(batch_size_gt, -1, self.grid_size,
                                      self.grid_size,
                                      self.grid_size)  # (BxN, C, 6, 6, 6)

                shared_features_gt_more = self.shared_fc_layers(
                    pooled_features_gt_more.view(batch_size_gt, -1, 1))

                # 添加转换器
                if self.use_convertor:
                    shared_features_gt_more = self.convertor(shared_features_gt_more)


                rcnn_cls_gt_more = self.cls_layers(shared_features_gt_more).transpose(1,
                                                                                      2).contiguous().squeeze(
                    dim=1)
                rcnn_reg_gt_more = self.reg_layers(shared_features_gt_more).transpose(1,
                                                                                      2).contiguous().squeeze(
                    dim=1)
                rcnn_iou_gt_more = self.iou_layers(shared_features_gt_more).transpose(1,
                                                                                      2).contiguous().squeeze(
                    dim=1)
                batch_dict['stage2_out']['rcnn_cls_mv_more'] = rcnn_cls_gt_more
                batch_dict['stage2_out']['rcnn_iou_mv_more'] = rcnn_iou_gt_more
                batch_dict['stage2_out']['rcnn_reg_mv_more'] = rcnn_reg_gt_more

            # 添加额外监督，这里不用sigmod，nn里面自带了，这里只能算真正的loss，其他的加强loss要放到原本的loss上
            if self.add_more_pos_supervision:
                # 这里是纯粹的正样本
                batch_dict = self.assign_targets_mv(batch_dict)
                cls_gt_mv = rcnn_cls_gt.view(1, -1, 1)
                cls_gt_fv = rcnn_cls_gt_fused.view(1, -1, 1)
                cls_tgt_gt_mv = batch_dict['rcnn_label_dict_mv']["cls_tgt_mv"].view(1, -1, 1)
                cls_tgt_gt_fv = batch_dict['rcnn_label_dict_mv']["cls_tgt_fv"].view(1, -1, 1)

                loss_cls_gt_mv = weighted_sigmoid_binary_cross_entropy(cls_gt_mv, cls_tgt_gt_mv)
                loss_cls_gt_fused = weighted_sigmoid_binary_cross_entropy(cls_gt_fv, cls_tgt_gt_fv)

            if self.s_s_weight > 0 or self.cls_box_weight > 0:
                if self.mv_loss_sigmod:
                    rcnn_cls_gt = rcnn_cls_gt.sigmoid().view(-1)
                    rcnn_cls_gt_fused = rcnn_cls_gt_fused.sigmoid().view(-1)
                    rcnn_iou_gt = rcnn_iou_gt.sigmoid().view(-1)
                    rcnn_iou_gt_fused = rcnn_iou_gt_fused.sigmoid().view(-1)
                else:
                    rcnn_cls_gt = rcnn_cls_gt.view(-1)
                    rcnn_cls_gt_fused = rcnn_cls_gt_fused.view(-1)
                    rcnn_iou_gt = rcnn_iou_gt.view(-1)
                    rcnn_iou_gt_fused = rcnn_iou_gt_fused.view(-1)

                # print("mv output:")
                # print(rcnn_cls_gt)
                # print("fv output:")
                # print(rcnn_cls_gt_fused)

                cls_box_data_1 = rcnn_cls_gt.unsqueeze(1).repeat(1, rcnn_reg_gt.shape[0], 1)
                cls_box_data_2 = rcnn_cls_gt.unsqueeze(0).repeat(rcnn_reg_gt.shape[0], 1, 1)

                # print("计算cls_box信息",cls_box_data_1.shape,cls_box_data_2.shape)
                # print(rcnn_reg_gt)
                # print(rcnn_cls_gt)
                # print(rcnn_iou_gt)

                if rcnn_cls_gt.shape[0] > 1:
                    cls_box_loss = torch.norm(cls_box_data_1 - cls_box_data_2, p=1).sum() * self.cls_box_weight / (
                            rcnn_cls_gt.shape[0] * (rcnn_cls_gt.shape[0] - 1))

                # 第一步，根据个数将其拆分开
                car_box_num = batch_dict['gt_box_record_len']
                start, mv_box_loss_data = 0, []
                # bs*car*box,1 -> box1,1  box2,1 …… boxn,1  这个是按照car去拆分
                for t in car_box_num:
                    if self.compute_var:
                        # 为了统计方差，所记录的features 数据
                        mv_box_loss_data.append(shared_features_gt[start:start + t])
                    else:
                        if self.mv_loss_object == "all":
                            mv_box_loss_data.append(shared_features_gt[start:start + t])
                        elif self.mv_loss_object == "cls":
                            mv_box_loss_data.append(rcnn_cls_gt[start:start + t])
                        elif self.mv_loss_object == "reg":
                            mv_box_loss_data.append(rcnn_reg_gt[start:start + t])
                        elif self.mv_loss_object == "iou":
                            mv_box_loss_data.append(rcnn_iou_gt[start:start + t])
                        else:
                            mv_box_loss_data.append(rcnn_cls_gt[start:start + t])

                    start += t

                bs_box_num = batch_dict['gt_boxes_len']
                start, fused_box_loss_data = 0, []
                for t in bs_box_num:
                    if self.mv_loss_object == "all":
                        fused_box_loss_data.append(shared_features_gt_fused[start:start + t])
                    elif self.mv_loss_object == "cls":
                        fused_box_loss_data.append(rcnn_cls_gt_fused[start:start + t])
                    elif self.mv_loss_object == "reg":
                        fused_box_loss_data.append(rcnn_reg_gt_fused[start:start + t])
                    elif self.mv_loss_object == "iou":
                        fused_box_loss_data.append(rcnn_iou_gt_fused[start:start + t])
                    else:
                        fused_box_loss_data.append(rcnn_cls_gt_fused[start:start + t])

                    start += t

                box_loss_data_bs, box_index, point_num_bs = [], [], []
                bs_len = batch_dict['record_len']


                points_in_boxes = batch_dict['Multi_view_box_points']
                Multi_view_box_index = batch_dict['Multi_view_box_index']
                start_bs = 0
                # n1,1  n2,1 …… nx,1 - > [n1,1  n2,1],[n3,1 n4,1 …… nx,1]  将这个car按照bs组合起来
                for t in bs_len:
                    tmp1, tmp2, tmp3 = [], [], []
                    for j in range(t):
                        tmp1.append(mv_box_loss_data[start_bs + j])
                        tmp2.append(Multi_view_box_index[start_bs + j])
                        tmp3.append(points_in_boxes[start_bs + j])

                    point_num_bs.append(tmp3)
                    box_loss_data_bs.append(tmp1)
                    box_index.append(tmp2)
                    start_bs += t

                box_loss_data_mv, point_num_mv = [], []
                label_record_len = batch_dict['gt_boxes_len']
                for bs_fused_data, bs_data, bs_num, bs_index, bs_box_len in zip(fused_box_loss_data, box_loss_data_bs,
                                                                                point_num_bs, box_index,
                                                                                label_record_len):
                    tmp = [[] for t in range(bs_box_len)]
                    tmp_num = [[] for t in range(bs_box_len)]

                    for index, object_list in enumerate(tmp):
                        object_list.append(bs_fused_data[index].unsqueeze(0))

                    for index, car in enumerate(bs_index):
                        cnt = 0
                        for t in range(bs_box_len):
                            if car[t] == 1:
                                tmp[t].append(bs_data[index][cnt].unsqueeze(0))
                                tmp_num[t].append(bs_num[index][t].unsqueeze(0))
                                cnt += 1

                    # 如果所有的计算都符合逻辑的话，那么最终这个对齐应该很简单
                    for t in tmp:
                        if self.compute_var:
                            # 为了统计方差,所以这里已经统计进去了只有多视角信息的代码
                            if len(t) > 1:
                                # print([m.shape for m in t])
                                # 因为统计信息不需要 mv -> fused
                                box_loss_data_mv.append(torch.cat(t[1:], dim=0))
                        else:
                            if len(t) > 0:
                                box_loss_data_mv.append(torch.cat(t, dim=0))

                    for t in tmp_num:
                        if len(t) > 0:
                            point_num_mv.append(torch.cat(t, dim=0))

                box_mse_data = []
                box_mse_data_add = []
                for i, data in enumerate(box_loss_data_mv):

                    if self.compute_var and data.shape[0] > 1:
                        var_compute = data.squeeze()
                        # var_feature = torch.var(var_compute, dim=0, unbiased=False)
                        self.feature_var_list.extend(var_compute.detach().cpu().numpy().tolist())

                    t = data[1:]
                    repeat_num = t.shape[0]

                    # print(t.shape)
                    dim_repeat1 = [1] * len(t.squeeze().shape)
                    dim_repeat2 = [1] * len(t.squeeze().shape)
                    dim_repeat1.insert(0, repeat_num)
                    dim_repeat2.insert(1, repeat_num)

                    if repeat_num > 1:
                        # nolimit
                        entroy_box_datas1 = t.squeeze().unsqueeze(0).repeat(*dim_repeat1)
                        entroy_box_datas2 = t.squeeze().unsqueeze(1).repeat(*dim_repeat2)
                        # print(entroy_box_datas1.shape,entroy_box_datas2.shape)

                        if self.limit_mv_to_fused:
                            # 由于靠齐的局限性，所以需要设置较小的权重
                            # self.s_s_weight = 0.1
                            # 融合后的信息需要detach
                            entroy_box_datas1 = data[0].squeeze().repeat(repeat_num).detach().view(t.shape)
                            entroy_box_datas2 = t

                        if self.mv_loss_compute == "l1":
                            # L1
                            box_mse_data.append(torch.norm(entroy_box_datas1 - entroy_box_datas2, p=1).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))
                        elif self.mv_loss_compute == "kl":
                            # kl散度
                            # print(kl_divergence(entroy_box_datas1 , entroy_box_datas2),entroy_box_datas1.shape)
                            box_mse_data.append(kl_divergence(entroy_box_datas1, entroy_box_datas2).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))
                        elif self.mv_loss_compute == "js":
                            # js散度
                            box_mse_data.append(js_divergence(entroy_box_datas1, entroy_box_datas2).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))
                        else:
                            # 默认L1
                            box_mse_data.append(torch.norm(entroy_box_datas1 - entroy_box_datas2, p=1).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))

                        if self.add_s_f:
                            # print(data[0])
                            entroy_box_datas3 = data[0].squeeze().repeat(repeat_num).detach().view(t.shape)
                            # print(entroy_box_datas3)
                            entroy_box_datas4 = t
                            # print("reg shape:",data.shape,t.shape,entroy_box_datas4.shape,entroy_box_datas3.shape)
                            if self.mv_loss_compute == "l1":
                                # L1
                                box_mse_data_add.append(
                                    torch.norm(entroy_box_datas3 - entroy_box_datas4, p=1).unsqueeze(0) / (
                                            repeat_num * (repeat_num - 1)))
                            elif self.mv_loss_compute == "kl":
                                # kl散度
                                # print(kl_divergence(entroy_box_datas3 , entroy_box_datas4),entroy_box_datas3.shape)
                                box_mse_data_add.append(
                                    kl_divergence(entroy_box_datas3, entroy_box_datas4).unsqueeze(0) / (
                                            repeat_num * (repeat_num - 1)))
                            elif self.mv_loss_compute == "js":
                                # js散度
                                box_mse_data_add.append(
                                    js_divergence(entroy_box_datas3, entroy_box_datas4).unsqueeze(0) / (
                                            repeat_num * (repeat_num - 1)))
                            else:
                                # 默认L1
                                box_mse_data_add.append(
                                    torch.norm(entroy_box_datas3 - entroy_box_datas4, p=1).unsqueeze(0) / (
                                            repeat_num * (repeat_num - 1)))


                if len(box_mse_data) > 0:
                    mv_box_loss = torch.cat(box_mse_data, dim=0).sum() * self.s_s_weight / len(bs_len)

                if self.add_s_f and len(box_mse_data_add) > 0:
                    # if self.mv_loss_object == "reg":
                    mv_box_loss = mv_box_loss + 0.01 * torch.cat(box_mse_data_add,
                                                                 dim=0).sum() * self.s_s_weight / len(bs_len)

        batch_dict['mv_box_loss'] = mv_box_loss
        batch_dict['cls_box_loss'] = cls_box_loss

        batch_dict['loss_cls_gt_mv'] = loss_cls_gt_mv
        batch_dict['loss_cls_gt_fused'] = loss_cls_gt_fused
        print("算出来的loss", mv_box_loss, cls_box_loss, loss_cls_gt_mv, loss_cls_gt_fused)

        import json
        if self.compute_var and self.write_cnt % 100 == 0:
            with open(self.feature_var_path, 'w', encoding='utf-8') as f:
                json.dump(self.feature_var_list, f, ensure_ascii=False, indent=4)
        self.write_cnt += 1


        return batch_dict


# 测试，下述代码无bug
import torch
import torch.nn.functional as F


# 定义 KL 散度函数
def kl_divergence(logits1, logits2):
    # 将1维的logits通过sigmoid转换为概率
    prob1 = logits1.view(-1, 1)
    prob2 = logits2.view(-1, 1)

    # 将每个目标框的概率值组成二分类的分布 (p, 1-p)
    prob1 = torch.cat([prob1, 1 - prob1], dim=-1)
    prob2 = torch.cat([prob2, 1 - prob2], dim=-1)


    # 使用KL散度来比较两个分布
    kl_loss = F.kl_div(prob1.log(), prob2, reduction='sum')
    return kl_loss


def js_divergence(x, y):
    # 确保x和y的维度一样
    assert x.size() == y.size()
    # 计算softmax，将x转换为概率分布，现在已经是概率分布了
    # p = torch.nn.functional.softmax(x, dim=1)
    # q = torch.nn.functional.softmax(y, dim=1)
    prob1 = x.view(-1, 1)
    prob2 = y.view(-1, 1)
    # 将每个目标框的概率值组成二分类的分布 (p, 1-p)
    p = torch.cat([prob1, 1 - prob1], dim=-1)
    q = torch.cat([prob2, 1 - prob2], dim=-1)
    p = F.softmax(p, dim=-1)
    q = F.softmax(q, dim=-1)
    KL = nn.KLDivLoss(reduction='sum')
    m = ((q + p) / 2).log()
    return (KL(m, p) + KL(m, q)) / 2


def weighted_sigmoid_binary_cross_entropy(preds, tgts, weights=None,
                                          class_indices=None):
    if weights is not None:
        weights = weights.unsqueeze(-1)
    if class_indices is not None:
        weights *= (
            indices_to_dense_vector(class_indices, preds.shape[2])
                .view(1, 1, -1)
                .type_as(preds)
        )
    per_entry_cross_ent = nn.functional.binary_cross_entropy_with_logits(preds,
                                                                         tgts,
                                                                         weights)
    return per_entry_cross_ent


# 示例: logits1 和 logits2 是模型输出的1维类别分数
logits1 = torch.tensor([0.8])  # 例如第一个视角的输出
logits2 = torch.tensor([0.9])  # 第二个视角的输出

# 计算 KL 散度
kl_loss = kl_divergence(logits1, logits2)
js_loss = js_divergence(logits1, logits2)
# print("KL散度:", kl_loss.item())
# print("JS散度:", js_loss.item())
