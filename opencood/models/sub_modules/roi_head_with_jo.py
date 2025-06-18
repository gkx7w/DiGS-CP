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
        # self.fake_shared_fc_layers, pre_channel1 = self._make_fc_layers(pre_channel1,
        #                                                                 fc_layers)
        # self.fake_cls_head, pre_channel1 = self._make_fc_layers(pre_channel1,
        #                                                         fc_layers,
        #                                                         output_channels=
        #                                                         self.model_cfg[
        #                                                             'num_cls'])
        # self.fake_reg_layers, _ = self._make_fc_layers(pre_channel1, fc_layers,
        #                                                output_channels=
        #                                                self.model_cfg[
        #                                                    'num_cls'] * 7)

        pre_channel = grid_size * grid_size * grid_size * c_out

        self.shared_fc_layers, pre_channel = self._make_fc_layers(pre_channel,
                                                                  fc_layers)

        # 新增 mv->fv converter
        # self.convertor = None
        self.convertor, pre_channel = self._make_fc_layers(pre_channel, fc_layers)
        
        # self.cls_layers, pre_channel = self._make_fc_layers(pre_channel,
        #                                                     fc_layers,
        #                                                     output_channels=
        #                                                     self.model_cfg[
        #                                                         'num_cls'])
        # self.iou_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
        #                                           output_channels=
        #                                           self.model_cfg['num_cls'])
        # self.reg_layers, _ = self._make_fc_layers(pre_channel, fc_layers,
        #                                           output_channels=
        #                                           self.model_cfg[
        #                                               'num_cls'] * 7)

        
        
        
        # 解耦代码 
        # self.decoupling = True
        # self.factor_num = 8
        # self.factor_dim = 32
        # self.factor_encoder = nn.modules.ModuleList()
        # fc_layers = [128,64]
        # for i in range(self.factor_num):
        #     self.factor_encoder.append(self._make_fc_layers(pre_channel, fc_layers,
        #                                           output_channels=
        #                                            self.factor_dim)[0])

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
        # nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

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


    def first_stage_roi_grid_pool(self, batch_dict, rois, empty_roi_mask):
        # roi list的长度
        batch_size = len(rois)

        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        # 每个roi的中车辆的数量
        label_record_len = [t.shape[0] for t in rois]
        
        mv_len = batch_dict['record_len']
        rois = torch.cat(rois, dim=0) # [所有车个数(BxN)，7]
        batch_size = len(label_record_len) #roi list的长度

        point_features = torch.cat(point_features, dim=0)
        # (BxN, 6x6x6, 3)
        global_roi_grid_points, local_roi_grid_points = \
            self.get_global_grid_points_of_roi(rois)

        xyz = torch.cat(point_coords, dim=0)
        kpt_mask_flag = batch_dict['kpt_mask_flag']
        if kpt_mask_flag:
            xyz_batch_cnt = xyz.new_zeros(batch_size + 1).int()
        else:
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()

        #point_coords不是batch维度的，要把空的box对应的point_coords也mask
        xyz_batch_id = 0
        
        for bs_idx in range(batch_dict['batch_size']): # batch_dict['record_len'] 多batch要改成batch_size，但还有报错3164/6765
            if empty_roi_mask[bs_idx] or kpt_mask_flag:
                xyz_batch_cnt[xyz_batch_id] = len(point_coords[bs_idx])
                kpt_mask_flag = False
                xyz_batch_id += 1
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

        pooled_features = pooled_features.view(-1, self.grid_size ** 3,
                                               pooled_features.shape[-1]) # (BxN, 6x6x6, C)

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


    # 这里包含了给多视角box分配GT
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
            rcnn_labels = (max_ious > self.score_threshold).float()
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

        # 融合后的框 pool 融合后的点 
        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, self.grid_size,
                            self.grid_size,
                            self.grid_size)  # (BxN, C, 6, 6, 6)
        shared_features = self.shared_fc_layers(
            pooled_features.view(batch_size_rcnn, -1, 1)).squeeze(-1) # (BxN, C)
        
        object_factors_batch = []
        box_idx = 0
        for i in range(len(batch_dict['boxes_fused'])):
            object_factors_batch.append(shared_features[box_idx:box_idx + len(batch_dict['boxes_fused'][i])])
            box_idx += len(batch_dict['boxes_fused'][i])
        
        
        
        # rcnn_cls = self.cls_layers(shared_features).transpose(1,
        #                                                       2).contiguous().squeeze(
        #     dim=1)  # (B, 1 or 2)
        # rcnn_iou = self.iou_layers(shared_features).transpose(1,
        #                                                       2).contiguous().squeeze(
        #     dim=1)  # (B, 1)
        # rcnn_reg = self.reg_layers(shared_features).transpose(1,
        #                                                       2).contiguous().squeeze(
        #     dim=1)  # (B, C)
        
        # batch_dict['stage2_out'] = {
        #     'rcnn_cls': rcnn_cls,
        #     'rcnn_iou': rcnn_iou,
        #     'rcnn_reg': rcnn_reg,
        # }

        
        # if 'det_boxes' in batch_dict:
        #     dets_list = batch_dict['det_boxes']

        #     boxes = []
        #     empty_box_mask = []
        #     for i, dets in enumerate(dets_list):
        #         # 这个是新增的部分
        #         dets = dets[:, [0, 1, 2, 5, 4, 3, 6]]  # hwl -> lwh
        #         if len(dets)==0: #这里直接continue的话，会导致first_stage_roi_grid_pool中batch_size的大小不对！！无法记录到对应point_coords的数据！！
        #             empty_box_mask.append(False)
        #             continue
        #         empty_box_mask.append(True)
        #         boxes.append(dets)

        # pooled_features_mv = self.first_stage_roi_grid_pool(batch_dict, boxes, empty_box_mask)

        # batch_size_mv = pooled_features_mv.shape[0] # BxN
        # pooled_features_mv = pooled_features_mv.permute(0, 2, 1). \
        #     contiguous().view(batch_size_mv, -1, self.grid_size,
        #                         self.grid_size,
        #                         self.grid_size)  # (BxN, C, 6, 6, 6)
        # # 共享特征提取
        # shared_features_mv = self.shared_fc_layers(
        #     pooled_features_mv.view(batch_size_mv, -1, 1)).squeeze() # (BxN, 512)

        # # 解耦
        # # boxes_factors = []
        # # for i in range(self.factor_num):
        # #     boxes_factors.append(self.factor_encoder[i](shared_features_mv).squeeze())

        # # processed_boxes_factors = [factor.unsqueeze(0) if factor.dim() == 1 else factor for factor in boxes_factors]
        # # # all box , 32 *8
        # # boxes_factors = torch.cat(processed_boxes_factors,dim=1)
        
        # # 寻找公共物体
        # cluster_indices,cur_cluster_id = batch_dict['cluster_indices'], batch_dict['cur_cluster_id']
        # object_factors_batch = []
        # # 可以直接融合，取平均值，因子内融合
        # record_len = [int(l) for l in batch_dict['record_len']]
        # # 获取实际batch中box的总数量 [b1中box数量, b2中box数量, ...]
        # batch_box_len = []
        # current_idx = 0
        # for l in record_len:
        #     total_boxes = 0
        #     for j in range(l):
        #         total_boxes += len(dets_list[current_idx + j])
        #     batch_box_len.append(total_boxes)
        #     current_idx += l
        # # 首先计算每个batch的起始索引
        # start_indices = [0]
        # for i in range(len(batch_box_len)-1):
        #     start_indices.append(start_indices[-1] + batch_box_len[i])
        
        # for i, l in enumerate(record_len):
        #     object_factors = []
        #     start_idx = start_indices[i]
        #     end_idx = start_idx + batch_box_len[i]
        #     # 获取当前批次的box特征
        #     current_boxes_factors = shared_features_mv[start_idx:end_idx] #shared_features_mv
        #     # 检查current_boxes_factors是否为空
        #     if len(current_boxes_factors) == 0:
        #         default_factor = torch.zeros(1, 256, device=shared_features_mv.device) #shared_features_mv
        #         object_factors_batch.append(default_factor)
        #         continue
            
        #     for j in range(1, cur_cluster_id[i]):
        #         mask = (cluster_indices[i] == j)
        #         if cluster_indices[i].shape[0] < current_boxes_factors.shape[0]:
        #             # 将掩码扩展，保留原始True位置，其余设为False
        #             mask = torch.zeros(len(current_boxes_factors), dtype=torch.bool, device=cluster_indices[i].device)
        #             mask[:len(cluster_indices[i])] = (cluster_indices[i] == j)
        #         object_factors.append(current_boxes_factors[mask].mean(dim=0).unsqueeze(dim=0)) #.mean(dim=0)
            
        #     all_features = torch.cat(object_factors, dim=0) #.unsqueeze(0)  # [1, sum(N_i), D]
        #     attn_output, _ = self.self_attention(all_features, all_features, all_features)
        #     enhanced_features = self.feature_enhance(attn_output.squeeze(0))
        #     object_factors_batch.append(enhanced_features)
        #     # object_factors_batch.append(torch.cat(object_factors,dim=0)) #这里融合没有可优化的参数，故diffusion无法监督融合，监督的还是前面的解耦
            
        #     # if len(object_factors) > 0:  # 如果有非背景聚类
        #     #     object_factors = torch.cat(object_factors,dim=0)
        #     # else:
        #     #     print("没有非背景聚类???预测的box被过滤了")
        #     #     object_factors = None
            
        #     # if object_factors is not None:
        #     #     object_factors_batch.append(object_factors)

        batch_dict["fused_object_factors"] = object_factors_batch # 物体数量, 32 *8
        
        return batch_dict
    