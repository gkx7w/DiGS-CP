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

from opencood.utils.transformation_utils import x1_to_x2, x_to_world, get_pairwise_transformation, get_pairwise_transformation_from_poselist



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
        
        # stage1_align 配置文件
        if "stage1_align" in model_cfg:
            # stage1 结果需要保存并使用
            # self.stage1_result = read_json(self.stage1_result_path)
            self.stage1_align_args = model_cfg['stage1_align']['args']
        
        
        # 解耦部分神经网络
        self.factor_num = 8
        self.factor_dim = 32
        self.factor_encoder = nn.modules.ModuleList()
        fc_layers = [128,64]
        print(fc_layers)
        for i in range(self.factor_num):
            self.factor_encoder.append(self._make_fc_layers(pre_channel, fc_layers,
                                                  output_channels=
                                                   self.factor_dim)[0])

        self._init_weights(weight_init='xavier')

        # 统计数据
        self.cnt_refine = []
        self.all_totalkb = []
        self.all_totalmb = []

        # 停止记录
        self.stop_write = False


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


    def first_stage_roi_grid_pool(self, batch_dict, rois):
        batch_size = len(rois)

        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        label_record_len = [t.shape[0] for t in rois]
        
        mv_len = batch_dict['record_len']
        # print("拿gt去产生多视角gt的输出：")
        # print([t.shape for t in rois])

        rois = torch.cat(rois, dim=0)
        batch_size = len(label_record_len)

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

    def get_mv_boxes_feature(self,batch_dict):
        
        if 'det_boxes' in batch_dict:
            dets_list = batch_dict['det_boxes']
            boxes = []
            for i, dets in enumerate(dets_list):
                # 这个是新增的部分
                dets = dets[:, [0, 1, 2, 5, 4, 3, 6]]  # hwl -> lwh
                # if len(dets)==0:
                #     continue
                boxes.append(dets)

            # print("查看gt和第一阶段框区别：")
            # print(batch_dict['gt_boxes'].shape)
            # # print([ t.shape for t in batch_dict['gt_boxes']])
            # print([ t.shape for t in boxes])

        pooled_features_mv = self.first_stage_roi_grid_pool(batch_dict, boxes)
        # pooled_features_mv = self.first_stage_roi_grid_pool(batch_dict, batch_dict['gt_boxes'])

        batch_size_mv = pooled_features_mv.shape[0]
        pooled_features_mv = pooled_features_mv.permute(0, 2, 1). \
            contiguous().view(batch_size_mv, -1, self.grid_size,
                                self.grid_size,
                                self.grid_size)  # (BxN, C, 6, 6, 6)
        # 共享特征提取
        shared_features_mv = self.shared_fc_layers(
            pooled_features_mv.view(batch_size_mv, -1, 1))

        batch_dict["mv_boxes_feature"] = shared_features_mv
        return batch_dict

    def decoupling(self, batch_dict):
        batch_dict = self.get_mv_boxes_feature(batch_dict)
        shared_features_mv = batch_dict['mv_boxes_feature']

        # 下一步才是解耦，但是这里有一个问题就是需要找到公共物体，
        # 公共物体的寻找是基于投影+iou找到的，理论上要经过一次matcher_new，
        # 不然这个融合框问题的 ，这个特征能拼到box+分数，送入融合
        boxes_factors = []
        for i in range(self.factor_num):
            boxes_factors.append(self.factor_encoder[i](shared_features_mv).squeeze())
        print([t.shape for t in boxes_factors])
        # all box , 32 *8
        boxes_factors = torch.cat(boxes_factors,dim=1)
        # 需要用iou来寻找公共物体再融合，当前box没有投影，点也没有投影，先投影再运算
        # 这些代码，已经存在在matcher 的结果上了，问题就在于找到最终索引就能合并框了
        cluster_indices,cur_cluster_id =  batch_dict['cluster_indices'], batch_dict['cur_cluster_id']
        # 寻找公共物体
        object_factors_batch = []
        # 可以直接融合，取平均值，因子内融合
        record_len = [int(l) for l in batch_dict['record_len']]
        for i, l in enumerate(record_len):
            object_factors = [] 
            for j in range(1, cur_cluster_id[i]):
                object_factors.append(boxes_factors[cluster_indices[i]==j].mean(dim=0).unsqueeze(dim=0))
            object_factors = torch.cat(object_factors,dim=0)
        object_factors_batch.append(object_factors)
        # print([t.shape for t in object_factors])
        
        # 这是物体级的feature
        # 区分，区分每个场景下，特征属于哪个场景
        # object_factors = torch.cat(object_factors,dim=0)
        # print("解耦后并融合的物体级因子： 物体数量 ， 8 * 32 ：",[t.shape for t in object_factors_batch])
        batch_dict["fused_object_factors"] = object_factors_batch

        # 直接用 boxes_fused  22*7 
        
        return batch_dict

    def getTopK(self,batch_dict,k):
        # k是选择手段 
        max_shape = max([t.shape[0] for t in batch_dict['det_scores']])
        box_shape =  batch_dict['det_boxes'][0].shape[-1]
        # 因为历史原因，这一项最后一位始终=1，倒数第二才是真正维度
        feature_shape = batch_dict["mv_boxes_feature"].shape[-2]
        device = batch_dict['det_scores'][0].device
        batch_size = len(batch_dict['det_scores'])

        stage1_scores = torch.zeros([batch_size,max_shape]).to(device)
        stage1_boxes = torch.zeros([batch_size,max_shape,box_shape]).to(device)
        stage1_features = torch.zeros([batch_size,max_shape,feature_shape]).to(device)
        start = 0
        for i,t in enumerate(batch_dict['det_scores']):
            num = t.shape[0]
            if num==0:
                continue
            tmp = batch_dict["mv_boxes_feature"][start:start+num].squeeze(-1)
            
            if len(tmp.shape) == 1:
                tmp = tmp.unsqueeze(0)
            
            stage1_features[i,:num] = tmp
            
            stage1_scores[i,:num] = t
            stage1_boxes[i,:num] = batch_dict['det_boxes'][i]
            start+=num
        # print(stage1_scores.shape,stage1_boxes.shape,stage1_features.shape)        
        s,b,f = self.topk(stage1_scores,stage1_boxes,stage1_features,k)
        # print(s.shape,b.shape,f.shape)
    
        # 转化为list输出，同时考虑到每个场景下的真实车辆，每个list元素是一个tensor
        now_shape = [min([t.shape[0],k]) for t in batch_dict['det_scores']]
        res_s,res_b,res_f = [],[],[]
        for i in range(batch_size):
            res_s.append(s[i][:now_shape[i]])
            res_b.append(b[i][:now_shape[i]])
            res_f.append(f[i][:now_shape[i]])
        return res_s,res_b,res_f

    def compute_size(self,tensor_example):
        # 创建一个示例张量，例如形状为(2, 3, 256, 256)的浮点数张量
        # 找到非零元素的位置
        non_zero_elements = tensor_example.nonzero(as_tuple=False)

        # 获取每个元素的字节数，例如float32类型的元素占4个字节
        bytes_per_element = tensor_example.element_size()

        # 计算非零元素的数量，并乘以每个元素的字节数得到总字节数
        total_bytes = non_zero_elements.size(0) * bytes_per_element

        # 转换为KB或MB
        total_kb = total_bytes / 1024
        total_mb = total_kb / 1024
        # print(f"非零张量元素的大小为: {total_kb} KB")
        # print(f"非零张量元素的大小为: {total_mb:.5f} MB")        
        return total_kb,total_mb

    def topk(self,scores,boxes,features,k):
        # scores bs n
        # box bs n m
        # feature bs n h
        res_scores,ind = torch.sort(scores,dim=1,descending=True)
        res_scores = res_scores[:,:k]
        ind = ind[:,:k]
        ind1 = ind.unsqueeze(-1).expand(-1, -1, features.size(2))
        ind2 = ind.unsqueeze(-1).expand(-1, -1, boxes.size(2))
        res_features = torch.gather(features, 1, ind1)
        res_boxes = torch.gather(boxes, 1, ind2)
        return res_scores,res_boxes,res_features

    def selectp(self,features,p):
        # feature [tensor tensor ...]
        # 这个shape下的特征=0
        now_shape = [int(t.shape[0]*p) for t in features]
        for i,t in enumerate(now_shape):
            # 多少不含特征
            features[i][:t,:] = 0
        return features

    def selects(self,features,scores,s):
        # feature [tensor tensor ...]
        # 这个shape下的特征=0
        for i in range(len(scores)):
            # 多少不含特征
            ind = scores[i]>s
            features[i][ind] = 0
        return features

    def stage1_align(self, batch_dict,selectk,selectp,selects,abandon_hard_cases=True):
        self.stage1_align_args["abandon_hard_cases"] = abandon_hard_cases
        from opencood.models.sub_modules.stage1_align import stage1_alignment_relative_sample_np
        
        # 这里获取的就是规则的 数据和特征,取得最大的k个数，
        s,b,f = self.getTopK(batch_dict,selectk)
        # 按照比例筛选
        # f = self.selectp(f,selectp)
        # 按照分数筛选 
        f = self.selects(f,s,selects)
        
        if not self.stop_write:
            # 顺便统计一下通讯量大小,通信量跑一次就可以了
            total_kb,total_mb = 0,0
            for si,bi,fi in zip(s,b,f):
                si_kb,si_mb = self.compute_size(si)
                bi_kb,bi_mb = self.compute_size(bi)
                fi_kb,fi_mb = self.compute_size(fi)
                total_kb = total_kb + si_kb +bi_kb +fi_kb
            
            # print("总通信量：",total_kb)
            # self.all_totalkb.append(total_kb)

            # # 知道了这个场景下的通信量，存起来吧
            # if len(self.all_totalkb)>20:
            #     # 20次存一下，降低IO时间
            #     append_list_to_file("/home/ypy/projects/Code/tmp_data/comm/s/05_select.txt",self.all_totalkb)
            #     self.all_totalkb = []
        # 此处的通讯量是否 = 中后期融合通信量，不太等于，

        # lwh，先转化为 n 8 3
        # stage1_boxes = [box_utils.boxes_to_corners_3d(t,"lwh") for t in boxes]
        # hwl
        stage1_boxes = [box_utils.boxes_to_corners_3d(t,"hwl") for t in b]
        # 将分数转化为不确定性
        stage1_scores = [t.unsqueeze(1).repeat(1,3) for t in s]

        # 特征后续待处理
        stage1_feature = []
        for t in f:
            if len(t.shape) == 1:
                stage1_feature.append(np.array(t.unsqueeze(0).cpu()))
            else:
                stage1_feature.append(np.array(t.cpu()))

        cur_agnet_pose_clean = np.array(batch_dict['lidar_pose_clean'].cpu())

        # 只有大于一辆车才能匹配
        cur_agnet_pose = np.array(batch_dict['lidar_pose'].cpu())
        if stage1_boxes is not None and len(stage1_boxes)>1:
            # 这是单车无噪下的检测结果
            # 获取有噪声 的 pose
            diff1 = np.abs(cur_agnet_pose_clean-cur_agnet_pose)
            
            # 看到当前有噪声的车在过去无噪声环境下看到的框 
            pred_corners_list = [ np.array(t.cpu()) for t in stage1_boxes]
            uncertainty_list = [np.array(t.cpu()) for t in stage1_scores]
            pred_feature_list = stage1_feature

            # 修正位置  pred_corners_list（观察到的车（car n 8 3）），cur_agnet_pose（有噪声的定位 car 6） uncertainty_list（car n 3）
            if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                refined_pose = stage1_alignment_relative_sample_np(pred_corners_list,
                                                                cur_agnet_pose, 
                                                                certainty_tensor_list=batch_dict['det_scores'],
                                                                pred_feature_list=pred_feature_list,
                                                                uncertainty_list=uncertainty_list, 
                                                                **self.stage1_align_args)
                cur_agnet_pose[:,[0,1,4]] = refined_pose 
            
            diff2 = np.abs(cur_agnet_pose_clean - cur_agnet_pose)
            error,refine = np.sum(diff1),np.sum(diff2)
            # print("处理前误差： ",error)
            # print("处理后误差： ",refine)
            # print("修正概率： ",refine/(error+1e-8))
            # self.cnt_refine.append(refine/(error+1e-8))
            # 保存数据
            # if len(self.cnt_refine)>20:
            #     # append_list_to_file("/home/ypy/projects/Code/tmp_data/sdcoper_2refine.txt",self.cnt_refine)
            #     self.cnt_refine = []

        batch_dict['lidar_pose'] = torch.tensor(cur_agnet_pose).to(batch_dict['lidar_pose_clean'].device)
        # 每个车的坐标被修正，以此计算投影矩阵
        # 这个矩阵在使用的时候要在投影时候采用，回到matcher_new
        pairwise_t_matrix = \
            get_pairwise_transformation_from_poselist(cur_agnet_pose,
                                            batch_dict['max_cav'],
                                            batch_dict['proj_first'])
        return batch_dict,pairwise_t_matrix

    def stage1_refine(self, batch_dict,abandon_hard_cases=True):
        self.stage1_align_args["abandon_hard_cases"] = abandon_hard_cases
        
        from opencood.models.sub_modules.stage1_align import stage1_alignment_relative_sample_np
        # 获取了第一阶段box，分数，特征
        # lwh，先转化为 n 8 3
        # stage1_boxes = [box_utils.boxes_to_corners_3d(t,"lwh") for t in boxes]
        # hwl
        stage1_boxes = [box_utils.boxes_to_corners_3d(t,"hwl") for t in batch_dict['det_boxes_refine']]
        # 将分数转化为不确定性
        stage1_scores = [t.unsqueeze(1).repeat(1,3) for t in batch_dict['det_scores_refine']]

        # 特征后续待处理
        start = 0
        stage1_feature = []

        cur_agnet_pose_clean = np.array(batch_dict['lidar_pose_clean'].cpu())

        # 只有大于一辆车才能匹配
        cur_agnet_pose = np.array(batch_dict['lidar_pose'].cpu())
        if stage1_boxes is not None and len(stage1_boxes)>1:
            # 这是单车无噪下的检测结果
            # 获取有噪声 的 pose
            diff1 = np.abs(cur_agnet_pose_clean-cur_agnet_pose)
            # 看到当前有噪声的车在过去无噪声环境下看到的框 
            pred_corners_list = [ np.array(t.cpu()) for t in stage1_boxes]
            uncertainty_list = [np.array(t.cpu()) for t in stage1_scores]

            # 修正位置  pred_corners_list（观察到的车（car n 8 3）），cur_agnet_pose（有噪声的定位 car 6） uncertainty_list（car n 3）
            if sum([len(pred_corners) for pred_corners in pred_corners_list]) != 0:
                refined_pose = stage1_alignment_relative_sample_np(pred_corners_list,
                                                                cur_agnet_pose, 
                                                                certainty_tensor_list=batch_dict['det_scores_refine'],
                                                                pred_feature_list=None,
                                                                uncertainty_list=uncertainty_list, 
                                                                **self.stage1_align_args)
                cur_agnet_pose[:,[0,1,4]] = refined_pose 
            
            diff2 = np.abs(cur_agnet_pose_clean - cur_agnet_pose)
            error,refine = np.sum(diff1),np.sum(diff2)
            # print("处理前误差： ",error)
            # print("处理后误差： ",refine)
            # print("修正概率： ",refine/(error+1e-8))
            # self.cnt_refine.append(refine/(error+1e-8))
            # 保存数据
            # if len(self.cnt_refine)>20:
            #     # append_list_to_file("/home/ypy/projects/Code/tmp_data/sdcoper_2refine.txt",self.cnt_refine)
            #     self.cnt_refine = []

        batch_dict['lidar_pose'] = torch.tensor(cur_agnet_pose).to(batch_dict['lidar_pose_clean'].device)
        # 每个车的坐标被修正，以此计算投影矩阵
        # 这个矩阵在使用的时候要在投影时候采用，回到matcher_new
        pairwise_t_matrix = \
            get_pairwise_transformation_from_poselist(cur_agnet_pose,
                                            batch_dict['max_cav'],
                                            batch_dict['proj_first'])
        return batch_dict,pairwise_t_matrix

    def forward(self, batch_dict):
        # 训练部分代码

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
            pooled_features.view(batch_size_rcnn, -1, 1))
        #  shared_features 融合后
        #  a 11 b 9  fuse 13 
        #  13   12 25个
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


def append_list_to_file(filepath, data_list):
    with open(filepath, 'a', encoding='utf-8') as f:  # 使用'a'模式进行追加操作
        for item in data_list:
            # 将每个元素转换为字符串并写入新行
            f.write(f"{item}\n")

# 示例: logits1 和 logits2 是模型输出的1维类别分数
logits1 = torch.tensor([0.8])  # 例如第一个视角的输出
logits2 = torch.tensor([0.9])  # 第二个视角的输出

# 计算 KL 散度
kl_loss = kl_divergence(logits1, logits2)
js_loss = js_divergence(logits1, logits2)
# print("KL散度:", kl_loss.item())
# print("JS散度:", js_loss.item())
