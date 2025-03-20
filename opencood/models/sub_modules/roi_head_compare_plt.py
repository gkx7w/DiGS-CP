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
    plt.scatter(x, y, s=1, c='gray', alpha=0.7)
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


def plot_grid_with_points_all_256(x, y, boxes1, boxes2, iou, label,color, title):
    fig, axes = plt.subplots(1, 1, dpi=1200)
    axes.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    # plt.figure(dpi=100)
    color_list = ["#f23752","#47b8e0","#FF9000","#dcceca"]
    # #f0de28
    plt.scatter(x, y, s=1, c=color_list[color], alpha=1, edgecolors=color_list[color], marker=".")
    axes.set_xlim(-150, 150)  # 这里假设你的数据范围是 0 到 1
    axes.set_ylim(-150, 150)  # 同样假设 y 轴的数据范围也是 0 到 1
    # 绘制点云
    # for index, box in enumerate(boxes1):
    #
    #     # if iou[index]<0.7:
    #
    #     x1, y1, width, height = box[0], box[1], box[3], box[4]
    #
    #     if label[index] >= 1.0:
    #
    #         rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="red",
    #                                  facecolor='none')
    #     else:
    #         rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="yellow",
    #                                  facecolor='none')
    #
    #     plt.text(x1, y1, str(iou[index])[:5], ha='center', va='center', fontsize=3)
    #
    #     # plt.text(x1+width, y1+height, str(label[index])[:5], ha='center', va='center')
    #     axes.add_patch(rect)
    #
    # for index, box in enumerate(boxes2):
    #     x1, y1, width, height = box[0], box[1], box[3], box[4]
    #     rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="green",
    #                              facecolor='none')
    #     axes.add_patch(rect)
    axes.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    plt.title("two box  256   " + title)
    plt.savefig('/home/ypy/projects/CoAlign/save_png/'+title+".png", transparent=True)
    plt.show()

def plot_grid_with_points_all_256_mv(points, boxes1, boxes2, iou, label, cars,title):
    fig, axes = plt.subplots(1, 1, dpi=1200)
    axes.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    # plt.figure(dpi=100)
    color_list = ["#f23752","#47b8e0","#FF9000","#dcceca"]


    for now_show in range(cars):
        show_data = points[:, 0].long() == now_show
        plt.scatter(
            points[show_data][:, 1].detach().cpu().numpy(),
            points[show_data][:, 2].detach().cpu().numpy(),
            s=1, c=color_list[now_show], alpha=1,edgecolors=color_list[now_show], marker=".")

    axes.set_xlim(-150, 150)  # 这里假设你的数据范围是 0 到 1
    axes.set_ylim(-150, 150)  # 同样假设 y 轴的数据范围也是 0 到 1
    axes.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    plt.title("two box  256   " + title)
    plt.savefig('/home/ypy/projects/CoAlign/save_png/'+title+".png", transparent=True)
    plt.show()


def plot_grid_with_points_all_256_mv_with_boxes(points, boxes1, boxes2, iou, label, cars,title):
    fig, axes = plt.subplots(1, 1, dpi=1200)
    axes.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    # plt.figure(dpi=100)
    color_list = ["#f23752","#47b8e0","#FF9000","#dcceca"]

    for now_show in range(cars):
        show_data = points[:, 0].long() == now_show
        plt.scatter(
            points[show_data][:, 1].detach().cpu().numpy(),
            points[show_data][:, 2].detach().cpu().numpy(),
            s=1, c=color_list[now_show], alpha=1,edgecolors=color_list[now_show], marker=".")

    axes.set_xlim(-150, 150)  # 这里假设你的数据范围是 0 到 1
    axes.set_ylim(-150, 150)  # 同样假设 y 轴的数据范围也是 0 到 1

    for index, box in enumerate(boxes2):
        x1, y1, width, height = box[0], box[1], box[3], box[4]
        rect = patches.Rectangle((x1 - width / 2, y1 - height / 2), width, height, linewidth=1, edgecolor="green",
                                 facecolor='none')
        axes.add_patch(rect)

    axes.patch.set_alpha(0.0)
    fig.patch.set_alpha(0.0)
    plt.title(" gt box   " + title)
    plt.savefig('/home/ypy/projects/CoAlign/save_png/'+title+".png", transparent=True)
    plt.show()



import torch
import numpy as np
pi = 3.141592653
def random_translation(box, max_translation=0.5):
    """ 对box进行随机平移 """
    translation = (torch.rand((box.shape[0],3)).to(box.device) * 2 - 1) * max_translation  # -max_translation 到 +max_translation
    box[:,:3] += translation
    return box

def random_scaling(box, min_scale=0.8, max_scale=1.2):
    """ 对box进行随机缩放 """
    scale_factor = torch.empty((box.shape[0],3)).uniform_(min_scale, max_scale).to(box.device)
    box[:,3:6] *= scale_factor  # 只缩放尺寸
    return box

def random_rotation(box, max_angle=45.0):
    """ 对box进行随机旋转 """
    angle = (torch.rand((box.shape[0])).to(box.device) * 2 - 1) * max_angle  # -max_angle 到 +max_angle 度
    angle_rad = angle*(pi/180)
    box[:,6] += angle_rad  # 更新旋转角度
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

        # 统计绘图
        self.save_png_cnt = 1


        # 调整分类头
        self.big_head = True

        # 计算loss方式，当前是 l1 计划加入 kl 散度
        self.mv_loss_compute = "l1"
        self.mv_loss_sigmod = False

        # 对mv多视角数据，要不要用convertor
        self.use_convertor = False

        # 确定是否需要添加额外监督信号,这个是全部正样本
        self.add_more_pos_supervision = True

        # 这个是添加增强的gt信息
        self.add_more_augmented_supervision = True

        # 对齐的时候采用融合前向融合后对齐
        self.limit_mv_to_fused = False

        self.mv_box_weight = 1
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
            fc_layers = [self.model_cfg['n_fc_neurons']*4,self.model_cfg['n_fc_neurons']*2]
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
        self.convertor, pre_channel = self._make_fc_layers(pre_channel,fc_layers)

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

            # 统计各个类别数量
            rcnn_labels_03 = (max_ious > 0.3).float().sum().item()
            rcnn_labels_05 = (max_ious > 0.5).float().sum().item()
            rcnn_labels_07 = (max_ious > 0.7).float().sum().item()
            rcnn_all = rcnn_labels.shape[0]
            rcnn_res = rcnn_all - rcnn_labels_03
            rcnn_03_07 = rcnn_labels_03 - rcnn_labels_07
            rcnn_05_07 = rcnn_labels_05 - rcnn_labels_07
            rcnn_03_05 = rcnn_labels_03 - rcnn_labels_05

            # print("数据统计  ,rcnn_all,  rcnn_labels_07,  rcnn_labels_05,  rcnn_labels_03, rcnn_res, rcnn_03_07,  rcnn_03_05, rcnn_05_07")
            #
            #
            # print("数据统计  ",rcnn_all,rcnn_labels_07,rcnn_labels_05,rcnn_labels_03,rcnn_res,rcnn_03_07,rcnn_03_05,rcnn_05_07)
            #
            # print("数据统计  ",rcnn_labels_07/rcnn_all,rcnn_labels_05/rcnn_all,rcnn_labels_03/rcnn_all,rcnn_res/rcnn_all,rcnn_03_07/rcnn_all,rcnn_03_05/rcnn_all,rcnn_05_07/rcnn_all)

            # print("数据统计  ", rcnn_03_07 / rcnn_all)

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

    # 这两个pool方法是永久有效的，后面即使产生了偏差也能算,
    # 这两个最好指定一下捏
    def gt_roi_grid_pool(self, batch_dict, rois):
        batch_size = len(batch_dict['gt_boxes_len'])
        # 只用修改一行，让gt取代，但是这需要matcher不要改变数据结构才能实现
        # 这里的point信息不能融合！，不能合并，必须分开算，纯循环，应该好实现
        # 第二个问题，大问题，这个gt_of_rois对不上！，里面有些东西被修改了！
        # 需要拆分，拆分为n个车就有的n个box

        # 加不加这个
        # rois = batch_dict['gt_boxes']

        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']
        label_record_len = batch_dict['gt_boxes_len']
        Multi_view_box_index = batch_dict['Multi_view_box_index']
        mv_len = batch_dict['record_len']
        # 根据mv 修改rois,但是注意几个问题！
        # 1.后续还要靠mv对齐一个框在不同视角下内容
        # 2.要注意len
        # 这个cat问题是小问题，说明有个场景一个框都没有，这种需要bs = 2也能解决，或者直接跳过即可
        tmp_rois,start_index,start_rois,tmp_rois_len = [],0,0,[]
        for i,v in enumerate(mv_len):
            for j in range(v):
                tmp = []
                for k in range(label_record_len[i]):
                    if Multi_view_box_index[start_index+j][k]==1:
                        tmp.append(rois[start_rois+k].unsqueeze(0))
                if len(tmp)>0:
                    tmp_rois.append(torch.cat(tmp,dim=0))
                    tmp_rois_len.append(len(tmp))
                else:
                    print("为啥会是0？",Multi_view_box_index[start_index+j].sum())
                    tmp_rois_len.append(0)
            start_index += v
            start_rois += label_record_len[i]

        rois = torch.cat(tmp_rois,dim=0)
        # print("cat为啥报错",tmp_rois_len,mv_len,Multi_view_box_index.shape)
        label_record_len = tmp_rois_len
        batch_size = len(label_record_len)
        batch_dict['gt_box_record_len'] = label_record_len

        # 这里处理会报错，不用这个计算
        # print('point_num_deal1')
        # points_in_boxes = []
        # for points,boxes in zip(point_coords,tmp_rois):
        #     print("处理前:",points.shape,boxes.shape)
        #     # 用key point会报错！还是用普通的值吧，我们在最外面处理她！
        #     point_indices = points_in_boxes_cpu(points[:, :3], boxes[:,
        #                                                        [0, 1, 2, 5, 4,
        #                                                         3, 6]])
        #     cur_num = point_indices.sum(axis=1).unsqueeze(1)
        #     points_in_boxes.append(cur_num)
        #
        # print('point_num_deal2')
        #
        # points_in_boxes = torch.cat(points_in_boxes,dim=0)
        # batch_dict['points_in_boxes'] = points_in_boxes


        point_features = torch.cat(point_features, dim=0)
        # (BxN, 6x6x6, 3)
        global_roi_grid_points, local_roi_grid_points = \
            self.get_global_grid_points_of_roi(rois)
        # (B, Nx6x6x6, 3)
        # global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)


        xyz = torch.cat(point_coords, dim=0)
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()

        # a, b = [t.shape for t in point_coords], [t.shape for t in tmp_rois]
        # print("报错前信息2", len(a), len(b),a,b)
        # print("原版shape2:", global_roi_grid_points.shape, batch_size, rois.shape, label_record_len, xyz.shape)

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

    def gt_fused_roi_grid_pool(self, batch_dict,rois):
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

    # 要先生成框，然后再得到feature，其实可以写成一个函数
    def assign_targets_mv(self, batch_dict):
        batch_dict['rcnn_label_dict_mv'] = {
            # 目前只有一个任务，分类，让所有标签=1即可
        }

        # print("融合前后对应关系")
        # print(batch_dict['gt_boxes_len'],sum(batch_dict['gt_boxes_len']))
        # print(batch_dict['Multi_view_box_index'].shape)

        # 这里面很多信息要修改，添加奇奇妙妙框，这些框走的是融合点还是单视角点，应该走单视角，那就需要狠狠地区分了!,用尽可能简单的方式对gt扩列
        # 比如单纯增加两倍，顺序完全不变，然后在送入模型，这样应该是能保持信息不变的
        label_cnt = batch_dict['Multi_view_box_index'].sum().int().item()
        cls_tgt_mv = torch.ones((label_cnt,1)).to(batch_dict['Multi_view_box_index'].device)
        batch_dict['rcnn_label_dict_mv']["cls_tgt_mv"] = cls_tgt_mv

        # 这个家伙会排除没有点的gt_box，不过让没有点的box也学习到东西是我们的目标之一
        # label_cnt = (batch_dict['Multi_view_box_index'].sum(dim=0)>0).int().sum().item()
        cls_tgt_fv = torch.ones((sum(batch_dict['gt_boxes_len']), 1)).to(batch_dict['Multi_view_box_index'].device)
        batch_dict['rcnn_label_dict_mv']["cls_tgt_fv"] = cls_tgt_fv

        # print("新label：",cls_tgt_fv.shape,cls_tgt_mv.shape)
        return batch_dict

    # 这些确实能带来很多其他类型的样本，准确说已经是新的代码了，后面要针对多视角获取一下这些框
    def assign_targets_fv_more(self, batch_dict):
        # 这个过程可以多做几次，算求，为了简单，只做一次
        batch_dict['rcnn_label_dict_fv_more'] = {
            'rois': [],
            'gt_of_rois': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
        }
        # gt boxes 代表融合后，直接就能训练
        # 只针对这一个做就好了，后面mv的本身也就是基于他去操作
        gt_boxes_len = batch_dict['gt_boxes_len']
        gt_boxes = batch_dict['gt_boxes_list']
        print([t.shape for t in gt_boxes])
        for gts in gt_boxes:  # each frame

            # 这句话有点意思，emm，因为这样会让这一帧全部变成1
            # 添加干扰
            # print(gts)
            rois = augment_box(gts)
            # print(rois)
            # 测试iou
            ious = boxes_iou3d_gpu(rois, gts)
            # print(ious)
            max_ious = torch.diag(ious)
            # print(max_ious)
            # rcnn_labels = (max_ious > 0.3).float()
            rcnn_labels = (max_ious > self.score_threshold).float()
            # 统计各个类别数量
            rcnn_labels_03 = (max_ious > 0.3).float().sum().item()
            rcnn_labels_05 = (max_ious > 0.5).float().sum().item()
            rcnn_labels_07 = (max_ious > 0.7).float().sum().item()
            rcnn_all = rcnn_labels.shape[0]
            rcnn_res = rcnn_all - rcnn_labels_03
            rcnn_03_07 = rcnn_labels_03 - rcnn_labels_07
            rcnn_05_07 = rcnn_labels_05 - rcnn_labels_07
            rcnn_03_05 = rcnn_labels_03 - rcnn_labels_05
            print("新增数据：",rcnn_res,rcnn_03_05,rcnn_05_07,rcnn_labels_07)


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
        # 这个过程可以多做几次，算求，为了简单，只做一次
        batch_dict['rcnn_label_dict_mv_more'] = {
            'rois': [],
            'gt_of_rois': [],
            'cls_tgt': [],
            'reg_tgt': [],
            'iou_tgt': [],
        }

        rcnn_label_fv_more = batch_dict['rcnn_label_dict_fv_more']
        print([[k,v.shape]for k,v in rcnn_label_fv_more.items()])
        label_record_len = batch_dict['gt_boxes_len']
        Multi_view_box_index = batch_dict['Multi_view_box_index']
        mv_len = batch_dict['record_len']
        # print(label_record_len,mv_len)
        # print(Multi_view_box_index)
        # print(Multi_view_box_index.sum())

        # gt boxes 代表融合后，直接就能训练
        # 只针对这一个做就好了，后面mv的本身也就是基于他去操作
        gt_boxes_len = batch_dict['gt_boxes_len']
        gt_boxes = batch_dict['gt_boxes']
        start_index,start_rois,tmp_rois_len = 0,0,[]
        for i,v in enumerate(mv_len):
            for j in range(v):
                tmp_rois_now, tmp_gt_of_rois_now, tmp_cls_tgt_now, tmp_reg_tgt_now, tmp_iou_tgt_now = [], [], [], [], []
                for k in range(label_record_len[i]):
                    if Multi_view_box_index[start_index+j][k]==1:
                        tmp_rois_now.append(rcnn_label_fv_more['rois'][start_rois+k].unsqueeze(0))
                        tmp_gt_of_rois_now.append(rcnn_label_fv_more['gt_of_rois'][start_rois+k].unsqueeze(0))
                        tmp_cls_tgt_now.append(rcnn_label_fv_more['cls_tgt'][start_rois+k].unsqueeze(0))
                        tmp_reg_tgt_now.append(rcnn_label_fv_more['reg_tgt'][start_rois+k].unsqueeze(0))
                        tmp_iou_tgt_now.append(rcnn_label_fv_more['iou_tgt'][start_rois+k].unsqueeze(0))
                if len(tmp_rois_now)>0:
                    # tmp_rois.append(torch.cat(tmp_rois_now,dim=0))
                    # tmp_gt_of_rois.append(torch.cat(tmp_gt_of_rois_now,dim=0))
                    # tmp_cls_tgt.append(torch.cat(tmp_cls_tgt_now,dim=0))
                    # tmp_reg_tgt.append(torch.cat(tmp_reg_tgt_now,dim=0))
                    # tmp_iou_tgt.append(torch.cat(tmp_iou_tgt_now,dim=0))

                    batch_dict['rcnn_label_dict_mv_more']['rois'].append(torch.cat(tmp_rois_now,dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['gt_of_rois'].append(torch.cat(tmp_gt_of_rois_now,dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['cls_tgt'].append(torch.cat(tmp_cls_tgt_now,dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['reg_tgt'].append(torch.cat(tmp_reg_tgt_now,dim=0))
                    batch_dict['rcnn_label_dict_mv_more']['iou_tgt'].append(torch.cat(tmp_iou_tgt_now,dim=0))

                    tmp_rois_len.append(len(tmp_rois_now))
                else:
                    print("为啥会是0？",Multi_view_box_index[start_index+j].sum())
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
    # 1.mv 和 fv
    # 2.带误差的mv和fv
    # mv_loss 只需要1
    # 但是roi_head 两个都需要

    def forward(self, batch_dict):
        batch_dict = self.assign_targets(batch_dict)

        # 统计相关信息：
        print(batch_dict['point_coords'][0].shape)
        print([t.shape for t in batch_dict['point_coords']],batch_dict['batch_size'])
        print([t.shape for t in batch_dict['merge_point_coords']])

        print(len(batch_dict['point_coords']),batch_dict['point_coords'][0].shape,batch_dict['origin_lidar_for_vsa_project'].shape,batch_dict['origin_lidar_for_vsa_noproject'].shape)
        print(len(batch_dict['origin_lidar']),batch_dict['origin_lidar'][0].shape)


        # 可视化代码
        # 上色+清晰化试图+限制坐标
        if batch_dict['batch_size']>=2:
            self.save_png_cnt += 1
            for now_show in range(batch_dict['batch_size']):
                show_data = batch_dict['origin_lidar_for_vsa_project'][:,0].long() == now_show
                plot_grid_with_points_all_256(batch_dict['origin_lidar_for_vsa_project'][show_data][:, 1].detach().cpu().numpy(),
                                           batch_dict['origin_lidar_for_vsa_project'][show_data][:, 2].detach().cpu().numpy(),

                                           # 这个不太行，我们要看的是256框
                                           # batch_dict['boxes_fused'][0].detach().cpu().numpy(),
                                          # 这256个框最好由不同颜色划分开
                                          batch_dict['rcnn_label_dict']['rois'].detach().cpu().numpy(),

                                           # 这个没问题，因为他确实是gt box
                                           batch_dict['rcnn_label_dict']['gt_box_show'].detach().cpu().numpy(),

                                           #
                                           batch_dict['rcnn_label_dict']['iou_tgt'].detach().cpu().numpy(),
                                           batch_dict['rcnn_label_dict']['cls_tgt'].detach().cpu().numpy(),
                                            now_show,
                                            str(self.save_png_cnt)+"_s_"+str(now_show)+"_c"
                                           )

            plot_grid_with_points_all_256_mv(batch_dict['origin_lidar_for_vsa_project'],
                                       # 这个不太行，我们要看的是256框
                                       # batch_dict['boxes_fused'][0].detach().cpu().numpy(),
                                      # 这256个框最好由不同颜色划分开
                                      batch_dict['rcnn_label_dict']['rois'].detach().cpu().numpy(),

                                       # 这个没问题，因为他确实是gt box
                                       batch_dict['rcnn_label_dict']['gt_box_show'].detach().cpu().numpy(),

                                       batch_dict['rcnn_label_dict']['iou_tgt'].detach().cpu().numpy(),
                                       batch_dict['rcnn_label_dict']['cls_tgt'].detach().cpu().numpy(),
                                        batch_dict['batch_size'],
                                        str(self.save_png_cnt)+"_s_s"
                                    )
            plot_grid_with_points_all_256_mv_with_boxes(batch_dict['origin_lidar_for_vsa_project'],
                                             # 这个不太行，我们要看的是256框
                                             # batch_dict['boxes_fused'][0].detach().cpu().numpy(),
                                             # 这256个框最好由不同颜色划分开
                                             batch_dict['rcnn_label_dict']['rois'].detach().cpu().numpy(),

                                             # 这个没问题，因为他确实是gt box
                                             batch_dict['rcnn_label_dict']['gt_box_show'].detach().cpu().numpy(),

                                             batch_dict['rcnn_label_dict']['iou_tgt'].detach().cpu().numpy(),
                                             batch_dict['rcnn_label_dict']['cls_tgt'].detach().cpu().numpy(),
                                             batch_dict['batch_size'],
                                             str(self.save_png_cnt) + "_s_de"
                                            )
        # batch_dict['project_vsa_bs'] = []
        # batch_dict['noproject_vsa_bs'] = []

        # 这俩点云里面的东西怎么会一样呢？到底投影了吗？
        # print("投影的点云：  ",[t.shape for t in batch_dict['project_vsa_bs']])
        # print("没有投影的点云：  ",[t.shape for t in batch_dict['noproject_vsa_bs']])
        #
        # tmp_bs_start,tmp_bs_end = 0,0
        # for select in range(now_show):
        #     tmp_bs_start += batch_dict['record_len'][select]
        # tmp_bs_end = tmp_bs_start + batch_dict['record_len'][now_show]

        # for show in range(tmp_bs_start,tmp_bs_end):
        #
        #     plot_grid_with_points_all_256(batch_dict['noproject_vsa_by_car'][show][:, 0].detach().cpu().numpy(),
        #                                   batch_dict['noproject_vsa_by_car'][show][:, 1].detach().cpu().numpy(),
        #
        #                                   # 这个不太行，我们要看的是256框
        #                                   # batch_dict['boxes_fused'][0].detach().cpu().numpy(),
        #                                   # 这256个框最好由不同颜色划分开
        #                                   batch_dict['rcnn_label_dict']['rois'].detach().cpu().numpy(),
        #
        #                                   # 这个没问题，因为他确实是gt box
        #                                   batch_dict['rcnn_label_dict']['gt_box_show'].detach().cpu().numpy(),
        #
        #                                   #
        #                                   batch_dict['rcnn_label_dict']['iou_tgt'].detach().cpu().numpy(),
        #                                   batch_dict['rcnn_label_dict']['cls_tgt'].detach().cpu().numpy(),
        #                                   "before  "+str(show)
        #                                   )
        #     # #
        #     plot_grid_with_points_all_256(batch_dict['project_vsa_by_car'][show][:, 0].detach().cpu().numpy(),
        #                                   batch_dict['project_vsa_by_car'][show][:, 1].detach().cpu().numpy(),
        #
        #                                   # 这个不太行，我们要看的是256框
        #                                   # batch_dict['boxes_fused'][0].detach().cpu().numpy(),
        #                                   # 这256个框最好由不同颜色划分开
        #                                   batch_dict['rcnn_label_dict']['rois'].detach().cpu().numpy(),
        #
        #                                   # 这个没问题，因为他确实是gt box
        #                                   batch_dict['rcnn_label_dict']['gt_box_show'].detach().cpu().numpy(),
        #
        #                                   #
        #                                   batch_dict['rcnn_label_dict']['iou_tgt'].detach().cpu().numpy(),
        #                                   batch_dict['rcnn_label_dict']['cls_tgt'].detach().cpu().numpy(),
        #                                   "after  "+str(show)
        #                                   )

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1). \
            contiguous().view(batch_size_rcnn, -1, self.grid_size,
                              self.grid_size,
                              self.grid_size)  # (BxN, C, 6, 6, 6)
        shared_features = self.shared_fc_layers(
            pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, 1 or 2)
        rcnn_iou = self.iou_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, 1)
        rcnn_reg = self.reg_layers(shared_features).transpose(1,
                                                              2).contiguous().squeeze(
            dim=1)  # (B, C)

        # print("对结果影响")
        # print(rcnn_reg)
        # print(rcnn_cls)
        # print(rcnn_iou)

        batch_dict['stage2_out'] = {
            'rcnn_cls': rcnn_cls,
            'rcnn_iou': rcnn_iou,
            'rcnn_reg': rcnn_reg,
        }


        mv_box_loss = torch.tensor(0.).to(rcnn_cls.device)
        cls_box_loss = torch.tensor(0.).to(rcnn_cls.device)
        loss_cls_gt_mv = torch.tensor(0.).to(rcnn_cls.device)
        loss_cls_gt_fused = torch.tensor(0.).to(rcnn_cls.device)

        # 如果这个场景下没有框就算求，有可能有框，但是框里面点太少，所有在gt做运算的时候还=0
        if batch_dict['Multi_view_box_index'].sum()>0 and sum(batch_dict['gt_boxes_len']) > 0 :
        # if sum(batch_dict['gt_boxes_len']) > 0 and (
        #             self.mv_box_weight > 0 or self.cls_box_weight > 0):
            # gt fused
            pooled_features_gt_fused = self.gt_fused_roi_grid_pool(batch_dict,  batch_dict['gt_boxes'] )
            batch_size_gt = pooled_features_gt_fused.shape[0]
            pooled_features_gt_fused = pooled_features_gt_fused.permute(0, 2, 1). \
                contiguous().view(batch_size_gt, -1, self.grid_size,
                                  self.grid_size,
                                  self.grid_size)  # (BxN, C, 6, 6, 6)

            shared_features_gt_fused = self.shared_fc_layers(
                pooled_features_gt_fused.view(batch_size_gt, -1, 1))
            # 这是fused_gt对应的分数，如果需要用的话，还要把一些信息加上，因为融合后和融合前要一一对齐
            rcnn_cls_gt_fused = self.cls_layers(shared_features_gt_fused).transpose(1,
                                                                        2).contiguous().squeeze(
                dim=1)


            # gt mv
            pooled_features_gt = self.gt_roi_grid_pool(batch_dict, batch_dict['gt_boxes'])
            # fake 系列不用了
            # self.fake_cls_head.load_state_dict(self.cls_layers.state_dict())
            # self.fake_shared_fc_layers.load_state_dict(self.shared_fc_layers.state_dict())
            # self.fake_reg_layers.load_state_dict(self.reg_layers.state_dict())
            # print("直接一次就行吗？",pooled_features_gt.shape,pooled_features.shape)

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
                # print("转换器前 ：",shared_features_gt.shape)
                shared_features_gt = self.convertor(shared_features_gt)
                # print("转换器后 ：",shared_features_gt.shape)

            # 用来约束同一类别的所有框，这里需要一下sigmod
            rcnn_cls_gt =  self.cls_layers(shared_features_gt).transpose(1,
                                                                           2).contiguous().squeeze(
                dim=1)  # (B, 1 or 2)


            # 用来约束不同视角下的框
            rcnn_reg_gt = self.reg_layers(shared_features_gt).transpose(1,
                                                                        2).contiguous().squeeze(
                dim=1)  # (B, C)
            # 这个iou有点东西，我想盯住这个iou来算
            rcnn_iou_gt = self.iou_layers(shared_features_gt).transpose(1,
                                                                        2).contiguous().squeeze(
                dim=1)  # (B, C)


            # 添加信息，并且是加了噪声的！
            if self.add_more_augmented_supervision:
                batch_dict = self.assign_targets_fv_more(batch_dict)
                # 这里要对融合前和融合后分别做，对他们做监督，对gt的 mv 监督 和 fv监督
                pooled_features_gt_fused_more = self.gt_fused_roi_grid_pool(batch_dict, batch_dict['rcnn_label_dict_fv_more']['rois'])
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
                # 这是fused_gt对应的分数，如果需要用的话，还要把一些信息加上，因为融合后和融合前要一一对齐
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
                cls_gt_mv = rcnn_cls_gt.view(1,-1,1)
                cls_gt_fv = rcnn_cls_gt_fused.view(1,-1,1)
                cls_tgt_gt_mv = batch_dict['rcnn_label_dict_mv']["cls_tgt_mv"].view(1,-1,1)
                cls_tgt_gt_fv = batch_dict['rcnn_label_dict_mv']["cls_tgt_fv"].view(1,-1,1)

                loss_cls_gt_mv = weighted_sigmoid_binary_cross_entropy(cls_gt_mv, cls_tgt_gt_mv)
                loss_cls_gt_fused = weighted_sigmoid_binary_cross_entropy(cls_gt_fv, cls_tgt_gt_fv)

            # 再这一步其实就能算出mv_loss和cls_loss了
            if self.mv_box_weight > 0 or self.cls_box_weight > 0:
                # 使用sigmod观察，要不要加sigmod
                # print(rcnn_cls_gt)
                if self.mv_loss_sigmod:
                    rcnn_cls_gt = rcnn_cls_gt.sigmoid().view(-1)
                    rcnn_cls_gt_fused = rcnn_cls_gt_fused.sigmoid().view(-1)
                else:
                    rcnn_cls_gt = rcnn_cls_gt.view(-1)
                    rcnn_cls_gt_fused = rcnn_cls_gt_fused.view(-1)

                # print("mv output:")
                # print(rcnn_cls_gt)
                # print("fv output:")
                # print(rcnn_cls_gt_fused)

                # 基于分类的box理论上是可以直接算的，不用任何操作
                cls_box_data_1 = rcnn_cls_gt.unsqueeze(1).repeat(1,rcnn_reg_gt.shape[0],1)
                cls_box_data_2 = rcnn_cls_gt.unsqueeze(0).repeat(rcnn_reg_gt.shape[0],1,1)

                # print("计算cls_box信息",cls_box_data_1.shape,cls_box_data_2.shape)
                # print(rcnn_reg_gt)
                # print(rcnn_cls_gt)
                # print(rcnn_iou_gt)

                if rcnn_cls_gt.shape[0]>1:
                    cls_box_loss = torch.norm(cls_box_data_1 - cls_box_data_2, p=1).sum()*self.cls_box_weight / (
                            rcnn_cls_gt.shape[0] * (rcnn_cls_gt.shape[0] - 1))

                # 第一步，根据个数将其拆分开
                car_box_num = batch_dict['gt_box_record_len']
                start, mv_box_loss_data = 0, []
                # bs*car*box,1 -> box1,1  box2,1 …… boxn,1  这个是按照car去拆分
                for t in car_box_num:
                    mv_box_loss_data.append(rcnn_cls_gt[start:start + t])
                    # mv_box_loss_data.append(rcnn_reg_gt[start:start + t])
                    start += t

                # 把多个bs下面的场景级别gt也拆分
                bs_box_num =batch_dict['gt_boxes_len']
                start, fused_box_loss_data = 0, []
                # bs*car*box,1 -> box1,1  box2,1 …… boxn,1  这个是按照bs去拆分
                for t in bs_box_num:
                    fused_box_loss_data.append(rcnn_cls_gt_fused[start:start + t])
                    start += t

                box_loss_data_bs, box_index, point_num_bs = [], [], []
                bs_len = batch_dict['record_len']
                # 现在的数据格式是 b*n,100
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
                # a,b = []
                # print("初始维度：",Multi_view_box_index.shape,points_in_boxes.shape,rcnn_cls_gt.shape)
                # [n1,1  n2,1],[n3,1 n4,1 …… nx,1] -> [mv1*1,mv2*2,mv3*3,……] len = box 最后分割为需要的数据
                # 问题是num的长度是100……
                box_loss_data_mv, point_num_mv = [], []
                label_record_len = batch_dict['gt_boxes_len']
                for bs_fused_data, bs_data, bs_num, bs_index, bs_box_len in zip(fused_box_loss_data ,box_loss_data_bs, point_num_bs, box_index, label_record_len):
                    tmp = [[] for t in range(bs_box_len)]
                    tmp_num = [[] for t in range(bs_box_len)]

                    for index,object_list in enumerate(tmp):
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
                        if len(t) > 0:
                            box_loss_data_mv.append(torch.cat(t, dim=0))

                    for t in tmp_num:
                        if len(t) > 0:
                            point_num_mv.append(torch.cat(t,dim=0))


                # box_loss_data_shape = [t.shape for t in mv_box_loss_data]
                # box_loss_data_bs_shape = [[j.shape for j in t] for t in box_loss_data_bs]
                # box_loss_data_mv_shape = [t.shape for t in box_loss_data_mv]
                # print("loss 计算结果：",box_loss_data_shape,box_loss_data_bs_shape,box_loss_data_mv_shape)
                # print(box_loss_data_mv)
                # print(point_num_mv)
                box_mse_data = []
                for i, data in enumerate(box_loss_data_mv):
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

                        # 试图添加一个新的朝向：融合后向融合前靠齐
                        if self.limit_mv_to_fused:
                            # 融合后的信息需要detach
                            entroy_box_datas1 = data[0].squeeze().repeat(repeat_num).detach()
                            entroy_box_datas2 = t


                        # print(entroy_box_datas1)
                        # print(entroy_box_datas2)

                        # num
                        # data1 [1,2] [[1,2]] [[1,2],[1,2]]
                        # data2 [1,2] [[1],[2]] [[1,1],[2,2]]
                        # nums1 - num2 = - (num2 - num1) = [[0,1],[-1,0]] 只会选择那一个对象

                        # cls
                        # data1 [1,2] [[1,2]] [[1,2],[1,2]]
                        # data2 [1,2] [[1],[2]] [[1,1],[2,2]]  x
                        # 2视角固定，1视角向着2视角移动，原因在于1视角的点比2视角少！

                        # limit远处的框
                        # entroy_box_datas1 = t.squeeze().unsqueeze(0).repeat(*dim_repeat1)
                        # entroy_box_datas2 = t.squeeze().unsqueeze(1).repeat(*dim_repeat2).detach()
                        #
                        # box_pointnum1 = point_num_mv[i].squeeze().unsqueeze(0).repeat(*dim_repeat1)
                        # box_pointnum2 = point_num_mv[i].squeeze().unsqueeze(1).repeat(*dim_repeat2)
                        # select_pointnum = (box_pointnum1 - box_pointnum2) < 0
                        # select_data = (entroy_box_datas1 - entroy_box_datas2)[select_pointnum]

                        # print(111,entroy_box_datas1,entroy_box_datas2,select_data,select_pointnum,point_num_mv[i])


                        if self.mv_loss_compute == "l1":
                            # L1
                            box_mse_data.append(torch.norm(entroy_box_datas1 - entroy_box_datas2, p=1).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))
                        elif self.mv_loss_compute == "kl":
                            # kl散度
                            # print(kl_divergence(entroy_box_datas1 , entroy_box_datas2),entroy_box_datas1.shape)
                            box_mse_data.append(kl_divergence(entroy_box_datas1 , entroy_box_datas2).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))
                        elif self.mv_loss_compute == "js":
                            # js散度
                            box_mse_data.append(js_divergence(entroy_box_datas1 , entroy_box_datas2).unsqueeze(0) / (
                                    repeat_num * (repeat_num - 1)))
                        else:
                            # 默认L1
                            box_mse_data.append(torch.norm(entroy_box_datas1 - entroy_box_datas2, p=1).unsqueeze(0) / (
                                        repeat_num * (repeat_num - 1)))


                        #  这个limit有大问题！！！！！，为什么会反常增高！之前好像没有这个问题阿：
                        # 1.修改了一些代码，没能发现，导致直接寄！（80%）
                        # 2.这部分一直有问题，之前没发现（20%）
                        # limit
                        # if select_pointnum.sum()>0:
                        #     box_mse_data.append(torch.norm(select_data, p=1).unsqueeze(0) / select_pointnum.sum())
                        # else:
                        #     print("全部相等？？",box_pointnum1,box_pointnum2)

                if len(box_mse_data) > 0:
                    mv_box_loss = torch.cat(box_mse_data, dim=0).sum() * self.mv_box_weight / len(bs_len)

        batch_dict['mv_box_loss'] = mv_box_loss
        batch_dict['cls_box_loss'] = cls_box_loss

        batch_dict['loss_cls_gt_mv'] = loss_cls_gt_mv
        batch_dict['loss_cls_gt_fused'] = loss_cls_gt_fused
        print("算出来的loss",mv_box_loss,cls_box_loss,loss_cls_gt_mv,loss_cls_gt_fused)
        return batch_dict




# 测试，下述代码无bug
import torch
import torch.nn.functional as F


# 定义 KL 散度函数
def kl_divergence(logits1, logits2):
    # 将1维的logits通过sigmoid转换为概率
    prob1 = logits1.view(-1,1)
    prob2 = logits2.view(-1,1)

    # 将每个目标框的概率值组成二分类的分布 (p, 1-p)
    prob1 = torch.cat([prob1, 1 - prob1], dim=-1)
    prob2 = torch.cat([prob2, 1 - prob2], dim=-1)
    # print("kl 内部：",prob1.shape,prob2.shape)
    # print("mean shape ",F.kl_div(prob1.log(), prob2, reduction='mean'))
    # print("none shape ",F.kl_div(prob1.log(), prob2, reduction='none'))
    # print("bstch mean ",F.kl_div(prob1.log(), prob2, reduction='batchmean'))

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
    p = F.softmax(p,dim=-1)
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
print("KL散度:", kl_loss.item())
print("JS散度:", js_loss.item())
