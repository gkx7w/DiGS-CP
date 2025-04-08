import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.loss.ciassd_loss import CiassdLoss, weighted_smooth_l1_loss


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor,
                target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss

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


def indices_to_dense_vector(
        indices, size, indices_value=1.0, default_value=0, dtype=np.float32
):
    """Creates dense vector with indices set to specific value and rest to zeros.
    This function exists because it is unclear if it is safe to use
        tf.sparse_to_dense(indices, [size], 1, validate_indices=False)
    with indices which are not ordered.
    This function accepts a dynamic size (e.g. tf.shape(tensor)[0])
    Args:
        indices: 1d Tensor with integer indices which are to be set to
            indices_values.
        size: scalar with size (integer) of output Tensor.
        indices_value: values of elements specified by indices in the output vector
        default_value: values of other elements in the output vector.
        dtype: data type.
    Returns:
        dense 1D Tensor of shape [size] with indices set to indices_values and the
            rest set to default_value.
    """
    dense = torch.zeros(size).fill_(default_value)
    dense[indices] = indices_value

    return dense

class PointPillarLossDiffusion(nn.Module):
    def __init__(self, args):
        super(PointPillarLossDiffusion, self).__init__()
        self.reg_loss_func = WeightedSmoothL1Loss()
        self.alpha = 0.25
        self.gamma = 2.0
        self.total_loss = 0
        self.diff_loss = 0
        self.rcnn_loss = 0
        self.cls_loss = 0
        self.reg_loss = 0
        self.iou_loss = 0
        self.cls = args['stage2']['cls']
        self.reg = args['stage2']['reg']
        self.iou = args['stage2']['iou']
        # 三种类型：naive,naive+,naive+W3
        self.use_iou_loss = "naive"
        # 两种类型：True,Flase
        self.use_reg_loss_focal = True
        self.loss_dict = {}

    
    def sigmoid_weight(self, max_weight, epoch):
        return max_weight / 2 * (- (np.tanh(epoch / 4 - 5)) + 1)

    def stage2_loss(self, rcnn_cls,rcnn_iou,rcnn_reg,tgt_cls,tgt_iou,tgt_reg):
        rcnn_cls = rcnn_cls[0].view(1, -1, 1)
        rcnn_iou = rcnn_iou[0].view(1, -1, 1)
        rcnn_reg = rcnn_reg[0].view(1, -1, 7)

        tgt_cls = tgt_cls.view(1, -1, 1)
        tgt_iou = tgt_iou.view(1, -1, 1)
        tgt_reg = tgt_reg.view(1, -1, 7)

        pos_norm = tgt_cls.sum()
        # cls loss
        loss_cls = weighted_sigmoid_binary_cross_entropy(rcnn_cls, tgt_cls)

        # regression loss
        # [deprecated by Yifan Lu] Target resampling : Generate a weights mask to force the regressor concentrate on low iou predictions
        # sample 50% with iou>0.7 and 50% < 0.7
        weights = torch.ones(tgt_iou.shape, device=tgt_iou.device)
        weights[tgt_cls == 0] = 0
        # coalign 这边是直接设置为0，没有后续过程
        if self.use_reg_loss_focal:
            neg = torch.logical_and(tgt_iou < 0.7, tgt_cls != 0)
            pos = torch.logical_and(tgt_iou >= 0.7, tgt_cls != 0)
            # # 这里不仅仅删除一些正样本，并且对于负样本加量学习！先来个五倍吧
            num_neg = int(neg.sum(dim=1))
            num_pos = int(pos.sum(dim=1))
            num_pos_smps = max(num_neg, 5)
            pos_indices = torch.where(pos)[1]
            not_selsected = torch.randperm(num_pos)[:num_pos - num_pos_smps]
            # not_selsected_indices = pos_indices[not_selsected]
            weights[:, pos_indices[not_selsected]] = 0
            # print("正样本，负样本数量", num_neg, num_pos, weights[weights != 0].shape)
        loss_reg = weighted_smooth_l1_loss(rcnn_reg, tgt_reg,
                                        weights=weights / max(weights.sum(),
                                                                1)).sum()

        # iou loss
        weights_iou = torch.ones(tgt_iou.shape, device=tgt_iou.device)
        # TODO: also count the negative samples
        tgt_iou = 2 * (tgt_iou - 0.5)  # normalize to -1, 1
        if self.use_iou_loss == "naive":
            # print("use naive iou loss")
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
                                            weights=tgt_cls).mean()
        elif self.use_iou_loss == "naive+":
            # print("use naive+ iou loss")
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
                                            weights=weights_iou).mean()
        elif self.use_iou_loss == "naive+W3":
            # print("use naive+W3 iou loss")
            # 这个操作有点问题捏，这里不是0.7，而是0.4，这组实验需要重新跑
            weights_iou[tgt_iou < 0.4] = 3
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
                                            weights=weights_iou).mean()
        else:
            print("else use naive iou loss")
            loss_iou = weighted_smooth_l1_loss(rcnn_iou, tgt_iou,
                                            weights=tgt_cls).mean()

        loss_cls_reduced = loss_cls * self.cls['weight']
        loss_iou_reduced = loss_iou * self.iou['weight']
        loss_reg_reduced = loss_reg * self.reg['weight']
        
        return loss_cls_reduced,loss_iou_reduced,loss_reg_reduced
        
    def cal_diff_loss(self, p_e_batch, g_e_batch,weights):
        # 初始化总损失
        diff_loss = 0.0
        # 对每个批次中的每个特征计算损失并累加
        for batch_idx in range(len(p_e_batch)):
            p_features = p_e_batch[batch_idx]
            g_features = g_e_batch[batch_idx]
            # 计算当前批次的损失
            batch_diff_loss = self.diff_loss_func(
                p_features,
                g_features,
                weights=weights
            )
            # 累加总损失
            diff_loss += batch_diff_loss
        # 计算平均损失（如果有特征）
        diff_loss = diff_loss / len(p_e_batch)
        return diff_loss
        
    def forward(self, output_dict, target_dict, epoch = 1, train = True):
        """
        Parameters
        ----------
        output_dict : dict
        target_dict : dict
        """
        
        if 'rcnn_label_dict' in output_dict.keys():
            rcnn_cls = output_dict['stage2_out']['rcnn_cls']
            rcnn_iou = output_dict['stage2_out']['rcnn_iou']
            rcnn_reg = output_dict['stage2_out']['rcnn_reg']

            tgt_cls = output_dict['rcnn_label_dict']['cls_tgt']
            tgt_iou = output_dict['rcnn_label_dict']['iou_tgt']
            tgt_reg = output_dict['rcnn_label_dict']['reg_tgt']

            loss_cls_reduced, loss_iou_reduced, loss_reg_reduced = self.stage2_loss(rcnn_cls,rcnn_iou,rcnn_reg,tgt_cls,tgt_iou,tgt_reg)
            rcnn_loss = loss_cls_reduced + loss_iou_reduced + loss_reg_reduced
            total_loss = rcnn_loss
            
            self.loss_cls_reduced = loss_cls_reduced.item()
            self.loss_iou_reduced = loss_iou_reduced.item()
            self.loss_reg_reduced = loss_reg_reduced.item()
            self.rcnn_loss = rcnn_loss.item()
            self.total_loss = total_loss.item()   
            self.loss_dict.update({
                'total_loss': self.total_loss,
                'rcnn_loss': self.rcnn_loss,
                'cls_loss': self.loss_cls_reduced,
                'iou_loss': self.loss_iou_reduced,
                'reg_loss': self.loss_reg_reduced,
            })

        if 'pred_feature' in output_dict.keys() and 'rcnn_label_dict' in output_dict.keys():
            p_e_batch = output_dict['pred_feature'] 
            g_e_batch = output_dict['gt_feature']
            diff_loss = self.cal_diff_loss(p_e_batch, g_e_batch,weights=self.sigmoid_weight(3, epoch) if train else 0) #self.sigmoid_weight(4, epoch)
            total_loss = rcnn_loss + diff_loss

            self.total_loss = total_loss.item()
            self.diff_loss = diff_loss.item()
            self.loss_dict.update({'total_loss': self.total_loss,
                                    'diff_loss': self.diff_loss
                                    })
        elif 'pred_feature' in output_dict.keys():
            p_e_batch = output_dict['pred_feature']  # 批量列表
            g_e_batch = output_dict['gt_feature']  # 批量列表
            diff_loss = self.cal_diff_loss(p_e_batch, g_e_batch,weights=1)
            total_loss = diff_loss

            self.total_loss = total_loss.item()
            self.diff_loss = diff_loss.item()
            self.loss_dict.update({'total_loss': self.total_loss,
                                   'rcnn_loss': self.rcnn_loss,
                                   'reg_loss': self.reg_loss,
                                   'cls_loss': self.cls_loss,
                                   'iou_loss': self.iou_loss,
                                   'diff_loss': self.diff_loss 
                                    })
    
        return total_loss

    def cls_loss_func(self, input: torch.Tensor,
                      target: torch.Tensor,
                      weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

    def diff_loss_func(self, pred_noise: torch.Tensor,
                      gt_noise: torch.Tensor,
                      weights: int):
        return ((gt_noise - pred_noise) ** 2).sum(dim=1).mean()*weights
    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in    
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2


    def logging(self, epoch, batch_id, batch_len, writer, pbar=None):
        """
        Print out  the loss function for current iteration.

        Parameters
        ----------
        epoch : int
            Current epoch for training.
        batch_id : int
            The current batch.
        batch_len : int
            Total batch length in one iteration of training,
        writer : SummaryWriter
            Used to visualize on tensorboard
        """
        total_loss = self.loss_dict['total_loss'] if 'total_loss' in self.loss_dict.keys() else torch.tensor(0)
        reg_loss = self.loss_dict['reg_loss'] if 'reg_loss' in self.loss_dict.keys() else torch.tensor(0)
        cls_loss = self.loss_dict['cls_loss'] if 'cls_loss' in self.loss_dict.keys() else torch.tensor(0)
        iou_loss = self.loss_dict['iou_loss'] if 'iou_loss' in self.loss_dict.keys() else torch.tensor(0)
        rcnn_loss = self.loss_dict['rcnn_loss'] if 'rcnn_loss' in self.loss_dict.keys() else torch.tensor(0)
        if pbar is None:
            if 'diff_loss' in self.loss_dict:
                diff_loss = self.loss_dict['diff_loss']
                print("[epoch %d][%d/%d], || Loss: %.3f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc: %.3f || Iou: %.3f || Diff: %.3f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss, diff_loss))
            else:
                print("[epoch %d][%d/%d], || Loss: %.4f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc Loss: %.4f|| Iou: %.3f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss))
        else:
            if 'diff_loss' in self.loss_dict:
                diff_loss = self.loss_dict['diff_loss']
                pbar.set_description("[epoch %d][%d/%d], || Loss: %.3f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc: %.3f || Iou: %.3f || Diff: %.3f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss, diff_loss))
            else:
                pbar.set_description("[epoch %d][%d/%d], || Loss: %.4f || Rcnn: %.3f|| Cls: %.3f"
                " || Loc Loss: %.4f|| Iou: %.3f" % (
                    epoch, batch_id + 1, batch_len,
                    total_loss, rcnn_loss, cls_loss, reg_loss, iou_loss))

        if 'diff_loss' in self.loss_dict:
            writer.add_scalar('Reconstruction_loss', diff_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Regression_loss', reg_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Cls_loss', cls_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Iou_loss', iou_loss,
                          epoch*batch_len + batch_id)
        writer.add_scalar('Rcnn_loss', rcnn_loss,
                          epoch*batch_len + batch_id)