import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Muti_Stacked_Attention(nn.Module):
    def __init__(self, channel,out_dim):
        super(Muti_Stacked_Attention, self).__init__()
        #  比如 64 64 128 256
        #  等价于 1x 2x 4x 8x
        self.channel_list = channel
        self.stacked_num = len(channel)-1
        self.deal_q = nn.ModuleList()
        self.stacked_q = nn.ModuleList()
        self.point_feature_fusion = nn.Sequential(
            nn.Linear(sum(channel),out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
        )
        for c_in in self.channel_list:
            c_out = int(c_in/4)
            self.deal_q.append(
                nn.Sequential(
                nn.Linear(c_in, c_out),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
                )
            )
        before_q = 0
        for c_in in self.channel_list:
            c_out = int(c_in/4)
            self.stacked_q.append(
                nn.Sequential(
                nn.Linear(c_out+before_q, c_out+before_q),
                nn.BatchNorm1d(c_out+before_q),
                nn.ReLU(),
                )
            )
            before_q += c_out
        # print(self.stacked_q)

        self.deal_k = nn.ModuleList()
        for c_in in self.channel_list:
            self.deal_k.append(
                nn.Sequential(
                nn.Linear(c_in, c_in),
                nn.BatchNorm1d(c_in),
                nn.ReLU(),
                )
            )

    def forward(self,feature_list):

        res,before_q = [],None
        for i in range(len(feature_list)):
            if i == 0:
                res.append(feature_list[0])
                continue

            if i == 1:
                q = self.stacked_q[i-1](self.deal_q[i-1](res[i-1])).unsqueeze(1).repeat(1,4,1)
            else:
                # print(res[i-1].shape,before_q.shape)
                q = self.stacked_q[i-1](torch.cat([self.deal_q[i-1](res[i-1]),before_q],dim=-1)).unsqueeze(1).repeat(1,4,1)

            num,c = feature_list[i].shape
            k = self.deal_k[i](feature_list[i]).view(num,4,int(c/4))
            v = feature_list[i].view(num,4,int(c/4))
            # 计算点积
            matmul_qk = torch.matmul(q, k.transpose(-2, -1))  # q: (N, 1, c), k: (N, 4, c) -> (N, 1, 4)

            # 缩放
            dk = torch.tensor(k.size(-1), dtype=torch.float32)  # 获取键向量的维度大小
            scaled_attention_logits = matmul_qk / torch.sqrt(dk)

            # softmax 归一化
            attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # (N, 1, 4)

            # print(q.shape,k.shape,v.shape,matmul_qk.shape,attention_weights.shape)
            output = torch.matmul(attention_weights, v)  # (N, 4, 4) * (N, 4, c) -> (N, 1, c)
            res.append(output.view(num,-1))

            before_q = q[:,0,:]

        point_features = torch.cat(res, dim=1)
        outdata = self.point_feature_fusion(point_features)
        return outdata