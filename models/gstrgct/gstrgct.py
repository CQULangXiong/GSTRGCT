import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from models.gstrgct.spatial_plane import SRGCN
from models.gstrgct.temporal_plane import AutoTRT

class GSTRGCT(nn.Module):
    def __init__(self, num_nodes, in_len, out_len, channel,
                 embed_dim,g_lambda, l_mu,
                 d_model, n_heads, num_layers, dropout, factor, cheb_k,
                 spatial_attention, temporal_attention, full_attention
                 ):
        super(GSTRGCT, self).__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        self.srgcn = SRGCN(in_len, num_nodes, embed_dim, g_lambda, l_mu,
                           channel, d_model, cheb_k, spatial_attention)
        self.auto_trt = AutoTRT(num_nodes, 2*channel+1, d_model, n_heads,
                                dropout, num_layers, factor, temporal_attention, full_attention)
        # self.auto_trt = AutoTRT(num_nodes, channel, d_model, n_heads,
        #                         dropout, num_layers, factor, temporal_attention)
        # self.auto_trt = AutoTRT(num_nodes, d_model, d_model, n_heads,
        #                         dropout, num_layers, factor, temporal_attention)
        self.mlp_head = nn.Linear(in_len*d_model, out_len)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, s_w, x): # x = [64, 12, 307, 3]
        B, _, _, _ = x.shape
        # 张量分解为空间面与时间面
        # spatial plane
        s_out, s_attention, x_t_hat = self.srgcn(s_w, x)  # [64, 12, 307, 512]
        # x_t_hat->[64, 12, 307, 7]

        # s_out已经relu过了

        # temporal plane
        t_out, t_attention = self.auto_trt(x_t_hat.permute(0, 2, 1, 3))   # [64, 307, 12,512]
        # t_out, t_attention = self.auto_trt(x.permute(0, 2, 1, 3))
        #
        # 张量融合
        st_out = s_out*t_out.permute(0,2,1,3) # [64,12,307,512]

        # 并行运算
        # st_out, t_attention = self.auto_trt(s_out.permute(0, 2, 1, 3))

        # 张量融合要不要relu,可以实验
        st_out = self.norm(st_out)
        # st_out = F.relu(st_out)
        st_out_pred = self.mlp_head(st_out.view(B, self.num_nodes, -1)).view(B, self.out_len, self.num_nodes)

        # 自回归预测  -->这里其实可以先预测后自回归，但是违背了其本质， 理论上可以迁移到空间自回归预测，以及时间自回归预测，可操作性很强
        # 比如研究影响因素，那么1就可以使用原始的in_channel，再空间自回归或者时间自回归，以及时空自回归
        # st_out_pred = self.projection(st_out.permute(0, 2, 3, 1)).permute(0,3,1,2)
        # return st_out_pred, s_attention, None
        return st_out_pred, s_attention, t_attention

if __name__ == "__main__":
    device = torch.device('cuda')
    x = torch.rand(64,24,307,3).to(device)
    s_w = torch.rand(307, 307).to(device)
    model = GSTRGCT(num_nodes=307, in_len=24, out_len=6, channel=3,
                    embed_dim=10,g_lambda=0.5, l_mu=0.5,
                    d_model=64, n_heads=8, num_layers=1, dropout=0.0, factor=1, cheb_k=2,
                    spatial_attention=True, temporal_attention=True, full_attention=False).to(device)
    out, s_attns, t_attns =  model(s_w, x)
    print(out.shape)
    print(s_attns.shape)
    print(t_attns.shape)