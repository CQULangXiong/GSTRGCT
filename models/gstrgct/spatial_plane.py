import torch.nn.functional as F
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from scipy.sparse.linalg import eigs
from math import sqrt
from data import WeightProcess
'''
空间权重是由不同的空间节点拓扑关系组成的，是先验的信息
* 一种是收集到了一定的先验信息，语义权重关系由先验关系通过神经网络学习
* 另一种是无法获取先验信息，使用自适应的权重关系，参数化学习
* so,如何结合，并且具有动态空间权重关系？
* 建模：全局尺度因子lambda, 局部尺度因子mu
'''

# 自适应动态空间权重
class DASW(nn.Module):
    '''
    自适应空间权重-->向量嵌入-->降维减少参数
    '''
    def __init__(self, in_len, num_nodes, embed_dim, g_lambda, l_mu):
        '''
        动态自适应空间权重,该类用来微调波动的空间结构关系，学习的是D^(-1/2)AD^(-1/2)
        '''
        super(DASW, self).__init__()
        self.in_len = in_len
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim
        self.g_lambda = g_lambda
        self.l_mu = l_mu

        # 动态空间节点嵌入向量
        # 目的是减少需要学习的参数
        self.dn_embeddings = nn.Parameter(torch.randn(in_len, num_nodes, embed_dim),
                                          requires_grad=True)    # [T, N, embed_dim]

    def scaled_Laplacian(self, s_w):
        '''
        计算的是L_hat = 2/lambda_max-I
        lambda_max是L=I-D^(-1/2)AD(-1/2)=D-A的最大特征值
        '''

        D = torch.diag(torch.sum(s_w, dim=1))
        L = D - s_w
        L = torch.nan_to_num(L)
        lambda_max = max([torch.nan_to_num(torch.real(i)) for i in torch.linalg.eigvals(L)])
        return torch.nan_to_num((2 * L) / lambda_max - torch.eye(s_w.shape[0]).to(L.device))

    def forward(self, s_w): # [B, T, N, C]
        local_supports = F.softmax(F.relu(torch.einsum('tne, tse->tns', self.dn_embeddings, self.dn_embeddings)),
                                   dim=-1) # [T, N, N]
        if s_w is not None:
            global_supports = torch.stack([self.scaled_Laplacian(s_w) for _ in range(self.in_len)]) # [T, N, N]
            return self.g_lambda * global_supports + self.l_mu * local_supports
        return local_supports

class SRGCN(nn.Module):
    '''
    空间面的回归模型
    '''

    def __init__(self, in_len, num_nodes, embed_dim, g_lambda, l_mu,
                 in_dim, out_dim, cheb_k, spatial_attention=False):
        super(SRGCN,self).__init__()
        self.in_len = in_len
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.cheb_k = cheb_k
        self.spatial_attention = spatial_attention
        self.dasw = DASW(in_len, num_nodes, embed_dim, g_lambda, l_mu)

        # 广义时空回归参数学习
        self.alpha = nn.Parameter(torch.randn(in_len, cheb_k), requires_grad=True)
        self.beta = nn.Parameter(torch.randn(in_len, cheb_k, in_dim, out_dim), requires_grad=True)
        self.theta = nn.Parameter(torch.randn(in_len, cheb_k, in_dim, out_dim), requires_grad=True)
        # self.alpha = nn.Parameter(torch.randn(cheb_k), requires_grad=True)
        # self.beta = nn.Parameter(torch.randn(cheb_k, in_dim, out_dim), requires_grad=True)
        # self.theta = nn.Parameter(torch.randn(cheb_k, in_dim, out_dim), requires_grad=True)

    def cheb_polynomial(self, L_hat):
        '''
        切比雪夫多项式T_k(L_hat)
        cheb_k如果是2，那么有3项，阶数是0，1，2排列的
        # cheb_k>=3,才开始添加，否则多项式只有2项，因此建议cheb_k>=2
        '''
        cheb_polynomials = [torch.eye(self.num_nodes).to(L_hat.device), L_hat.repeat(1, 1)]  # 0阶，1阶
        for i in range(2, self.cheb_k):
            cheb_polynomials.append(2 * L_hat * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        return torch.stack(cheb_polynomials)   # [cheb_k, N,N] # cheb_k>=3,才开始添加，否则多项式只有2项，因此建议cheb_k=2...

    def forward(self, s_w, x): # [N,N], [B,T,N,C]
        B, T, N, C = x.shape
        # 动态自适应的空间权重-->L_hat
        supports = self.dasw(s_w) # [T, N, N]
        # supports = torch.stack([self.dasw.scaled_Laplacian(s_w) for _ in range(self.in_len)])
        # 计算每个时间的切比雪夫多项式
        T_L_hat = torch.stack([self.cheb_polynomial(t_supports) for t_supports in supports]) # [T, cheb_k, N, N]
        # ln, x_t, w_x_t
        l_n = torch.ones((self.in_len, self.num_nodes)).to(x.device) # [T,N]
        x_t = x # [B,T,N,C]
        # w*x_t,此时w是L_hat
        w_x_t = torch.einsum('btnc, tnn->btnc', x_t, supports) # [B,T,N,C]
        x_t_hat = torch.concat((l_n.unsqueeze(0).unsqueeze(-1).repeat(B,1,1,1),x_t, w_x_t), dim=-1) # [B,T,N,2C+1]
        theta_hat = torch.concat((self.alpha.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,self.out_dim),self.beta,self.theta), dim=-2) # [T,cheb_k,2C+1, F]
        s_r = torch.einsum('btns,tksf->btknf', x_t_hat, theta_hat)
        # 图卷积， T_L_hat * X_t_hat * theta
        out = torch.einsum('tknn, btknf->btnf', T_L_hat, s_r)
        if self.spatial_attention:
            return F.relu(out), supports, x_t_hat
        return F.relu(out), None, x_t_hat


if __name__ == '__main__':
    s_w = WeightProcess(root_path='../dataset', num_nodes=307, dataset='pems04').s_w
    print(s_w.min())
    # device = torch.device('cuda')
    # s_w = torch.rand(307,307).to(device)
    # x = torch.rand(64,12,307,3).to(device)
    # # model = DASW(in_len=12, num_nodes=307, embed_dim=10, g_lambda=0.5, l_mu=0.5)
    # model1 = SRGCN(in_len=12, num_nodes=307, embed_dim=10, g_lambda=0.5, l_mu=0.5,
    #               in_dim=3, out_dim=512, cheb_k=2, spatial_attention=True).to(device)
    # # supports= model(s_w)
    # out, s_attn, x_t_hat = model1(s_w, x)
    # # print(supports.shape)
    # print(out.shape)
    # print(s_attn.shape)
    # print(x_t_hat.shape)











