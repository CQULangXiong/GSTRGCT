import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os
from models.gstrgct.embedding import PositionalEmbedding, DataEmbedding

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    @property
    def mask(self):
        return self._mask

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        return x_hat

import torch.nn as nn



class MyLinear(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim):
        super(MyLinear, self).__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.list_Linear = nn.ModuleList()
        for _ in range(self.num_nodes):
            self.list_Linear.append(nn.Linear(input_dim, output_dim))

    def forward(self, x): # [B,N,L,C]
        out = []
        for i, linear in enumerate(self.list_Linear):
            out.append(linear(x[:, i, :, :]))
        return torch.stack(out, dim=1)

    def __repr__(self): # 确保后面只打印一行，而不是N行
        return f"MyLinear containing {len(self.list_Linear)} linear layers"
class MyConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=1):
        super(MyConv, self).__init__()
        padding = (kernel_size-1) // 2
        self.input_dim = input_dim
        self.output_dim = output_dim #
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size,
                              padding=padding, padding_mode='circular')

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x): # [B,N,L,d_model):
        out = self.conv(x.permute(0,3,1,2))
        return out.permute(0,2,3,1)




class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False, speed=False, full_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.speed = speed
        self.full_attention = full_attention

    def time_delay_agg(self, values, corr): #  [64, 307, 8, 64, 12], [64, 307, 8, 64, 12]
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        复杂度O(NLlogL)
        """
        #
        batch = values.shape[0]
        node = values.shape[1]
        head = values.shape[2]
        channel = values.shape[3]
        length = values.shape[4]
        # find top k
        top_k = int(self.factor * math.log(length))
        if self.speed:  # 空间注意力取平均
            mean_value = torch.mean(torch.mean(torch.mean(corr, dim=1), dim=1), dim=1)
            index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
            weights = torch.stack([mean_value[:, i] for i in index], dim=-1)
            tmp_corr = torch.softmax(weights, dim=-1)  # [64,2]
            tmp_values = values
            delays_agg = torch.zeros_like(values).float()
            for i in range(top_k):
                pattern = torch.roll(tmp_values, -int(index[i]), -1)  # [64,307,8,64,12]
                delays_agg += pattern * (
                    tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, head, channel,
                                                                                              length))
            return delays_agg
        mean_value = torch.mean(torch.mean(corr, dim=2), dim=2) # [64, 307, 12]
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1] # [307, 2]
        weights = torch.stack([torch.stack([node_mean_value[:, node_index[i]] for i in range(top_k)],
                                           dim=-1) for (node_mean_value, node_index) in zip(mean_value.permute(1, 0, 2), index)], dim=1) # [64,307,2]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1) # (64, top_k)-->[64, 307, 2]
        # aggregation
        tmp_values = values  #[B, N, H,E,L)-->[64, 307, 8, 64, 12]
        delays_agg = torch.zeros_like(values).float()
        delays_agg1 = []
        for node_delays_agg, node_tmp_values, node_index, node_tmp_corr in zip(delays_agg.permute(1, 0, 2, 3, 4),
                                                                               tmp_values.permute(1, 0, 2, 3, 4), index,
                                                                               tmp_corr.permute(1, 0, 2)):
            for j in range(top_k):
                node_pattern = torch.roll(node_tmp_values, -int(node_index[j]), -1)  # [64, 8, 64, 12]-->延迟
                node_delays_agg = node_delays_agg + node_pattern * (
                    node_tmp_corr[:, j].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, 1, channel, length)) # 聚合
            delays_agg1.append(node_delays_agg)
        delays_agg1 = torch.stack(delays_agg1, dim=1)   # [64, 307, 8, 64, 12]
        return delays_agg1

    def forward(self, queries, keys, values, attn_mask):
        B, N, L, H, E = queries.shape # [64, 307, 12, 8, 64]
        B, N, S, H, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :, :L, :, :]
            keys = keys[:, :, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 1, 3, 4, 2).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft) # [64, 307, 8, 64, 7]
        corr = torch.fft.irfft(res, dim=-1) # [64, 307, 8, 64, 12]
        # 计算自注意力
        if self.full_attention:
            scale = self.scale or 1. / sqrt(E)
            # scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            scores = torch.einsum('bnlhe, bnshe->bnhls', queries, keys)
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores.masked_fill_(attn_mask.mask, -np.inf)
            attn = self.dropout(torch.softmax(scale * scores, dim=-1)) # [B,N,H,L,L]
            # V = torch.einsum("bhls,bshd->blhd", A, values)
            V = torch.einsum("bnhls,bnshd->bnlhd", attn, values)
            if self.output_attention:
                return (V.contiguous(), attn)  # [64, 307, 8, 64, 12]-->[64,307,12,8,64]
            else:
                return (V.contiguous(), None)


        V = self.time_delay_agg(values.permute(0, 1, 3, 4, 2).contiguous(), corr).permute(0, 1, 4, 2, 3)# [b,n,l,h,e]
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 1, 4, 2, 3)) # [64, 307, 8, 64, 12]-->[64,307,12,8,64]
        else:
            return (V.contiguous(), None)

class AutoCorrelationLayer(nn.Module): #多头
    def __init__(self, correlation, d_model, n_heads, num_nodes,
                 d_keys=None,d_values=None):
        super(AutoCorrelationLayer, self).__init__()
        self.num_nodes = num_nodes
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        #
        # self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        # self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.query_projection = MyLinear(num_nodes, d_model, d_keys * n_heads)
        self.key_projection =  MyLinear(num_nodes, d_model, d_keys * n_heads)
        self.value_projection =  MyLinear(num_nodes, d_model, d_keys * n_heads)
        # self.query_projection = MyConv(d_model, d_keys * n_heads)
        # self.key_projection =  MyConv(d_model, d_keys * n_heads)
        # self.value_projection =  MyConv(d_model, d_keys * n_heads)

        # self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.out_projection = MyLinear(num_nodes, d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask): # [64, 307, 12, 512]
        B, N, L, _ = queries.shape # -是d_model
        _, _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, N, L, H, -1)
        keys = self.key_projection(keys).view(B, N, S, H, -1)
        values = self.value_projection(values).view(B, N, S, H, -1)
        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, N, L, -1)
        return self.out_projection(out), attn

class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # self.linear1 = nn.Linear(d_model, d_ff)
        # self.linear2 =  nn.Linear(d_ff, d_model)
        self.conv1 = nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, attn_mask=None): # [64, 307, 12 ,512]  -> [64,12,512]
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x = self.norm1(x)   # 残差连接 #注意力机制计算的输出，为了提高泛化能力，随机以dropout的概率使得输出注意力为0

        y = x  # [64, 307, 12 ,512]-->[64,12,512]
        # 前馈神经网络
        # 第一个线性层
        y = self.dropout(self.activation(self.conv1(y.permute(0, 3, 2, 1)))) # [64, 512, 12 ,307]-->[64, 2048, 12 ,307]
        # y = self.dropout(self.activation(self.conv1(y.permute(0, 2, 1))))
        # y = self.dropout(self.activation(self.linear1(y)))  # [64, 307, 12 ,2048]
        # 第2个线性层
        y = self.dropout(self.conv2(y).permute(0, 3, 2, 1))
        # y = self.dropout(self.conv2(y).permute(0, 2, 1)) # [64,12,512]
        # y = self.dropout(self.linear2(y))  # [64, 307, 12 ,512]

        return self.norm2(x + y), attn

class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
class AutoTRT(nn.Module):
    def __init__(self, num_nodes, channel, d_model, n_heads, dropout, num_layers, factor, output_attention, full_attention): #
        super(AutoTRT, self).__init__()
        self.channel = channel
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.num_layers = num_layers
        self.output_attention = output_attention
        self.factor = factor
        # self.speed = speed
        self.embed = DataEmbedding(channel, d_model, dropout)
        self.auto_correlation = AutoCorrelation(mask_flag=False, factor=factor, scale=None,
                                                attention_dropout=dropout,output_attention=output_attention,
                                                speed=True, full_attention=full_attention
                                                )
        self.auto_correlation_layer = AutoCorrelationLayer(self.auto_correlation, self.d_model, self.n_heads,num_nodes,
                                                           d_keys=None, d_values=None)
        self.encoder_layer = EncoderLayer(self.auto_correlation_layer, self.d_model,
                                          d_ff=None, dropout=self.dropout, activation="relu")

        self.encoder_layers = [self.encoder_layer for l in range(num_layers)]
        self.encoder = Encoder(self.encoder_layers,
                               conv_layers=None,
                               norm_layer=my_Layernorm(self.d_model))
        # self.decoder = nn.Conv2d(d_model, c_out, kernel_size=1, bias=False)
        # self.decoder = nn.Conv1d(d_model, c_out, kernel_size=1)
        # self.decoder = nn.Linear(d_model)

    def forward(self, x):
        embed_x = self.embed(x)  # [64,307,12,512]
        # embed_x = torch.mean(embed_x, dim=1) #
        enc_out, attns = self.encoder(embed_x)  # [batch_size, num_of_vertices, num_time_steps_in_put, d_model]
        if attns is not None:
            attns = torch.stack(attns, dim=0)
        return F.relu(enc_out), attns

if __name__ == "__main__":
    model = AutoTRT(num_nodes=307, channel=3, d_model=512, n_heads=8,
                    dropout=0.0, num_layers=1, factor=1, output_attention=True, full_attention=False)
    x = torch.rand(64,307,12,3)
    out2, attns = model(x)
    print(out2.shape)
    print(attns.shape)
