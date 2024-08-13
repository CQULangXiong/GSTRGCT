import numpy as np
import os
import time
import torch
import torch.nn as nn
import pandas as pd

class WeightProcess:
    def __init__(self, root_path, num_nodes, dataset='pems04'):
        self.adj_filename = '%s/%s/distance.csv' % (root_path, dataset)
        self.num_nodes = num_nodes
        self.dataset = dataset
        if os.path.exists(self.adj_filename):
            self.s_w = self.get_smooth_weight_matrix(scaling=False)
        else:
            self.s_w = None
    def get_adjacency_matrix(self):
        A = np.zeros((int(self.num_nodes), int(self.num_nodes)), dtype=np.float32)
        A_distance = np.zeros((int(self.num_nodes), int(self.num_nodes)), dtype=np.float32)

        distance = pd.read_csv(self.adj_filename)
        for i in range(len(distance)):
            from_index = distance['from'][i]
            to_index = distance['to'][i]
            cost = distance['cost'][i]
            A[from_index, to_index] = 1
            A_distance[from_index, to_index] = cost
        dist_mean = distance['cost'].mean()
        dist_std = distance['cost'].std()
        return A, A_distance, dist_mean, dist_std

    def get_smooth_weight_matrix(self, scaling=True):
        """
        :param A_distance: 空间距离矩阵
        :param scaling: 决定是否采用此平滑后的权重矩阵，否则使用0/1的A
        :return:空间权重
        """
        A, A_distance, dist_mean, dist_std = self.get_adjacency_matrix()
        if scaling:
            W = A_distance
            W = np.exp(-(W - dist_mean) * (W - dist_mean) / (dist_std * dist_std)) * A
            # refer to Eq.10
        else:
            W = A # 邻接矩阵
        return W
if __name__ == '__main__':
    s_w = WeightProcess(root_path='../dataset', num_nodes=307, dataset='pems04').s_w
    print(s_w.shape)