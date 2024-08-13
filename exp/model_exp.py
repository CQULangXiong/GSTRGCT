import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from data import PeMSDataset, get_data_loader, WeightProcess
from exp.base_exp import BaseExp
import numpy as np
from data import WeightProcess



class Exp(BaseExp):
    # def __init__(self):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.exp_name = '%s_exp' % self.args.name

    def setup_seed(self, seed=42):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def get_model(self):
        from models import GSTRGCT
        self.model = self.args.model.to(self.get_device())
        if self.args.mode == 'train':
            self.model.train()
        else:
            self.model.eval()
        return self.model

    def get_weight(self):
        '''
        先验空间权重
        '''
        self.s_w = WeightProcess(self.args.root_path, self.args.num_nodes, self.args.dataset).s_w
        return self.s_w


    def get_dataset(self, flag):
        '''
        flag: train/val/test
        '''
        return PeMSDataset(self.args.dataset, self.args.root_path, self.args.in_len, self.args.out_len,
                           split_size_1=0.6, split_size_2=0.2, mode=flag, normalizer=self.args.normalizer)


    def get_dataloader(self, flag):
        '''
        dataloader: train_loader/val_loader/test_loader
        '''
        data_loader, self.y_scaler = get_data_loader(self.get_dataset(flag),
                                                     self.args.batch_size, self.args.num_workers,
                                                     mode=flag)
        return data_loader

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        return torch.device('cpu')

    def get_criterion(self):
        self.criterion = nn.MSELoss()
        return self.criterion

    def get_optimizer(self):
        '''torch.optim.Optimizer'''
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        return self.optimizer

    def get_lr_scheduler(self):
        '''
        learning rate of iter_per_epoch
        '''
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 3, gamma=0.96)
        return self.scheduler

    def get_eval_dataset(self):
        return self.get_dataset(flag='val')

    def get_eval_loader(self):
        return self.get_dataloader(flag='val')



if __name__ == '__main__':
    # gstrgct_exp = Exp()
    # print(os.path.split(os.path.realpath(__file__))[1].split(".")[0])
    # # gstrgct_exp = Exp()
    # s_w = gstrgct_exp.get_weight()
    # print(s_w.shape)
    s_w = WeightProcess(root_path='../dataset', num_nodes=307, dataset='pems04').s_w
    print(s_w.min())


