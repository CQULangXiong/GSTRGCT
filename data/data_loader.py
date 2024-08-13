import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from utils import normalize
from sklearn.model_selection import train_test_split
from loguru import logger
from utils import custom_formatter

class PeMSDataset(Dataset):
    def __init__(self, dataset='pems04', root_path='../dataset',
                 in_len=12, out_len=3, split_size_1=0.6, split_size_2=0.2, mode='train', normalizer='StandardScaler'):
        self.root_path = root_path
        self.dataset = dataset
        self.in_len = in_len
        self.out_len = out_len
        self.split_size_1 = split_size_1
        self.split_size_2 = split_size_2
        self.mode = mode
        self.features = ['total flow', 'average occupancy', 'average speed']
        self.load_data()
        self.get_normalize_data(scale_type=normalizer)
        self.get_train_val_test()
        if not os.path.exists('./dataset_log/%s_info.log' % dataset):
            self.get_dataset_log()
        else:
            if mode == 'train':
                logger.info('dataset_log has logs, need no record again' % dataset)
                logger.remove()
            pass

    def __len__(self):
        if self.mode == "train":
            return self.train.shape[0] - self.in_len - self.out_len + 1
        elif self.mode == "val":
            return self.val.shape[0] - self.in_len - self.out_len + 1
        elif self.mode == "test":
            return self.test.shape[0] - self.in_len - self.out_len + 1
        else:
            raise ValueError

    def __getitem__(self, index):
        if self.mode == "train":
            return np.float32(self.train[index:index + self.in_len, :, :]), \
                self.train_label[index + self.in_len:index + self.in_len + self.out_len, :]
        elif self.mode == "val":
            return np.float32(self.val[index:index + self.in_len, :, :]), \
                self.val_label[index + self.in_len:index + self.in_len + self.out_len, :]
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.in_len, :, :]), \
                self.test_label[index + self.in_len:index + self.in_len + self.out_len, :]
        else:
            raise ValueError

    def load_data(self):

        self.data_path = os.path.join(self.root_path, '%s/%s.npz' % (self.dataset, self.dataset))
        self.data = np.load(self.data_path)['data']
        if self.data.shape[-1] == 3:
            self.features = ['total flow', 'average occupancy', 'average speed']
        self.num_nodes = self.data.shape[1]
        self.length = self.data.shape[0]

    def get_normalize_data(self, scale_type="StandardScaler"):  # MinMaxScaler
        self.X = []
        for i in range(self.data.shape[1]):
            if scale_type is not None:
                x_scaler = normalize(scale_type)
                x = x_scaler.fit_transform(self.data[:,i,:])
            else:
                x = self.data[:,i,:]
            self.X.append(x)
        self.X = np.transpose(np.array(self.X), (1,0,2)) # [l,n,3]
        Y = self.data[:, :, 0]  # [l,n]
        if scale_type is not None:
            self.y_scaler = normalize(scale_type)
            self.Y = self.y_scaler.fit_transform(Y)
        else:
            self.Y = Y
            self.y_scaler = None

    def get_train_val_test(self):
        split_1 = int(self.length * self.split_size_1)
        split_2 = int(self.length * (self.split_size_1 + self.split_size_2))

        self.train = self.X[:split_1, :, :]
        self.train_label = self.Y[:split_1,:]

        self.val = self.X[split_1:split_2, :, :]
        self.val_label = self.Y[split_1:split_2, :]

        self.test = self.X[split_2:, :, :]
        self.test_label = self.Y[split_2:, :]

    def get_dataset_log(self):
        if os.path.exists('./dataset_log/%s_info.log' % self.dataset):
            with open("./dataset_log/%s_info.log" % self.dataset, "w") as log_file:
                log_file.truncate()
        logger.add("./dataset_log/%s_info.log" % self.dataset, rotation="10 MB", format=custom_formatter)
        logger.info(f'This is {self.dataset} dataset, shape:{self.data.shape}')

        logger.info('-----train-----')
        logger.info(f'train feature shape:{self.train.shape}')
        logger.info(f'train label shape:{self.train_label.shape}')
        logger.info(f'train mean value: {np.mean(self.train, axis=(0, 1))}')
        logger.info(f'train std value: {np.std(self.train, axis=(0, 1))}')
        min_train_val = [f'{val:.2f}' for val in np.min(self.train, axis=(0, 1))]
        logger.info(f'test min value: {min_train_val}')
        logger.info(f'train max value: {np.max(self.train, axis=(0, 1))}')

        logger.info('-----val-----')
        logger.info(f'val feature shape:{self.val.shape}')
        logger.info(f'val label shape:{self.val_label.shape}')
        logger.info(f'val mean value: {np.mean(self.val, axis=(0, 1))}')
        logger.info(f'val std value: {np.std(self.val, axis=(0, 1))}')
        min_val_val = [f'{val:.2f}' for val in np.min(self.val, axis=(0, 1))]
        logger.info(f'test min value: {min_val_val}')
        logger.info(f'val max value: {np.max(self.val, axis=(0, 1))}')

        logger.info('-----test-----')
        logger.info(f'test feature shape:{self.test.shape}')
        logger.info(f'test label shape:{self.test_label.shape}')
        logger.info(f'test mean value: {np.mean(self.test, axis=(0, 1))}')
        logger.info(f'test std value: {np.std(self.test, axis=(0, 1))}')
        min_test_val = [f'{val:.2f}' for val in np.min(self.test, axis=(0, 1))]
        logger.info(f'test min value: {min_test_val}')
        logger.info(f'test max value: {np.max(self.test, axis=(0, 1))}')

        logger.remove()

def get_data_loader(dataset, batch_size, num_workers, mode):
    y_scaler = dataset.y_scaler
    if mode == 'train':
        shuffle = True
    else:
        shuffle = False
    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=num_workers)
    return data_loader, y_scaler