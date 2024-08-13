import yaml
import json
from sklearn.metrics import r2_score
import numpy as np
from loguru import logger
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
import random
from tabulate import tabulate
from utils.cal_adj import *
import pickle


from loguru import logger


# 自定义格式化程序
def custom_formatter(record):
    # 将数字列表转换为保留两位小数的字符串
    if isinstance(record["message"], list):
        numbers = ", ".join([f"{num:.2f}" for num in record["message"]])
        new_message = f"[{numbers}]"
    else:
        new_message = record["message"]

    return "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"

def save_dict_to_yaml(dict_value: dict, save_path: str):
    """dict保存为yaml"""
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(yaml.dump(dict_value, allow_unicode=True))
def save_dict_to_json(dict_value: dict, save_path: str):
    with open(save_path, 'w', encoding='utf-8') as file:
        json.dump(dict_value, file, ensure_ascii=False)

def read_yaml_to_dict(yaml_path: str):
    with open(yaml_path, 'r') as file:
        dict_value = yaml.safe_load(file)
        return dict_value


def mse(test_y,y_pred):
    error = test_y-y_pred
    return np.mean(np.array(error)**2)
def rmse(test_y,y_pred):
    error = test_y-y_pred
    return np.sqrt(np.mean(np.array(error)**2))
def mae(test_y,y_pred):
    error = np.abs(np.array(test_y-y_pred))
    return np.mean(error)
def mape(test_y,y_pred):
    error = np.abs(np.array((test_y-y_pred)/(test_y+1e-6)))
    return np.mean(error)
def r_2(test_y,y_pred):
    return r2_score(test_y,y_pred)
def get_all_result(test_y, y_pred, multiple=False):
    test_y = np.nan_to_num(test_y)
    y_pred = np.nan_to_num(y_pred)
    mse_day = mse(test_y,y_pred)
    rmse_day = rmse(test_y,y_pred)
    mae_day = mae(test_y,y_pred)
    mape_day = mape(test_y,y_pred)
    r2_day = r_2(test_y,y_pred)
    if multiple:
        # print(f'mse:{mse_day}, rmse:{rmse_day}, mae:{mae_day},mape:{mape_day}
        return mse_day, rmse_day, mae_day, mape_day, None
    else:
        # print(f'mse:{mse_day}, rmse:{rmse_day}, mae:{mae_day},mape:{mape_day},r2:{r2_day}')
        return mse_day, rmse_day, mae_day, mape_day, r2_day

def re_normalization(x, _mean, _std, _min, _max, scale_type='standard'):
    if scale_type == 'standard':
        x = x*_std + _mean
        return x
    else:
        x = x * (_max-_min)+_min
        return x


def normalize(default='MinMaxScaler'):
    if default == "StandardScaler":
        return StandardScaler()
    return MinMaxScaler(feature_range=(0, 1))

def get_cfg(cfg_data_file, cfg_model_file, cfg_default_file):
    cfg_data = read_yaml_to_dict(cfg_data_file)
    cfg_model = read_yaml_to_dict(cfg_model_file)
    cfg_default = read_yaml_to_dict(cfg_default_file)
    cfg = {**cfg_data, **cfg_model, **cfg_default}
    return cfg_data, cfg_model, cfg_default, cfg

def setup_seed(seed):
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def print_args_model_info(args, model, print_model=False):
    params = vars(args)
    # 使用tabulate函数将参数表格化
    params_table = tabulate(params.items(), headers=["Parameter", "Value"], tablefmt="grid")
    # 使用Loguru的logger打印参数表格
    logger.info("\n" + params_table)
    # 获取模型的层结构
    model_architecture = []
    for name, module in model.named_children():
        model_architecture.append((name, module))
    # 使用tabulate函数将模型架构和参数表格化
    architecture_table = tabulate(model_architecture, headers=["Layer Name", "Layer"], tablefmt="grid")
    if print_model:
        logger.info("\n" + "Model Architecture:\n" + architecture_table)




class EarlyStopping:
    '''
    根据每一轮的损失，以及是否触发early_stop状态，来进行早停
    基本原理是。连续多个epoch的验证损失不再提升即可触发
    '''
    def __init__(self, patience, verbose=False, delta=0):
        self.verbose = verbose   # 触发保存模型机制,打印最小损失
        self.patience = patience
        self.delta = delta
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.counter = 0
        self.best_score = None

    def __call__(self, val_loss, model, path):
        '''
        __call__:实例对象被调用，可以直接输入参数然后当作函数被使用
        score来表示损失的相反数，理论上；来讲如果score越来越小，没再增加，那么可能触发早停机制
        '''
        score = -val_loss
        if self.best_score is None: # 最开始的时候
            self.best_score = score
            # 先保存一个pth
            self.save_checkpoint(val_loss, model, path)
        elif score <= self.best_score + self.delta: # 分数下降或者一直不上升
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score  # 分数上涨了,best pth需要更新
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logger.info(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

# For D2STGNN-->LOAD WEIGHT
def load_pickle(pickle_file):
    r"""
    Description:
    -----------
    Load pickle data.

    Parameters:
    -----------
    pickle_file: str
        File path.

    Returns:
    -----------
    pickle_data: any
        Pickle data.
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(adj_mx, adj_type):
    r"""
    Description:
    -----------
    Load adjacent matrix and preprocessed it.

    Parameters:
    -----------
    file_path: str
        Adjacent matrix file path (pickle file).
    adj_type: str
        How to preprocess adj matrix.

    Returns:
    -----------
        adj_matrix
    """
    # try:
    #     # METR and PEMS_BAY
    #     sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(file_path)
    # except:
    #     # PEMS04
    #     # adj_mx = load_pickle(file_path)
    # adj_mx = WeightProcess(root_path, num_nodes, dataset).s_w
    if adj_type == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "normlap":
        adj = [calculate_symmetric_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adj_type == "symnadj":
        adj = [symmetric_message_passing_adj(adj_mx).astype(np.float32).todense()]
    elif adj_type == "transition":
        adj = [transition_matrix(adj_mx).T]
    elif adj_type == "doubletransition":
        adj = [np.nan_to_num(transition_matrix(adj_mx)).T, np.nan_to_num(transition_matrix(adj_mx.T)).T]
    elif adj_type == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32).todense()]
    elif adj_type == 'original':
        adj = adj_mx
    else:
        error = 0
        assert error, "adj type not defined"
    return adj, adj_mx



if __name__ == '__main__':
    dict = read_yaml_to_dict('../cfg/models/agcrn.yaml')
    print(dict)
    print(dict['pems04']['input_dim'])