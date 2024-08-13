import argparse
from core.args.base_args import BaseArgs

import os
from utils import read_yaml_to_dict, get_cfg, setup_seed, print_args_model_info
'''
1. 首先根据模型名字、数据名字-->获取对应的数据yaml、模型yaml、defaul yaml-->生成所有cfg-->公有的
2. 根据cfg自行添加额外变量
3. 生成对应的参数解析器
'''
class GSTRGCT_GenArgs(BaseArgs):
    def __init__(self, cfg):
        super(GSTRGCT_GenArgs, self).__init__(cfg)
        self.cfg = cfg
        self._gen_fix_args()
        self._gen_model_args()
        self.args = self.parser.parse_args()

    def _gen_model_args(self):
        # -----------------------------------------3. model----------------------------------------#
        # 公有的
        self.parser.add_argument('--in_dim', default=self.cfg['in_dim'], type=int,
                                 help="input dimension")
        self.parser.add_argument('--out_dim', default=self.cfg['out_dim'], type=int,
                                 help="output dimension")

        # spatial_plane
        self.parser.add_argument('--embed_dim', default=self.cfg['spatial_plane']['embed_dim'], type=int,
                                 help="embedding dimension of SRGCN")
        self.parser.add_argument('--g_lambda', default=self.cfg['spatial_plane']['g_lambda'], type=int,
                                 help="global or prior spatial weight ratio")
        self.parser.add_argument('--l_mu', default=self.cfg['spatial_plane']['l_mu'], type=float,
                                 help="local spatial/adaptive or posterior spatial weight ratio")
        self.parser.add_argument('--cheb_k', default=self.cfg['spatial_plane']['cheb_k'], type=int,
                                 help="number of terms in Chebyshev polynomials")
        self.parser.add_argument('--spatial_attention', default=self.cfg['spatial_plane']['spatial_attention'], type=bool,
                                 help="whether to output spatial attention")

        # temporal plane
        # 可调
        self.parser.add_argument('--d_model', default=self.cfg['temporal_plane']['d_model'], type=int,
                                 help="d_model of AutoTRT")
        self.parser.add_argument('--n_heads', default=self.cfg['temporal_plane']['n_heads'], type=int,
                                 help="head number of AutoTRT")
        self.parser.add_argument('--dropout', default=self.cfg['temporal_plane']['dropout'], type=float,
                                 help="dropout probability")
        self.parser.add_argument('--num_layers', default=self.cfg['temporal_plane']['num_layers'], type=int,
                                 help="layer number of AutoTRT")
        self.parser.add_argument('--factor', default=self.cfg['temporal_plane']['factor'], type=int,
                                 help="hyperparameter of Auto-correlation")
        # 没必要调
        self.parser.add_argument('--temporal_attention', default=self.cfg['temporal_plane']['temporal_attention'], type=bool,
                                 help="whether to output temporal attention/auto-correlation attention")
        self.parser.add_argument('--full_attention', default=self.cfg['temporal_plane']['full_attention'], type=bool,
                                 help="choose full attention")
        self.parser.add_argument('--activation', default=self.cfg['temporal_plane']['activation'], type=str,
                                 help="kind of activation function")

        # 网格搜参
        self.parser.add_argument('--grid_ed', default=self.cfg['spatial_plane']['grid_ed'], type=list,
                                 help="grid parameters of embed_dim")
        self.parser.add_argument('--grid_g_lambda', default=self.cfg['spatial_plane']['grid_g_lambda'], type=list,
                                 help="grid parameters of g_lambda")
        self.parser.add_argument('--grid_cheb_k', default=self.cfg['spatial_plane']['grid_cheb_k'], type=list,
                                 help="grid parameters of cheb_k")
        # 网格搜参
        self.parser.add_argument('--grid_factor', default=self.cfg['temporal_plane']['grid_factor'], type=list,
                                 help="grid parameters of factor")

if __name__ == '__main__':
    DATASET = 'pems08'
    DEVICE = 'cuda:0'
    MODEL = 'GSTRGCT'
    ROOT_PATH = '../../dataset/'

    # 读取基本的参数文件
    # cfg_files = ['cfg/datasets/%s.yaml' % DATASET, 'cfg/models/%s.yaml' % (MODEL.lower()), 'cfg/default.yaml']
    cfg_data_file = '../../cfg/datasets/%s.yaml' % DATASET
    cfg_model_file = '../../cfg/models/%s.yaml' % (MODEL.lower())
    cfg_default_file = '../../cfg/default.yaml'
    cfg_data, cfg_model, cfg_defalut, cfg_all = get_cfg(cfg_data_file, cfg_model_file, cfg_default_file)
    EXPERIMENT_NAME = '%s_%s_%s' % (MODEL, cfg_defalut['batch_size'], cfg_defalut['lr'])
    # EXPERIMENT_NAME = '%s_%s_%s_re1' % (MODEL, cfg_defalut['batch_size'], cfg_defalut['lr'])
    CHECK_POINT = os.path.join('logs', DATASET, EXPERIMENT_NAME, 'checkpoint.pth')
    cfg_all['DATASET'] = DATASET
    cfg_all['DEVICE'] = DEVICE
    cfg_all['MODEL'] = MODEL
    cfg_all['ROOT_PATH'] = ROOT_PATH
    cfg_all['EXPERIMENT_NAME'] = EXPERIMENT_NAME
    cfg_all['CHECK_POINT'] = CHECK_POINT
    gstrgct_args = GSTRGCT_GenArgs(cfg_all).args
    print(gstrgct_args.name)


















