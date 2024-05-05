import os
import logging
import yaml
from torch import distributed as dist
from easydict import EasyDict
from mmdet.datasets import (build_dataloader, build_dataset)
logger = logging.getLogger('ptq4sam')

import numpy as np
import torch
import copy

def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        cur_config = config
        cur_path = config_file
        while 'root' in cur_config:
            root_path = os.path.dirname(cur_path)
            cur_path = os.path.join(root_path, cur_config['root'])
            with open(cur_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                cur_config = root_config
    config = EasyDict(config)
    return config

# hook function
class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward
        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch
        if self.store_output:
            self.output_store = output_batch
        if self.stop_forward:
            raise StopForwardException


# load data
def load_data(cfg, distributed, num_samples):
    train_dataloader_default_args = dict(
        samples_per_gpu=2,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=cfg.seed,  
    )
    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    test_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    cfg.data.test.test_mode = True
    # build the dataloader
    test_data = build_dataset(cfg.data.test)
    test_loader = build_dataloader(test_data, **test_loader_cfg)

    train_data = build_dataset(cfg.data.train)
    train_loader = build_dataloader(train_data, **train_loader_cfg)
    num_samples_per_gpu = num_samples // len(cfg.gpu_ids)
    cali_data = []
    bs = 2
    for i, data_batch in enumerate(train_loader):
        cali_data.append(data_batch['img'].data[0])
        if len(cali_data) * bs == num_samples_per_gpu:
            break
    rank = dist.get_rank()
    logger.info('the length of cali data is {}, the rank is {}'.format(cali_data[0].flatten()[0], rank))
    return test_loader, test_data, cali_data

def load_calibration(cfg, distributed, num_samples):
    train_dataloader_default_args = dict(
        samples_per_gpu=1,
        workers_per_gpu=2,
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        seed=6769,
        shuffle=False
    )
    train_loader_cfg = {
        **train_dataloader_default_args,
        **cfg.data.get('train_dataloader', {})
    }
    train_loader_cfg['samples_per_gpu'] = 1
    # import pdb;pdb.set_trace()
    train_data = build_dataset(cfg.data.train)
    # train_data_c = train_data
    train_loader = build_dataloader(train_data, **train_loader_cfg)
    
    num_samples_per_gpu = num_samples // len(cfg.gpu_ids)
    # np.random.seed(6)
    # inds=np.random.permutation(len(train_data_c))[:num_samples_per_gpu]
    # # calib_set=torch.utils.data.Subset(copy.deepcopy(train_dataset),inds)
    # sub_train_data = torch.utils.data.Subset(copy.deepcopy(train_data_c),inds)
    # sub_train_loader = build_dataloader(sub_train_data, **train_loader_cfg)
    cali_data = []
    # cali_data = sub_train_loader
    
    bs = 1
    for i, data_batch in enumerate(train_loader):
        t = {
            'img':[data_batch['img'][0].cuda()],
            'img_metas':data_batch['img_metas'][0].data,
        }
        # cali_data.append(data_batch['img'][0])
        cali_data.append(t)
        if len(cali_data) * bs == num_samples_per_gpu:
            break
    
    print('the length of cali data is {}.'.format(len(cali_data)))
    return cali_data
