from clearml import Task

task = Task.current_task()
import os
import datetime
import shutil
import random
from copy import deepcopy
import numpy as np
from omegaconf import OmegaConf as Ocfg
import torch.distributed as dist

import torch
from torch.utils.tensorboard import SummaryWriter


class Operator:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.var.obj_operator = self

        if (not isinstance(idx := cfg.exp.idx_device, int)) and len(list(idx)) > 1:
            cfg.var.is_parallel = True
            if os.name == 'nt': # if system is windows
                dist.init_process_group(backend='gloo')
            else:
                dist.init_process_group(backend='nccl')
            # need to add this line to avoid the following bug:
            # https://discuss.pytorch.org/t/distributeddataparallel-gru-module-gets-additional-processes-on-gpu-0-1st-gpu-and-takes-more-memory/140225
            # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
            torch.cuda.set_device(cfg.exp.idx_device[dist.get_rank()])
            torch.cuda.empty_cache()
        else:
            cfg.var.is_parallel = False

    def _init_seed(self):
        if self.cfg.exp.rand_seed is None:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
        else:
            seed = int(self.cfg.exp.rand_seed)
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.enabled = False

    def _init_device(self):
        idx = self.cfg.exp.idx_device

        if self.cfg.var.is_parallel:
            name_device = f'cuda:{idx[dist.get_rank()]}'
        elif idx >= 0:
            name_device = f'cuda:{idx}'
        elif idx == -1:
            name_device = 'cpu'
        else:
            raise ValueError(f"Unknown device index: {idx}")
        self.device = torch.device(name_device)

    def _init_dirs(self):
        self.time_exp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.name_exp = f'{self.cfg.exp.mode}_{self.cfg.exp.name}_{self.time_exp}'
        if self.cfg.exp.path_save.startswith('.'):
            task = Task.current_task()
            path_dir_working = task.get_user_properties()['Working Dir']['value']
            self.path_exp = os.path.join(path_dir_working, self.cfg.exp.path_save, self.name_exp)
        else:
            self.path_exp = os.path.join(self.cfg.exp.path_save, self.name_exp)
        self.path_exp = os.path.realpath(self.path_exp)
        task.set_user_properties({'name': 'Experiment Path', 'value': self.path_exp})
        if self.cfg.exp.mode == 'train':
            self.path_checkpoints = os.path.join(self.path_exp, 'checkpoints')
            if not os.path.exists(self.path_checkpoints):
                os.makedirs(self.path_checkpoints)
        self.path_vis = os.path.join(self.cfg.exp.path_save, 'runs', self.name_exp)
        if not os.path.exists(self.path_vis):
            os.makedirs(self.path_vis)
        self.path_log = self.path_exp
        if not os.path.exists(self.path_log):
            os.makedirs(self.path_log)

        # if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
        #     path_save = os.path.dirname(self.path_exp)
        #     dirs_exp = os.listdir(path_save)
        #     for dir in dirs_exp:
        #         delete = False
        #         if 'tmp' in dir:
        #             delete = True
        #         dir_wo_time = '_'.join(dir.split('_')[:-2])
        #         if dir_wo_time == f'{self.cfg.exp.mode}_{self.cfg.exp.name}':
        #             delete = True
        #         if self.cfg.exp.names_exp_delete is not None and dir_wo_time in self.cfg.exp.names_exp_delete:
        #             delete = True
        #         if dir == self.name_exp:
        #             delete = False
        #         if delete:
        #             shutil.rmtree(os.path.join(path_save, dir))
        #             if os.path.exists(os.path.join(path_save, 'runs', dir)):
        #                 shutil.rmtree(os.path.join(path_save, 'runs', dir))

    def _init_writer(self):
        if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
            self.writer = SummaryWriter(self.path_vis)

    def _init_log_basic_info(self):
        if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
            self.logger_extra.warn(f'Experiment saved in {self.path_exp}')
            cfg_printed = deepcopy(self.cfg)
            Ocfg.set_readonly(cfg_printed, False)
            del cfg_printed.var
            self.logger_extra.warn(Ocfg.to_yaml(cfg_printed))
            Ocfg.save(config=cfg_printed, f=os.path.join(self.path_log, 'configs.yml'))
