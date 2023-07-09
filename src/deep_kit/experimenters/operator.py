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
            seed = int(self.cfg.random_seed)
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
        def check_substrings():
            if isinstance(self.cfg.exp.names_exp_delete, str):
                names_exp_delete = [self.cfg.exp.names_exp_delete]
            else:
                names_exp_delete = self.cfg.exp.names_exp_delete

            for substring in names_exp_delete:
                if substring in dir:
                    return True
            return False

        self.time_exp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        self.name_exp = f'{self.cfg.exp.mode}_{self.cfg.exp.name}_{self.time_exp}'
        self.path_exp = os.path.join(self.cfg.exp.path_save, self.name_exp)
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
        self.path_backup = os.path.join(self.path_exp, 'backup')
        if not os.path.exists(self.path_backup):
            os.makedirs(self.path_backup)

        if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
            for path in self.model.paths_file_net + [self.path_file_model, self.path_file_dataset]:
                _ = shutil.copyfile(path, os.path.join(self.path_backup, os.path.basename(path)))

            if (self.cfg.exp.names_exp_delete is not None) and os.path.exists(self.cfg.exp.path_save):
                dirs_exp = os.listdir(self.cfg.exp.path_save)
                for dir in dirs_exp:
                    if dir != self.name_exp and check_substrings():
                        shutil.rmtree(os.path.join(self.cfg.exp.path_save, dir))
                        if os.path.exists(os.path.join(self.cfg.exp.path_save, 'runs', dir)):
                            shutil.rmtree(os.path.join(self.cfg.exp.path_save, 'runs', dir))

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
