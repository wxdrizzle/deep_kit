from rich.traceback import install

install()

import torch
import time
from .cfgs.collect_cfg import cfg
from .experimenters.trainer import Trainer


def do_exp():
    trainer = Trainer(cfg)
    if cfg.exp.mode == 'train':
        trainer.train()
    elif cfg.exp.mode == 'test':
        trainer.test()
    else:
        raise ValueError(f'cfg.exp.mode={cfg.exp.mode} is not supported. Please choose from [train, test]')


# if cfg.exp.compile_model:
#     print('------------- Compiling -------------')
#     start = time.time()
#     do_exp = torch.compile(do_exp)
#     print(f'------------- Compilation finished, took {time.time() - start:.2f} s -------------')
