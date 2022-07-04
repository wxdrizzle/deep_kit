from rich.traceback import install

from .cfgs.collect_cfg import cfg
install()


def train():
    from .experimenters.trainer import Trainer
    trainer = Trainer(cfg)
    trainer.train()


def test():
    from .experimenters.tester import Tester
    tester = Tester(cfg)
    tester.test()
