import os
import math
from rich.progress import track

import torch
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader

from .operator import Operator
from ..utils import setup_logger, find_class


class Trainer(Operator):
    def __init__(self, cfg):
        super().__init__(cfg)

        self._init_seed()
        self._init_dataloaders()
        self._init_device()
        cls_model, self.path_file_model = find_class(cfg.model.name, 'model')
        self.model = cls_model(cfg)
        self._init_dirs()
        self._init_loggers()
        self._init_writer()
        self._init_log_basic_info()

    def _init_dataloaders(self):
        cls_dataset, self.path_file_dataset = find_class(self.cfg.dataset.name, 'dataset')

        self.train_set = cls_dataset(mode='train', cfg=self.cfg)
        self.val_set = cls_dataset(mode='val', cfg=self.cfg)
        self.test_set = cls_dataset(mode='test', cfg=self.cfg)

        self.train_loader = DataLoader(
            dataset=self.train_set,
            batch_size=self.cfg.exp.train.batch_size,
            collate_fn=getattr(self.train_set, 'get_batch', None),
            num_workers=self.cfg.exp.n_workers,
            shuffle=not issubclass(type(self.train_set), IterableDataset),
        )
        self.val_loader = DataLoader(
            dataset=self.val_set,
            batch_size=self.cfg.exp.val.batch_size,
            collate_fn=getattr(self.val_set, 'get_batch', None),
            num_workers=self.cfg.exp.n_workers,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=self.cfg.exp.test.batch_size,
            collate_fn=getattr(self.test_set, 'get_batch', None),
            num_workers=self.cfg.exp.n_workers,
            shuffle=False,
        )

    def _init_loggers(self):
        path_all = os.path.join(self.path_log, 'log_all.txt')
        path_train = os.path.join(self.path_log, 'log_train.txt')
        path_val = os.path.join(self.path_log, 'log_val.txt')
        path_checkpoints = os.path.join(self.path_log, 'log_checkpoints.txt')
        self.logger_extra = setup_logger('extra', path_all)
        self.logger_train = setup_logger('train', path_all, path_train)
        self.logger_val = setup_logger('val', path_all, path_val)
        self.logger_checkpoints = setup_logger('checkpoints', path_all, path_checkpoints)

    def _get_optimizer(self, params):
        name_opt = self.cfg.exp.train.optimizer.name
        cfg_opt = self.cfg.exp.train.optimizer[name_opt]

        if name_opt == 'sgd':
            optimizer = optim.SGD(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
                momentum=cfg_opt.momentum,
                nesterov=cfg_opt.nesterov,
            )
        elif name_opt in ('adam', 'adamw'):
            optimizer = optim.Adam(
                params=params,
                lr=self.cfg.exp.train.optimizer.lr,
                weight_decay=cfg_opt.weight_decay,
            )
        else:
            raise NotImplementedError(f'Unknown optimizer: {name_opt}')

        name_sch = self.cfg.exp.train.scheduler.name
        if name_sch is not None:
            cfg_sch = self.cfg.exp.train.scheduler[name_sch]
            name_sch = name_sch.lower()
            if name_sch == 'cycliclr':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=cfg_sch.lr_base,
                                                        max_lr=cfg_sch.lr_max, mode=cfg_sch.mode, gamma=cfg_sch.gamma,
                                                        cycle_momentum=cfg_sch.cycle_momentum)
            elif name_sch == 'multisteplr':
                scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=cfg_sch.milestones,
                                                           gamma=cfg_sch.gamma)
            else:
                raise NotImplementedError(f'Unknown scheduler: {name_sch}')
        else:
            scheduler = None

        return optimizer, scheduler

    def train(self):
        self.logger_extra.warn(f'------ Training ------')
        self.model = self.model.to(self.device)
        optimizer, scheduler = self._get_optimizer(self.model.parameters())

        self.score_best = -math.inf
        iter_total = 0
        for epoch in range(self.cfg.exp.train.n_epochs):
            # validation
            if epoch % self.cfg.exp.val.n_epochs_once == 0:
                self.val(epoch, mode='val')
            if self.is_best:
                self.val(epoch, mode='test')

            self.model.train()
            self.model.before_epoch(mode='train')
            for _, data in enumerate(track(self.train_loader, transient=True, description='training')):
                iter_total += 1

                optimizer.zero_grad()
                if hasattr(self.train_set, 'to_device'):
                    data = self.train_set.to_device(data, device=self.device)
                else:
                    input, ground_truth = data
                    data = input.to(self.device), ground_truth.to(self.device)
                output = self.model(data)
                metrics = self.model.get_metrics(data, output, mode='train')
                metrics['loss_final'].backward()
                optimizer.step()

                for name, value in metrics.items():
                    self.writer.add_scalar(f'train/{name}', value, iter_total)

            self.model.after_epoch(mode='train')
            self.model.vis(self.writer, epoch, data, output, mode='train')

            if scheduler:
                scheduler.step()

            result_log = [f'epoch: {epoch}']
            for name, value in self.model.metrics_epoch.items():
                result_log.append(f'{name}: {value:.4f}')
            info_logged = ', '.join(result_log)
            self.logger_extra.warn(f'[train] {info_logged}')
            self.logger_train.info(info_logged)

        self.logger_train.warn(f'------ Training finished ------')

    def val(self, epoch, mode='val'):
        assert mode in ['val', 'test']
        data_loader = getattr(self, f'{mode}_loader')
        dataset = getattr(self, f'{mode}_set')

        with torch.no_grad():
            self.model.eval()
            self.model.before_epoch(mode)
            for i, data in enumerate(track(data_loader, transient=True, description=mode)):
                if hasattr(dataset, 'to_device'):
                    data = dataset.to_device(data, device=self.device)
                else:
                    input, ground_truth = data
                    data = input.to(self.device), ground_truth.to(self.device)
                output = self.model(data)
                _ = self.model.get_metrics(data, output, mode=mode)
            self.model.after_epoch(mode)

            if mode == 'val':
                self.is_best = self.model.metrics_epoch['metric_final'] > self.score_best
                if self.is_best:
                    self.score_best = self.model.metrics_epoch['metric_final']
            self.model.vis(self.writer, epoch, data, output, mode=mode)

            result_log = [f'epoch: {epoch}']
            for (name, value) in self.model.metrics_epoch.items():
                self.writer.add_scalar(f'{mode}/{name}', value, epoch)
                result_log.append(f'{name}: {value:.4f}')
            info_logged = ', '.join(result_log)
            self.logger_extra.warn(f'[{mode}] {info_logged}')
            self.logger_val.info(info_logged)

            # save best model
            if self.is_best and mode == 'val':
                self.logger_checkpoints.warn(f'Saving best model: epoch {epoch}')
                torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, 'model_best.pth'))
