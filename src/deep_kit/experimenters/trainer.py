import os
import math
from rich.progress import track

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .operator import Operator
from ..utils import setup_logger, find_class


class MyDistributedDataParallel(DistributedDataParallel):

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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

        if self.cfg.exp.mode == 'train':
            self.train_set = cls_dataset(mode='train', cfg=self.cfg)
            self.val_set = cls_dataset(mode='val', cfg=self.cfg)

            if self.cfg.var.is_parallel:
                self.sampler_train = DistributedSampler(self.train_set)
                shuffle_train = False
            else:
                shuffle_train = not issubclass(type(self.train_set), IterableDataset)

            self.train_loader = DataLoader(
                dataset=self.train_set,
                batch_size=self.cfg.exp.train.batch_size,
                collate_fn=getattr(self.train_set, 'get_batch', None),
                num_workers=self.cfg.exp.n_workers,
                shuffle=shuffle_train,
                pin_memory=True,
                drop_last=True,
                sampler=self.sampler_train if self.cfg.var.is_parallel else None,
            )
            self.val_loader = DataLoader(
                dataset=self.val_set,
                batch_size=self.cfg.exp.val.batch_size,
                collate_fn=getattr(self.val_set, 'get_batch', None),
                num_workers=self.cfg.exp.n_workers,
                shuffle=False,
                pin_memory=True,
            )
        elif self.cfg.exp.mode != 'test':
            raise ValueError

        self.test_set = cls_dataset(mode='test', cfg=self.cfg)
        self.test_loader = DataLoader(
            dataset=self.test_set,
            batch_size=self.cfg.exp.test.batch_size,
            collate_fn=getattr(self.test_set, 'get_batch', None),
            num_workers=self.cfg.exp.n_workers,
            shuffle=False,
            pin_memory=True,
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
            elif name_sch == 'exponentiallr':
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=cfg_sch.gamma)
            elif name_sch == 'lambdalr':
                lambdas_lr = []
                for where in cfg_sch.where: # e.g., model.lambda_lr_0
                    attrs = where.split('.')
                    lambda_lr = self
                    for attr in attrs:
                        lambda_lr = getattr(lambda_lr, attr)
                    lambdas_lr.append(lambda_lr)
                scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambdas_lr)
            else:
                raise NotImplementedError(f'Unknown scheduler: {name_sch}')
        else:
            scheduler = None

        return optimizer, scheduler

    def train(self):
        self.logger_extra.warn(f'------ Training ------')
        self.model = self.model.to(self.device)
        if self.cfg.var.is_parallel:
            id_device = self.cfg.exp.idx_device[dist.get_rank()]
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = MyDistributedDataParallel(self.model, device_ids=[id_device])

        optimizer, scheduler = self._get_optimizer(getattr(self.model, 'get_params', 'parameters')())

        if self.cfg.exp.train.path_model_trained is not None:
            if self.cfg.var.is_parallel:
                dist.barrier()
                id_device = self.cfg.exp.idx_device[dist.get_rank()]
                map_location = {'cuda:0': f'cuda:{id_device}'}
            else:
                map_location = self.device
            self.model.load_state_dict(torch.load(self.cfg.exp.train.path_model_trained, map_location=map_location),
                                       strict=False)
        self.score_best = -math.inf
        self.is_best = True
        iter_total = 0
        for epoch in range(self.cfg.exp.train.epoch_start, self.cfg.exp.train.n_epochs):
            # validation
            if self.cfg.exp.val.skip_initial_val and epoch == self.cfg.exp.train.epoch_start:
                skip_val = True
            elif epoch % self.cfg.exp.val.n_epochs_once != 0:
                skip_val = True
            else:
                skip_val = False
            if not skip_val:
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.val(epoch, mode='val')
                    if self.is_best:
                        self.val(epoch, mode='test')
            if self.cfg.var.is_parallel:
                dist.barrier()

            self.model.train()
            self.model.before_epoch(mode='train', i_repeat=epoch)

            if self.cfg.var.is_parallel:
                # see WARNING in https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
                self.sampler_train.set_epoch(epoch)

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
                    if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                        self.writer.add_scalar(f'train/{name}', value, iter_total)

            self.model.after_epoch(mode='train')
            if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                self.model.vis(self.writer, epoch, data, output, mode='train', in_epoch=False)

            if scheduler:
                scheduler.step()

            result_log = [f'epoch: {epoch}']
            for name, value in self.model.metrics_epoch.items():
                result_log.append(f'{name}: {value:.4f}')
            info_logged = ', '.join(result_log)
            self.logger_extra.warn(f'[train] {info_logged}')
            self.logger_train.info(info_logged)

        self.logger_train.warn(f'------ Training finished ------')

        if self.cfg.var.is_parallel:
            dist.destroy_process_group()

    def val(self, epoch, mode='val'):
        assert mode in ['val', 'test']
        data_loader = getattr(self, f'{mode}_loader')
        dataset = getattr(self, f'{mode}_set')

        with torch.no_grad():
            self.model.eval()

            for i_repeat in range(self.cfg.exp[mode].n_repeat):
                self.model.before_epoch(mode, i_repeat)
                for _, data in enumerate(track(data_loader, transient=True, description=mode)):
                    if hasattr(dataset, 'to_device'):
                        data = dataset.to_device(data, device=self.device)
                    else:
                        input, ground_truth = data
                        data = input.to(self.device), ground_truth.to(self.device)
                    if self.cfg.var.is_parallel and dist.get_rank() == 0:
                        # if we use self.model(data), then after validation, training will get stuck
                        # see https://github.com/pytorch/pytorch/issues/54059#issuecomment-801754630
                        output = self.model.module(data)
                    else:
                        output = self.model(data)
                    _ = self.model.get_metrics(data, output, mode=mode)
                    if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                        self.model.vis(self.writer, epoch, data, output, mode=mode, in_epoch=True)
                self.model.after_epoch(mode)

                if mode == 'val':
                    self.is_best = self.model.metrics_epoch['metric_final'] > self.score_best
                    if self.is_best:
                        self.score_best = self.model.metrics_epoch['metric_final']

                result_log = [f'epoch: {epoch}']
                for (name, value) in self.model.metrics_epoch.items():
                    result_log.append(f'{name}: {value:.4f}')
                info_logged = ', '.join(result_log)
                self.logger_extra.warn(f'[{mode}] {info_logged}')
                self.logger_val.info(info_logged)

            if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                self.model.vis(self.writer, epoch, data, output, mode=mode, in_epoch=False)

            for (name, value) in self.model.metrics_epoch.items():
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.writer.add_scalar(f'{mode}/{name}', value, epoch)

            # save best model
            if self.is_best and mode == 'val':
                if (not self.cfg.var.is_parallel) or dist.get_rank() == 0:
                    self.logger_checkpoints.warn(f'Saving best model: epoch {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, 'model_best.pth'))

            if mode == 'val':
                if getattr(self.cfg.exp.val, 'save_every_model', False):
                    self.logger_checkpoints.warn(f'Saving current model: epoch {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_epoch{epoch}.pth'))
                elif self.is_best and getattr(self.cfg.exp.val, 'save_every_better_model', False):
                    self.logger_checkpoints.warn(f'Saving current model: epoch {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, f'model_epoch{epoch}.pth'))
                if self.cfg.exp.val.save_latest_model:
                    self.logger_checkpoints.warn(f'Saving latest model: epoch {epoch}')
                    torch.save(self.model.state_dict(), os.path.join(self.path_checkpoints, 'model_latest.pth'))

    def test(self):
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load(self.cfg.exp.test.path_model_trained, map_location=self.device),
                                   strict=False)
        self.is_best = False
        self.val(epoch=0, mode='test')

        if self.cfg.var.is_parallel:
            dist.destroy_process_group()
