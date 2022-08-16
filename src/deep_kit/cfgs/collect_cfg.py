from omegaconf import OmegaConf as Ocfg
import importlib.resources
from .. import cfgs

Ocfg.register_new_resolver('len', lambda x: len(x))

cfg_base = Ocfg.create(importlib.resources.read_text(cfgs, 'base.yml'))
cfg_default_exp = Ocfg.load('cfgs/default/experiment.yml')
cfg_cli = Ocfg.from_cli()

prefix = cfg_cli.exp.name[:2]
cfg_cli.exp.name = cfg_cli.exp.name[2:]

if prefix == 'tr':
    cfg_exp = Ocfg.load(f'cfgs/{cfg_cli.exp.name}.yml')
elif prefix == 'te':
    cfg_exp = Ocfg.load(f'cfgs/test/{cfg_cli.exp.name}.yml')
else:
    raise ValueError
cfg_default_model = Ocfg.load(f'cfgs/default/models/{cfg_exp.model.name}.yml')
cfg_default_dataset = Ocfg.load(f'cfgs/default/datasets/{cfg_exp.dataset.name}.yml')

cfg = Ocfg.unsafe_merge(cfg_base, cfg_default_exp, cfg_default_model, cfg_default_dataset, cfg_exp, cfg_cli)
cfg.var = Ocfg.create(flags={"allow_objects": True})
Ocfg.resolve(cfg)

cfg.exp.train.optimizer = Ocfg.masked_copy(cfg.exp.train.optimizer, ['name', 'lr', cfg.exp.train.optimizer['name']])
keys_masked_sch = ['name']
if cfg.exp.train.scheduler.name is not None:
    keys_masked_sch.append(cfg.exp.train.scheduler.name)
cfg.exp.train.scheduler = Ocfg.masked_copy(cfg.exp.train.scheduler, keys_masked_sch)

Ocfg.set_readonly(cfg, True)
Ocfg.set_readonly(cfg.var, False)
