from clearml import Task

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
if 'name' in cfg_exp.exp and cfg_exp.exp.name != cfg_cli.exp.name:
    raise ValueError(
        f'The yaml file name and the "exp.name" in the yaml must be the same, but got "{cfg_cli.exp.name}" and "{cfg_exp.exp.name}", respectively.'
    )
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

if cfg.exp.mode == 'train':
    task_type = 'training'
elif cfg.exp.mode == 'test':
    task_type = 'testing'
else:
    raise ValueError
task = Task.init(project_name=cfg._meta.project, task_name=cfg.exp.name, task_type=task_type)


def flatten_dict(nested_dict, parent_key='', sep='.'):
    flat_dict = {}
    for key, value in nested_dict.items():
        if key in ['var', '_meta']:
            continue
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, new_key))
        else:
            flat_dict[new_key] = value
    return flat_dict


dict_cfg = flatten_dict(Ocfg.to_container(cfg))
dict_cfg = {k.replace('.', '/', 1): v for k, v in dict_cfg.items()}
task.set_parameters(dict_cfg)
