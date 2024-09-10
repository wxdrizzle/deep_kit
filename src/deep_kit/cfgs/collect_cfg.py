from clearml import Task, Dataset
import os
from omegaconf import OmegaConf as Ocfg
import importlib.resources
from .. import cfgs
from .. import utils

Ocfg.register_new_resolver('len', lambda x: len(x))

cfg_base = Ocfg.create(importlib.resources.read_text(cfgs, 'base.yml'))
cfg_default_exp = Ocfg.load('cfgs/default/experiment.yml')
cfg_cli = Ocfg.from_cli()

if hasattr(cfg_cli, 'exp'): # run manually in command line by python xxx.py exp.name=xxx
    assert hasattr(cfg_cli.exp, 'name')
    prefix = cfg_cli.exp.name[:2]
    cfg_cli.exp.name = cfg_cli.exp.name[2:]
    if prefix == 'tr':
        cfg_cli.exp.mode = 'train'
        if os.path.exists(f'cfgs/train/{cfg_cli.exp.name}.yml'):
            cfg_exp = Ocfg.load(f'cfgs/train/{cfg_cli.exp.name}.yml')
        else:
            cfg_exp = Ocfg.load(f'cfgs/{cfg_cli.exp.name}.yml')
    elif prefix == 'te':
        cfg_cli.exp.mode = 'test'
        cfg_exp = Ocfg.load(f'cfgs/test/{cfg_cli.exp.name}.yml')
    else:
        raise ValueError
    cfg_exp.exp.name = cfg_cli.exp.name
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

    assert cfg.exp.mode in ['train', 'test']
    task_type = f'{cfg.exp.mode}ing'
    task = Task.init(project_name=cfg._meta.project, task_name=cfg.exp.name.split('/')[-1], task_type=task_type)

    dict_cfg = utils.flatten_dict(Ocfg.to_container(cfg), ignore_keys=['var', '_meta'])
    dict_cfg = {k.replace('.', '/', 1): v for k, v in dict_cfg.items()} # change first '.' in keys to '/'
    task.set_parameters(dict_cfg)

    if cfg._meta.path_conda_env is not None:
        task.set_base_docker(docker_image=cfg._meta.path_conda_env)

    tasks_pre = Task.get_tasks(project_name=cfg._meta.project, task_name=cfg.exp.name.split('/')[-1],
                               task_filter={'type': [task_type]})
    cfg.exp.name = cfg.exp.name.replace('/', '-')
    Ocfg.set_readonly(cfg, True)
    Ocfg.set_readonly(cfg.var, False)
    task.set_user_properties({'name': 'Working Dir', 'value': os.getcwd()})
else: # run by a clearML agent with new hyper-parameters
    task = Task.init()
    dict_cfg = task.get_parameters(cast=True)

    name_task = task.name.replace('/', '-')
    task.set_name(name_task)

    if task.name.startswith(dict_cfg['dataset/name']):
        dict_cfg['exp/name'] = task.name
    else:
        dict_cfg['exp/name'] = f'{dict_cfg["dataset/name"]}-{task.name}'
    idx_device = dict_cfg['exp/idx_device']
    if isinstance(idx_device, list):
        idx_device = list(range(len(idx_device)))
    else:
        idx_device = 0
    dict_cfg['exp/idx_device'] = idx_device
    task.set_parameters(dict_cfg)

    dict_cfg = {k.replace('/', '.', 1): v for k, v in dict_cfg.items()}
    dict_cfg = utils.unflatten_dict(dict_cfg)
    cfg = Ocfg.create(dict_cfg)

    # def make_cfg_type_correct(cfg):
    #     import omegaconf
    #     if not isinstance(cfg, omegaconf.dictconfig.DictConfig):
    #         return cfg
    #     Ocfg.set_readonly(cfg, False)
    #     for key in cfg:
    #         if key == 'var':
    #             continue
    #         elif isinstance(cfg[key], omegaconf.dictconfig.DictConfig):
    #             cfg[key] = make_cfg_type_correct(cfg[key])
    #         elif isinstance(cfg[key], str):
    #             try:
    #                 cfg[key] = eval(cfg[key])
    #             except:
    #                 pass
    #     return cfg
    #
    # cfg = make_cfg_type_correct(cfg)

    Ocfg.set_readonly(cfg, False)
    cfg.var = Ocfg.create(flags={"allow_objects": True})
    Ocfg.set_readonly(cfg, True)
    Ocfg.set_readonly(cfg.var, False)

    assert cfg.exp.mode in ['train', 'test']
    task_type = f'{cfg.exp.mode}ing'
    tasks_pre = Task.get_tasks(project_name=task.get_project_name(), task_name=task.name,
                               task_filter={'type': [task_type]})

# otherwise clearML task agent will set up a new environment and install all packages
# rather than using the existing environment
task.set_packages([])
task.upload_artifact('code', os.path.join(os.getcwd(), 'core', '*.py'))
# for task_pre in tasks_pre:
#     if task_pre.id != task.id and task_pre.get_parameters(cast=True)['dataset/name'] == cfg.dataset.name:
#         task_pre.delete(raise_on_error=True)
