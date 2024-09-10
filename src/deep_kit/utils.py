import importlib
import logging
from rich.logging import RichHandler
from clearml import Task
import os
import shutil


def setup_logger(name, *paths_files):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # rich handler
    handler = RichHandler(log_time_format='[%Y-%m-%d %H:%M:%S]', show_level=False, show_path=False)
    handler.setLevel(logging.WARNING)
    logger.addHandler(handler)

    # to files
    formatter = logging.Formatter(datefmt='[%Y-%m-%d %H:%M:%S]', fmt='%(asctime)s %(message)s')
    for path in paths_files:
        handler = logging.FileHandler(path)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def find_class(name: str, module: str, prefix: str = 'core'):
    """find the file using specified name and mode.

    Args:
        name: case-insensitive, e.g., 'BrainWeb'
        mode: find which type of file, e.g., 'model', 'dataset', etc.
    Returns:
        cls: class in a corresponding file, e.g., BrainWebDataset() in datasets/brainweb_dataset.py
    """
    name = name.lower()
    name_file = f'{prefix}.{module}s.{name}'
    path_file = f'{prefix}/{module}s/{name}.py'
    lib = importlib.import_module(name_file)
    name_cls = name.replace('_', '')

    cls = None
    for key, value in lib.__dict__.items():
        if key.lower() == name_cls:
            cls = value

    if cls is None:
        raise NotImplementedError(f"class '{name_cls}' not found in {path_file}")

    return cls, path_file


def flatten_dict(nested_dict, parent_key='', sep='.', ignore_keys=None):
    flat_dict = {}
    for key, value in nested_dict.items():
        if ignore_keys is not None and key in ignore_keys:
            continue
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat_dict.update(flatten_dict(value, new_key))
        else:
            flat_dict[new_key] = value
    return flat_dict


def unflatten_dict(flat_dict, sep='.'):
    nested_dict = {}
    for key, value in flat_dict.items():
        keys = key.split(sep)
        d = nested_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        if value == '':
            value = None
        d[keys[-1]] = value
    return nested_dict


def clean_tasks(project_name):
    tasks = Task.get_tasks(project_name=project_name)
    for task in tasks:
        if task.get_archived():
            try:
                path_exp = task.get_user_properties()['Experiment Path']['value']
                if os.path.exists(path_exp):
                    shutil.rmtree(path_exp)
                name_exp = os.path.basename(path_exp)
                path_folder = os.path.dirname(path_exp)
                path_tensorboard = os.path.join(path_folder, 'runs', name_exp)
                if os.path.exists(path_tensorboard):
                    shutil.rmtree(path_tensorboard)
            except Exception as e:
                print(e)
            try:
                task.delete(raise_on_error=True)
            except Exception as e:
                print(e)


def run_test_for_tasks(task_ids, commit_id=None, diff=None, dict_params_override=None):
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        path_exp = task.get_user_properties()['Experiment Path']['value']
        path_model_best_val = os.path.join(path_exp, 'checkpoints', 'model_best_val.pth')
        name_server = task.data.runtime['hostname']

        task_new = Task.clone(source_task=task, name=task.name, parent=task.id)
        task_new.set_task_type('testing')
        task_new.set_parameter('exp/test.path_model_trained', path_model_best_val)
        task_new.set_parameter('exp/mode', 'test')
        if dict_params_override is not None:
            for key, value in dict_params_override.items():
                task_new.set_parameter(key, value)

        task_new.set_script(commit=commit_id, entry_point='main.py', diff=diff)
        Task.enqueue(task_new, queue_name=f'test_on_{name_server}_one_gpu')
