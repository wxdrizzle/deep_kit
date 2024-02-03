import importlib
import logging
from rich.logging import RichHandler


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
