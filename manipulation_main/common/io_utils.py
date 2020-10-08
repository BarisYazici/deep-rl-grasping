import os
import json
import yaml


def load_json(file_path):
    """Load a JSON file into a Python dict.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        A dict with the loaded configuration.
    """
    with open(os.path.expanduser(file_path), 'r') as f:
        config = json.load(f)
    return config


def save_json(config, file_path):
    """Save a Python dict to a JSON file.

    Args:
        config (dict): The dict to be saved.
        file_path (str): The path to the JSON file.
    """
    with open(os.path.expanduser(file_path), 'w') as f:
        json.dump(config, f, indent=2)


def load_yaml(file_path):
    """Load a YAML file into a Python dict.

    Args:
        file_path (str): The path to the YAML file.

    Returns:
        A dict with the loaded configuration.
    """
    with open(os.path.expanduser(file_path), 'r') as f:
        config = yaml.load(f)
    return config


def save_yaml(config, file_path):
    """Save a dict to a YAML file.

    Args:
        config (dict): The dict to be saved.
        file_path (str): The path to the YAML file.
    """
    with open(os.path.expanduser(file_path), 'w') as f:
        yaml.dump(config, f, default_flow_style=None)
