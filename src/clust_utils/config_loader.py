import yaml
import os
import re
import torch
import random
import numpy as np


def load_yaml_config(path: str) -> dict:
    """Loads a single YAML file and returns its contents as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def replace_placeholders(config: dict, variables: dict[str, int]) -> dict:
    """Replaces ${...} placeholders in config values using provided variables."""
    pattern = re.compile(r"\$\{(\w+)\}")
    replaced_config = {}

    for algo, params in config.items():
        new_params = {}
        for k, v in params.items():
            if isinstance(v, str):
                match = pattern.fullmatch(v)
                if match:
                    var_name = match.group(1)
                    if var_name not in variables:
                        raise ValueError(f"Variable '{var_name}' not provided for placeholder in key: {k}")
                    new_params[k] = variables[var_name]
                else:
                    new_params[k] = v  # string but not a placeholder
            else:
                new_params[k] = v  # non-string value
        replaced_config[algo] = new_params

    return replaced_config

def load_config(file_path: str, variables: dict[str, int]) -> dict:
    """Loads a config YAML file and replaces all placeholders using given variables."""
    raw_config = load_yaml_config(file_path)
    return replace_placeholders(raw_config, variables)

def load_all_configs(folder_path: str, variables: dict[str, int]) -> dict[str, dict]:
    """Loads and prepares all configs in a folder with given placeholder values."""
    configs = {}
    for fname in os.listdir(folder_path):
        if fname.endswith(".yaml"):
            name = fname.replace(".yaml", "")
            full_path = os.path.join(folder_path, fname)
            configs[name] = load_config(full_path, variables)
    return configs


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
