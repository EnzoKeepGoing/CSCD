# pycd/utils/config.py

import yaml
import json

def load_config(path: str):
    """
    读取 YAML 或 JSON 格式的配置文件，返回一个 dict。
    支持 .yaml/.yml/.json 后缀。
    """
    if path.endswith(('.yaml', '.yml')):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    elif path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {path}")

