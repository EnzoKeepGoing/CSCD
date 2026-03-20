# pycd/utils/logging.py
import datetime
import logging
import sys, os
def init_logger(level: int = logging.INFO, log_file: str = None):
    """
    初始化全局日志：
      - level: 日志级别，默认为 INFO
      - log_file: 如果不为 None，则把日志同时写入该文件
    """
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )

def get_experiment_dir(base_dir, model_name, dataset_name, params=None, seed=None):
    """
    生成实验目录路径 - 通用版本，自动排序参数，fold放最后
    
    格式: {model_name}_{dataset_name}_{sorted_param_values}_{fold}_{seed}_{timestamp}
    
    Args:
        base_dir: 基础目录
        model_name: 模型名称
        dataset_name: 数据集名称
        params: 字典，包含所有需要记录的参数
        seed: 随机种子
    
    Returns:
        str: 实验目录路径
    """
    import datetime
    
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 基础目录名
    dirname = f"{model_name}_{dataset_name}"
    
    if params:
        # 跳过这些标准参数，不放在路径中
        skip_params = {'device', 'save_dir', 'data_dir', 'base_dir'}
        
        # 收集所有非跳过的参数
        values = []
        fold_value = None
        
        # 按字母顺序排序参数（除了fold）
        sorted_params = sorted([k for k in params.keys() if k not in skip_params])
        
        for param_name in sorted_params:
            v = params[param_name]
            
            if isinstance(v, (tuple, list)):
                # 元组或列表用-连接
                values.append('-'.join(map(str, v)))
            elif isinstance(v, float):
                # 浮点数格式化
                if v < 0.001:
                    values.append(f"{v:.0e}")  # 科学计数法
                else:
                    values.append(str(v))
            else:
                values.append(str(v))
        
        # 处理fold参数（放最后）
        if 'fold' in params:
            fold_value = params['fold']
        
        # 添加参数值到目录名
        if values:
            dirname += "_" + "_".join(values)
        
        # 添加fold（如果存在）
        if fold_value is not None:
            dirname += f"_{fold_value}"
    
    # 添加种子
    if seed is not None:
        dirname += f"_{seed}"
    
    # 添加时间戳
    dirname += f"_{timestamp}"

    # 创建完整的实验目录路径
    exp_dir = os.path.join(base_dir, dirname)
    os.makedirs(exp_dir, exist_ok=True)
    
    return exp_dir
    
def save_experiment_config(exp_dir, model_name, dataset_name, model_params):
    """
    保存实验配置到文件
    
    Args:
        exp_dir: 实验目录路径
        model_name: 模型名称
        dataset_name: 数据集名称
        model_params: 模型参数字典
    """
    config_path = os.path.join(exp_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("Experiment Configuration:\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Dataset: {dataset_name}\n")
        for k, v in model_params.items():
            f.write(f"{k}: {v}\n")
