"""
文件名称：config.py
文件功能：加载并提供全局配置。
"""
import os
import yaml
from typing import List, Tuple

# 全局配置字典
config = {}

def find_config_file(config_path: str = 'config.yml') -> str:
    """查找配置文件路径。
    
    如果当前目录找不到配置文件，则尝试在上级目录查找。
    
    参数：
        config_path (str): 配置文件名，默认为 'config.yml'
    返回：
        str: 配置文件的完整路径
    """
    # 首先检查当前目录
    if os.path.exists(config_path):
        return config_path
    
    # 如果当前目录找不到，尝试上级目录
    parent_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
    if os.path.exists(parent_config_path):
        return parent_config_path
    
    # 如果都找不到，返回默认路径（会抛出异常，但信息更明确）
    return config_path

def load_config(config_path: str = 'config.yml'):
    """从 YAML 文件加载配置并填充全局字典。
    
    参数：
        config_path (str): 配置文件路径
    """
    global config
    # 查找配置文件的实际路径
    actual_config_path = find_config_file(config_path)
    
    with open(actual_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 解析路径，相对于项目根目录
    data_config = config['data']
    if not os.path.isabs(data_config['data_root']):
        config['data']['data_root'] = os.path.join(project_root, data_config['data_root'].lstrip('./'))
    else:
        config['data']['data_root'] = os.path.abspath(data_config['data_root'])
        
    if not os.path.isabs(data_config['label_root']):
        config['data']['label_root'] = os.path.join(project_root, data_config['label_root'].lstrip('./'))
    else:
        config['data']['label_root'] = os.path.abspath(data_config['label_root'])
        
    if not os.path.isabs(data_config['output_dir']):
        config['data']['output_dir'] = os.path.join(project_root, data_config['output_dir'].lstrip('./'))
    else:
        config['data']['output_dir'] = os.path.abspath(data_config['output_dir'])

    # 创建输出目录
    config['data']['models_dir'] = os.path.join(config['data']['output_dir'], 'models')
    config['data']['logs_dir'] = os.path.join(config['data']['output_dir'], 'logs')
    config['data']['plots_dir'] = os.path.join(config['data']['output_dir'], 'plots')
    config['data']['preds_dir'] = os.path.join(config['data']['output_dir'], 'preds')

    os.makedirs(config['data']['models_dir'], exist_ok=True)
    os.makedirs(config['data']['logs_dir'], exist_ok=True)
    os.makedirs(config['data']['plots_dir'], exist_ok=True)
    os.makedirs(config['data']['preds_dir'], exist_ok=True)

    # OpenMP 修复
    if os.name == 'nt' and not os.environ.get('KMP_DUPLICATE_LIB_OK'):
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 模块加载时自动加载配置
load_config()

# 提供便捷访问
def get_data_config():
    return config.get('data', {})

def get_training_config():
    return config.get('training', {})