#!/usr/bin/env python
"""
文件名称：run_training.py
文件功能：项目训练启动脚本，提供命令行参数支持。
创建日期：2025-11-19
版本：v1.0
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train_kfold import run_kfold
from src.config import get_data_config, get_training_config


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='前列腺癌 3D U-Net 分割训练脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认配置训练
  python run_training.py
  
  # 指定配置文件
  python run_training.py --config my_config.yml
  
  # 快速测试（只训练 1 折，5 个 epoch）
  python run_training.py --quick-test
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='配置文件路径 (默认: config.yml)'
    )

    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='快速测试模式：只训练 1 折，5 个 epoch'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='训练设备 (默认: auto - 自动检测)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='从检查点恢复训练（指定模型路径）'
    )

    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()

    # 加载配置
    from src.config import load_config
    load_config(args.config)

    data_config = get_data_config()
    training_config = get_training_config()

    # 快速测试模式
    if args.quick_test:
        print("\n⚠️  快速测试模式已启用")
        print("  - 只训练 1 折")
        print("  - 最大 epoch: 5")
        print("  - 早停耐心: 3\n")
        training_config['num_folds'] = 1
        training_config['num_epochs'] = 5
        training_config['patience'] = 3

    # 设备配置
    if args.device != 'auto':
        import torch
        if args.device == 'cuda' and not torch.cuda.is_available():
            print("⚠️  CUDA 不可用，将使用 CPU")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '0' if args.device == 'cuda' else ''

    # 显示配置信息
    print("\n" + "=" * 60)
    print("训练配置")
    print("=" * 60)
    print(f"配置文件: {args.config}")
    print(f"数据路径: {data_config['data_root']}")
    print(f"输出路径: {data_config['output_dir']}")
    print(f"模态: {data_config['modalities']}")
    print(f"批次大小: {training_config['batch_size']}")
    print(f"学习率: {training_config['learning_rate']}")
    print(f"折数: {training_config['num_folds']}")
    print("=" * 60 + "\n")

    # 确认开始训练
    if not args.quick_test:
        response = input("确认开始训练？(y/n): ")
        if response.lower() != 'y':
            print("训练已取消")
            return

    # 开始训练
    try:
        run_kfold()
        print("\n✓ 训练成功完成！")
        print(f"  模型保存在: {os.path.join(data_config['output_dir'], 'models')}")
        print(f"  日志保存在: {os.path.join(data_config['output_dir'], 'logs')}")
        print(f"  曲线保存在: {os.path.join(data_config['output_dir'], 'plots')}")

    except KeyboardInterrupt:
        print("\n\n⚠️  训练被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
