#!/usr/bin/env python
"""
文件名称：run_evaluation.py
文件功能：模型评估启动脚本。
作者：TraeAI 助手
创建日期：2025-11-19
版本：v1.0
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import run_eval
from src.aggregate_results import main as aggregate_main
from src.config import get_data_config


def parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='前列腺癌 3D U-Net 分割评估脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 评估所有折
  python run_evaluation.py
  
  # 评估并汇总结果
  python run_evaluation.py --aggregate
  
  # 只汇总已有结果
  python run_evaluation.py --aggregate-only
        """
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yml',
        help='配置文件路径 (默认: config.yml)'
    )
    
    parser.add_argument(
        '--aggregate',
        action='store_true',
        help='评估后自动汇总结果'
    )
    
    parser.add_argument(
        '--aggregate-only',
        action='store_true',
        help='只汇总已有的评估结果，不重新评估'
    )
    
    parser.add_argument(
        '--fold',
        type=int,
        default=None,
        help='只评估指定的折（0-4）'
    )
    
    return parser.parse_args()


def main():
    """主函数。"""
    args = parse_args()
    
    # 加载配置
    from src.config import load_config
    load_config(args.config)
    
    data_config = get_data_config()
    
    print("\n" + "="*60)
    print("模型评估")
    print("="*60)
    print(f"配置文件: {args.config}")
    print(f"输出路径: {data_config['output_dir']}")
    print("="*60 + "\n")
    
    # 只汇总结果
    if args.aggregate_only:
        print("汇总评估结果...")
        aggregate_main()
        return
    
    # 运行评估
    try:
        if args.fold is not None:
            print(f"评估 Fold {args.fold}...")
            # TODO: 实现单折评估
            print("⚠️  单折评估功能待实现")
        else:
            print("评估所有折...")
            run_eval()
        
        print("\n✓ 评估完成！")
        
        # 自动汇总
        if args.aggregate:
            print("\n汇总评估结果...")
            aggregate_main()
        
        print(f"\n结果保存在:")
        print(f"  - 评估指标: {os.path.join(data_config['output_dir'], 'logs')}")
        print(f"  - 预测结果: {os.path.join(data_config['output_dir'], 'preds')}")
        
    except Exception as e:
        print(f"\n✗ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
