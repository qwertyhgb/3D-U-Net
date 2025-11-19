"""
文件名称：predict.py
文件功能：实现独立的模型预测脚本，支持测试时增强(TTA)。
作者：TraeAI 助手
创建日期：2025-11-19
版本：v1.0
版权声明：Copyright (c) 2025, All rights reserved.

说明：
- 支持单模态和多模态预测；
- 集成测试时增强(TTA)功能；
- 可加载训练好的模型权重进行预测；
- 支持批量处理多个病例。
"""

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import get_data_config, get_training_config
from .dataset import ProstateDataset, collect_cases, collect_cases_multi
from .unet3d import UNet3D
from .utils import read_volume, clip_and_normalize, pad_or_crop_to_shape, save_nifti_like
# 导入torchio相关的TTA功能
try:
    from .torchio_transforms import apply_tta_single, apply_tta_multi
    TORCHIO_AVAILABLE = True
except ImportError:
    TORCHIO_AVAILABLE = False
    print("Warning: torchio not available. TTA will be disabled.")


def predict_single_case(model, case_info, device, modalities, use_tta=False):
    """对单个病例进行预测。
    
    参数：
        model (torch.nn.Module): 训练好的模型。
        case_info (dict): 病例信息，包含路径等。
        device (torch.device): 设备。
        modalities (list): 模态列表。
        use_tta (bool): 是否使用测试时增强。
    返回：
        np.ndarray: 预测结果。
    """
    # 读取数据
    if 'image' in case_info:
        # 单模态
        imgs = [read_volume(case_info['image'])]
    else:
        # 多模态
        imgs = [read_volume(case_info['images'][m]) for m in modalities]
    
    # 预处理
    imgs = [clip_and_normalize(v) for v in imgs]
    target_shape = tuple(get_data_config().get('target_shape', (256, 256, 32)))
    imgs = [pad_or_crop_to_shape(v, target_shape) for v in imgs]
    
    model.eval()
    
    with torch.no_grad():
        if use_tta and TORCHIO_AVAILABLE:
            # 使用测试时增强
            if len(imgs) == 1:
                prediction = apply_tta_single(imgs[0], model, device)
            else:
                prediction = apply_tta_multi(imgs, model, device)
        else:
            # 标准预测
            img_stack = np.stack(imgs, axis=0)  # 堆叠成 [C, Z, Y, X] 格式
            input_tensor = torch.from_numpy(img_stack).unsqueeze(0).float().to(device)
            # 模型输出已经包含 Sigmoid，不需要再次应用
            prediction = model(input_tensor).cpu().numpy()[0, 0]
    
    return prediction


def predict_batch(model_path, cases, output_dir, device, modalities, use_tta=False):
    """批量预测。
    
    参数：
        model_path (str): 模型权重路径。
        cases (list): 病例信息列表。
        output_dir (str): 输出目录。
        device (torch.device): 设备。
        modalities (list): 模态列表。
        use_tta (bool): 是否使用测试时增强。
    """
    # 初始化模型
    model = UNet3D(in_channels=len(modalities)).to(device)
    
    # 加载模型权重
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历所有病例进行预测
    for case_info in tqdm(cases, desc="Predicting"):
        case_id = case_info['id']
        
        try:
            # 进行预测
            prediction = predict_single_case(model, case_info, device, modalities, use_tta)
            
            # 保存预测结果
            output_path = os.path.join(output_dir, f"{case_id}_pred.nii")
            
            # 获取参考路径用于保存NIfTI
            ref_path = case_info['image'] if 'image' in case_info else case_info['images'][modalities[0]]
            
            # 保存为NIfTI格式
            save_nifti_like(ref_path, prediction, output_path)
            
            print(f"Saved prediction for {case_id} to {output_path}")
            
        except Exception as e:
            print(f"Error predicting {case_id}: {e}")
            continue


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="Prostate Cancer Segmentation Prediction")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default="./outputs/predictions", help="Output directory for predictions")
    parser.add_argument("--use_tta", action="store_true", help="Enable Test-Time Augmentation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")
    parser.add_argument("--modality", type=str, nargs="+", help="Input modalities (e.g., DWI ADC T2)")
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 获取配置
    data_config = get_data_config()
    training_config = get_training_config()
    
    # 确定模态
    modalities = args.modality or data_config.get('modalities', ['DWI'])
    print(f"Using modalities: {modalities}")
    
    # 收集病例
    if len(modalities) == 1:
        cases = collect_cases(modalities[0])
    else:
        cases = collect_cases_multi(modalities)
    
    print(f"Found {len(cases)} cases for prediction")
    
    # 进行预测
    predict_batch(
        model_path=args.model_path,
        cases=cases,
        output_dir=args.output_dir,
        device=device,
        modalities=modalities,
        use_tta=args.use_tta
    )
    
    print("Prediction completed!")


if __name__ == "__main__":
    main()