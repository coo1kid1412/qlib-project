#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AutoDL 云端训练配置
===================
用途：在 AutoDL 实例中一键配置并运行 HIST 模型训练
用法：python autodl_config.py --epochs 100
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_cuda():
    """检查 CUDA 环境"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
            return True
        else:
            print("⚠️  未检测到 GPU，将使用 CPU 训练")
            return False
    except ImportError:
        print("❌ PyTorch 未安装")
        return False


def setup_qlib():
    """配置 qlib 数据路径"""
    qlib_data = Path.home() / ".qlib" / "qlib_data" / "cn_data_yahoo"
    
    if not qlib_data.exists():
        print(f"❌ qlib 数据目录不存在: {qlib_data}")
        print("   请先上传并解压数据包")
        sys.exit(1)
    
    print(f"✅ qlib 数据目录: {qlib_data}")
    return str(qlib_data)


def run_training(epochs=100, early_stop=15, market="csi500"):
    """运行训练"""
    cmd = [
        sys.executable, "scripts/hist_benchmark.py",
        "--market", market,
        "--epochs", str(epochs),
        "--early-stop", str(early_stop),
    ]
    
    print(f"\n{'='*50}")
    print(f"  开始训练")
    print(f"{'='*50}")
    print(f"  市场: {market}")
    print(f"  Epochs: {epochs}")
    print(f"  早停: {early_stop}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*50}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print("\n✅ 训练完成！")
        print(f"模型保存位置: output/hist_report/hist_model_best.pt")
    else:
        print("\n❌ 训练失败，请检查日志")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="AutoDL 云端训练配置")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数 (default: 100)")
    parser.add_argument("--early-stop", type=int, default=15, help="早停轮数 (default: 15)")
    parser.add_argument("--market", default="csi500", help="市场 (default: csi500)")
    args = parser.parse_args()
    
    print("\n" + "="*50)
    print("  AutoDL 云端训练配置")
    print("="*50 + "\n")
    
    # 检查 GPU
    has_gpu = check_cuda()
    
    # 检查数据
    data_path = setup_qlib()
    
    # 运行训练
    run_training(
        epochs=args.epochs,
        early_stop=args.early_stop,
        market=args.market,
    )


if __name__ == "__main__":
    main()
