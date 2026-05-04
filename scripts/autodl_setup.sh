#!/bin/bash
# ============================================================
# AutoDL 云端环境部署脚本
# ============================================================
# 用途：在 AutoDL 实例上一键配置 HIST 训练环境
# 用法：SSH 连入 AutoDL 后执行 bash autodl_setup.sh
# ============================================================

set -e

echo "========================================="
echo "  AutoDL 环境部署脚本 - HIST 模型"
echo "========================================="

# 1. 检查 Python 环境
echo ""
echo "[1/6] 检查 Python 环境..."
if command -v python3 &>/dev/null; then
    PYTHON=python3
else
    PYTHON=python
fi

PYTHON_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo "✅ Python 版本: $PYTHON_VERSION"

# 2. 安装依赖
echo ""
echo "[2/6] 安装 Python 依赖..."
$PYTHON -m pip install --upgrade pip -q

echo "  → 安装 qlib..."
pip install pyqlib -q

echo "  → 安装数据工具..."
pip install akshare efinance pandas numpy -q

echo "  → 安装深度学习框架..."
pip install torch torchvision torchaudio -q

echo "✅ 依赖安装完成"

# 3. 克隆项目代码
echo ""
echo "[3/6] 克隆项目代码..."
if [ ! -d "qlib-project" ]; then
    git clone https://github.com/coo1kid1412/qlib-project.git
    echo "✅ 代码已克隆"
else
    cd qlib-project
    git pull
    cd ..
    echo "✅ 代码已更新"
fi

# 4. 解压数据
echo ""
echo "[4/6] 解压 qlib 数据..."
DATA_DIR="$HOME/.qlib/qlib_data"

# 查找最新的数据包
LATEST_PACKAGE=$(ls -t qlib_cn_data_*.tar.gz 2>/dev/null | head -1)

if [ -n "$LATEST_PACKAGE" ]; then
    mkdir -p "$DATA_DIR"
    tar -xzf "$LATEST_PACKAGE" -C "$DATA_DIR"
    echo "✅ 数据已解压: $LATEST_PACKAGE"
elif [ -d "$DATA_DIR/cn_data_yahoo" ]; then
    echo "✅ 数据目录已存在"
else
    echo "❌ 未找到数据包，请先上传 qlib_cn_data_*.tar.gz"
    exit 1
fi

# 5. 验证环境
echo ""
echo "[5/6] 验证环境..."
cd qlib-project

$PYTHON -c "
import qlib
import torch
import pandas as pd
import numpy as np
print('  qlib:', qlib.__version__ if hasattr(qlib, '__version__') else 'OK')
print('  torch:', torch.__version__)
print('  CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  GPU:', torch.cuda.get_device_name(0))
print('  pandas:', pd.__version__)
print('  numpy:', np.__version__)
"

echo "✅ 环境验证通过"

# 6. 生成训练脚本
echo ""
echo "[6/6] 生成快速训练脚本..."
cat > train_quick.sh << 'TRAINEOF'
#!/bin/bash
# 快速训练脚本 - 100 epochs
echo "开始训练 HIST 模型..."
python scripts/hist_benchmark.py --market csi500 --epochs 100 --early-stop 15

echo ""
echo "训练完成！"
echo "模型保存在: output/hist_report/hist_model_best.pt"
echo ""
echo "下载命令（在你的 Mac 上执行）："
echo "scp -P <端口号> root@<实例IP>:/root/qlib-project/output/hist_report/hist_model_best.pt \\"
echo "    ~/WorkSpace/QoderWorkspace/qlib_project/models/"
TRAINEOF

chmod +x train_quick.sh
echo "✅ 快速训练脚本已生成: train_quick.sh"

# 完成
echo ""
echo "========================================="
echo "  部署完成！"
echo "========================================="
echo ""
echo "下一步："
echo "1. 运行训练: bash train_quick.sh"
echo "2. 或直接运行: python scripts/hist_benchmark.py --market csi500 --epochs 100"
echo ""
echo "训练完成后下载模型："
echo "scp -P <端口号> root@<实例IP>:/root/qlib-project/output/hist_report/hist_model_best.pt \\"
echo "    ~/WorkSpace/QoderWorkspace/qlib_project/models/"
echo ""
echo "⚠️  记得用完立即关机释放实例！"
echo ""
