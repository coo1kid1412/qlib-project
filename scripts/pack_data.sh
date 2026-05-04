#!/bin/bash
# ============================================================
# 数据打包脚本 - 用于上传到云端 GPU 服务器
# ============================================================
# 用途：将 qlib 数据打包压缩，方便上传到 AutoDL 等云端平台
# 用法：bash scripts/pack_data.sh
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "  Qlib 数据打包脚本"
echo "========================================="

# 配置
QLIB_DATA_DIR="$HOME/.qlib/qlib_data/cn_data_yahoo"
STOCK_DATA_DIR="$HOME/.qlib/stock_data"
OUTPUT_DIR="$PROJECT_DIR/data_package"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 检查数据目录是否存在
echo ""
echo "[1/5] 检查数据目录..."
if [ ! -d "$QLIB_DATA_DIR" ]; then
    echo "❌ 错误: qlib 数据目录不存在: $QLIB_DATA_DIR"
    echo "   请先运行 update_qlib_data.py 下载数据"
    exit 1
fi

echo "✅ qlib 数据目录存在: $QLIB_DATA_DIR"

# 创建输出目录
echo ""
echo "[2/5] 创建输出目录..."
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录: $OUTPUT_DIR"

# 计算数据大小
echo ""
echo "[3/5] 计算数据大小..."
QLIB_SIZE=$(du -sh "$QLIB_DATA_DIR" | cut -f1)
echo "📊 qlib 二进制数据: $QLIB_SIZE"

if [ -d "$STOCK_DATA_DIR" ]; then
    STOCK_SIZE=$(du -sh "$STOCK_DATA_DIR" | cut -f1)
    echo "📊 stock CSV 数据: $STOCK_SIZE"
else
    echo "⚠️  stock CSV 数据目录不存在（可选）"
    STOCK_SIZE="0"
fi

# 打包 qlib 二进制数据（必需）
echo ""
echo "[4/5] 打包 qlib 二进制数据..."
PACK_NAME="qlib_cn_data_${TIMESTAMP}.tar.gz"
PACK_PATH="$OUTPUT_DIR/$PACK_NAME"

tar -czf "$PACK_PATH" -C "$HOME/.qlib/qlib_data" cn_data_yahoo

PACK_SIZE=$(du -sh "$PACK_PATH" | cut -f1)
echo "✅ 打包完成: $PACK_PATH ($PACK_SIZE)"

# 生成上传说明
echo ""
echo "[5/5] 生成上传说明..."
README_PATH="$OUTPUT_DIR/UPLOAD_INSTRUCTIONS_${TIMESTAMP}.txt"

cat > "$README_PATH" << EOF
=====================================
  Qlib 数据上传说明
=====================================

打包时间: $(date '+%Y-%m-%d %H:%M:%S')
数据包大小: $PACK_SIZE

上传步骤（AutoDL 为例）:
------------------------

方法 1: 使用 AutoDL 网盘
1. 登录 AutoDL 控制台
2. 进入「网盘」页面
3. 上传文件: $PACK_NAME
4. 在实例中挂载网盘或复制到实例

方法 2: 使用 scp 命令
scp $PACK_PATH root@<your-autodl-ip>:/root/

方法 3: 使用 rsync
rsync -avz $PACK_PATH root@<your-autodl-ip>:/root/

解压命令（在云端实例中）:
------------------------
cd /root
tar -xzf $PACK_NAME -C ~/.qlib/qlib_data/

验证解压:
--------
ls -la ~/.qlib/qlib_data/cn_data_yahoo/
# 应该看到: hist_data/, stocks/, 等目录

训练命令:
--------
cd /root/qlib-project
python scripts/hist_benchmark.py --market csi500 --epochs 100

=====================================
EOF

echo "✅ 上传说明已生成: $README_PATH"

# 显示总结
echo ""
echo "========================================="
echo "  打包完成!"
echo "========================================="
echo ""
echo "📦 数据包: $PACK_PATH ($PACK_SIZE)"
echo "📄 说明文件: $README_PATH"
echo ""
echo "下一步:"
echo "1. 将数据包上传到云端 GPU 服务器"
echo "2. 在云端运行 autodl_setup.sh 安装环境"
echo "3. 解压数据后运行训练脚本"
echo ""
