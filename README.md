# Qlib HIST 量化模型项目

A 股量化投资系统，基于微软 Qlib 框架，使用 HIST 模型进行股票预测。

## 📁 项目结构

```
qlib_project/
├── scripts/
│   ├── hist_benchmark.py      # HIST 模型训练与评估
│   ├── hist_daily_signal.py   # 每日选股信号生成
│   ├── hist_backtest.py       # 投资组合回测
│   ├── update_qlib_data.py    # 数据更新流水线
│   ├── fast_dump_bin.py       # 快速二进制数据重建
│   ├── pack_data.sh           # 数据打包脚本（本地）
│   ├── autodl_setup.sh        # AutoDL 环境部署脚本
│   └── autodl_config.py       # AutoDL 云端训练配置
├── output/                    # 训练输出
└── models/                    # 模型权重
```

## 🚀 快速开始

### 本地训练（CPU，适合调试）

```bash
# 安装依赖
pip install pyqlib akshare efinance

# 训练 3 epochs 测试
python scripts/hist_benchmark.py --market csi500 --epochs 3 --early-stop 2
```

### ☁️ AutoDL 云端训练（GPU，推荐）

#### 1. 本地打包数据

```bash
bash scripts/pack_data.sh
```

#### 2. 上传到 AutoDL

- 登录 [AutoDL 控制台](https://www.autodl.com)
- 创建 RTX 4090 实例（PyTorch 2.0+ 镜像）
- 通过网盘或 scp 上传数据包

#### 3. 云端部署

```bash
# SSH 连入实例
ssh root@<实例IP> -p <端口号>

# 一键部署环境
bash autodl_setup.sh

# 开始训练
python autodl_config.py --epochs 100
# 或
bash train_quick.sh
```

#### 4. 下载模型

```bash
scp -P <端口号> root@<实例IP>:/root/qlib-project/output/hist_report/hist_model_best.pt \
    ~/WorkSpace/QoderWorkspace/qlib_project/models/
```

## 📊 模型配置

| 参数 | 值 |
|------|-----|
| 市场 | CSI 500 |
| Handler | Alpha360 |
| 特征 | OHLCV (5 维时序) |
| 基础模型 | GRU |
| 隐藏层 | 64 |
| 层数 | 2 |
| Dropout | 0.0 |
| 学习率 | 0.0001 |
| 训练期 | 2008-2024 |
| 验证期 | 2025 |
| 测试期 | 2026+ |

## 💰 费用预估

| 平台 | GPU | 价格 | 100 epochs 时间 | 费用 |
|------|-----|------|----------------|------|
| 本地 Mac | CPU | ¥0 | ~42 小时 | ¥0 |
| AutoDL | RTX 4090 | ¥1.29/h | ~25 分钟 | ¥0.5-1 |
| 阿里云 | V100 | ¥5-8/h | ~30 分钟 | ¥3-5 |

## ⚠️ 注意事项

1. **用完即关机**：AutoDL 按小时计费，训练完立即释放
2. **保存数据**：关机前下载模型和重要文件
3. **学生认证**：AutoDL 学生认证可享 5-7 折优惠
4. **余额充足**：保持 ¥20+ 余额，避免强制关机

## 📝 修复记录

- NaN 保护：metric_fn 除零保护 (eps=1e-12)
- NaN 保护：test_epoch 使用 np.nanmean
- 路径修复：HIST_DATA_DIR 正确指向 cn_data_yahoo
- 设备修复：torch.get_device CPU 兼容性
- Handler 修复：Alpha158 → Alpha360（特征维度匹配）

## 🔗 相关链接

- [GitHub 仓库](https://github.com/coo1kid1412/qlib-project)
- [Qlib 官方](https://github.com/microsoft/qlib)
- [AutoDL](https://www.autodl.com)
