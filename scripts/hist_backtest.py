#!/usr/bin/env python3
"""
HIST 组合回测
============
功能：用训练好的模型预测 + TopkDropoutStrategy 进行组合回测
"""
import os
import sys
import argparse
import pickle

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.contrib.model.pytorch_hist import HISTModel
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord
from qlib.utils import init_instance_by_config
from qlib.tests.data import GetData

# ============================================================
# Configuration
# ============================================================
QLIB_PROVIDER = os.path.expanduser('~/.qlib/qlib_data/cn_data_yahoo')
HIST_DATA_DIR = os.path.join(QLIB_PROVIDER, 'hist_data')
FIT_START = '2008-01-01'
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'hist_report')


def run_backtest(model_path=None, topk=20, n_drop=5, market='csi500',
                 train_end='2024-12-31', valid_end='2025-12-31', output_dir=None):
    """运行完整回测流程。"""
    output_dir = output_dir or REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Init qlib
    qlib.init(provider_uri=QLIB_PROVIDER, region=REG_CN)

    instruments = market if market in ('csi300', 'csi500', 'csi800', 'all') else 'all'

    # Dataset config
    dataset_config = {
        "class": "DatasetH",
        "module_path": "qlib.data.dataset",
        "kwargs": {
            "handler": {
                "class": "Alpha360",
                "module_path": "qlib.contrib.data.handler",
                "kwargs": {
                    "start_time": FIT_START,
                    "end_time": "2026-12-31",
                    "fit_start_time": FIT_START,
                    "fit_end_time": train_end,
                    "instruments": instruments,
                },
            },
            "segments": {
                "train": (FIT_START, train_end),
                "valid": ("2025-01-01", valid_end),
                "test": ("2026-01-01", "2026-12-31"),
            },
        },
    }

    # Model config
    stock2concept_path = os.path.join(HIST_DATA_DIR, f'stock2concept_{market}.npy')
    stock_index_path = os.path.join(HIST_DATA_DIR, f'stock_index_{market}.npy')
    if not os.path.exists(stock2concept_path):
        stock2concept_path = os.path.join(HIST_DATA_DIR, 'stock2concept.npy')
        stock_index_path = os.path.join(HIST_DATA_DIR, 'stock_index.npy')

    model_config = {
        "class": "HIST",
        "module_path": "qlib.contrib.model.pytorch_hist",
        "kwargs": {
            "d_feat": 5,
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.0,
            "n_epochs": 100,
            "lr": 0.0001,
            "metric": "ic",
            "early_stop": 15,
            "loss": "mse",
            "base_model": "GRU",
            "stock2concept": stock2concept_path,
            "stock_index": stock_index_path,
            "optimizer": "adam",
            "GPU": -1,
            "seed": 42,
        },
    }

    # Strategy config
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "signal": "<PRED>",
            "topk": topk,
            "n_drop": n_drop,
        },
    }

    # Create dataset
    dataset = init_instance_by_config(dataset_config)

    # Create and train model
    model = init_instance_by_config(model_config)
    model.fit(dataset)

    # Save model
    model_path = model_path or os.path.join(output_dir, 'hist_model_best.pt')
    if not os.path.exists(model_path):
        import torch
        torch.save(model.HIST_model.state_dict(), model_path)

    # Run analysis with workflow
    with R.start(experiment_name="hist_backtest"):
        # Signal record
        sr = SignalRecord(model, dataset, output_dir)
        sr.generate()

        # Portfolio analysis
        par = PortAnaRecord(
            output_dir,
            config={
                "strategy": strategy_config,
                "start_time": "2026-01-01",
                "end_time": "2026-12-31",
                "account": 100000000,
                "benchmark": "SH000905",  # CSI500
            },
            risk_analysis_freq="day",
        )
        par.generate()

    print(f'\n回测完成! 结果保存在: {output_dir}')
    print(f'  - backtest_results.pkl')
    print(f'  - backtest_top20.png')

    return par


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HIST 组合回测')
    parser.add_argument('--topk', type=int, default=20)
    parser.add_argument('--n-drop', type=int, default=5)
    parser.add_argument('--market', type=str, default='csi500')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--model', type=str, default=None, help='已有模型路径（跳过训练直接回测）')
    args = parser.parse_args()

    run_backtest(
        model_path=args.model,
        topk=args.topk,
        n_drop=args.n_drop,
        market=args.market,
        output_dir=args.output,
    )
