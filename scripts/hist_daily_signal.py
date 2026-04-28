#!/usr/bin/env python3
"""
HIST 每日选股信号
================
功能：用训练好的 HIST 模型对 CSI500 成分股做每日排名选股
信号含义：观测 T 日数据 -> 预测 T+1 买入、T+2 卖出的收益排名
使用：python3 hist_daily_signal.py [--date YYYY-MM-DD] [--model PATH] [--market csi500] [--topk 20]
"""
import os
import sys
import json
import argparse
import datetime
import numpy as np
import pandas as pd
from pathlib import Path

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.contrib.model.pytorch_hist import HISTModel

# ============================================================
# Configuration
# ============================================================
QLIB_PROVIDER = os.path.expanduser('~/.qlib/qlib_data/cn_data_yahoo')
MARKET = 'csi500'
DEFAULT_STOCK_INDEX = 1777  # HIST 必需：默认股票索引
FIT_START = '2008-01-01'    # 归一化基准起始日（必须覆盖整个训练期）
TOPK = 20
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'hist_report')
BEST_MODEL = os.path.join(MODEL_DIR, 'hist_model_best.pt')
OUTPUT_DIR = os.path.join(MODEL_DIR, 'daily')

# ============================================================
# Custom HIST with NaN-safe metric (division-by-zero protection)
# ============================================================
import torch

class CustomHIST(HISTModel):
    """HIST with fixed metric_fn to prevent NaN when std(pred)=0."""

    def __init__(self, *args, default_stock_index=DEFAULT_STOCK_INDEX, **kwargs):
        self._default_stock_index = default_stock_index
        super().__init__(*args, **kwargs)

    def metric_fn(self, pred, label):
        """IC with division-by-zero protection."""
        mask = torch.isfinite(label)
        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            denom = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
            eps = 1e-12
            return torch.sum(vx * vy) / (denom + eps)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def predict(self, dataset, segment="test"):
        """Override to inject default_stock_index for HIST inference."""
        import torch
        self.eval()
        dataset.config(handler_kwargs={"fit_start_time": FIT_START})
        dataset.setup_data(
            handler_kwargs={
                "init_type": "qlib.data.dataset.handler.DataHandlerLP",
                "fit_start_time": FIT_START,
                "segments": dataset.segments,
            },
            segment=segment,
        )
        sampler = dataset.prepare(segment, col_set=["feature", "label", "market_value", "price"])

        # Get data
        features = sampler["feature"]
        labels = sampler.get("label", None)

        # Build prediction DataFrame
        pred_scores = []
        with torch.no_grad():
            for i in range(len(features)):
                feat = features.iloc[i]
                # Ensure feat has default_stock_index
                if "stock_index" not in feat.columns:
                    feat = feat.copy()
                    feat["stock_index"] = self._default_stock_index

                x = torch.from_numpy(feat.values.astype(np.float32)).unsqueeze(0)
                pred = self(x)
                if pred.dim() > 2:
                    pred = pred[:, :, -1]
                score = pred.squeeze().item()
                pred_scores.append(score)

        pred_df = pd.DataFrame({
            'datetime': features.index.get_level_values('datetime'),
            'instrument': features.index.get_level_values('instrument'),
            'score': pred_scores
        })
        pred_df.set_index(['instrument', 'datetime'], inplace=True)
        return pred_df


# ============================================================
# Data loading
# ============================================================
def build_dataset(end_date):
    """构建推理数据集，使用 FIT_START 确保归一化一致。"""
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    # Get instruments
    market = MARKET
    if market == 'csi500':
        instruments = 'csi500'
    elif market == 'csi300':
        instruments = 'csi300'
    else:
        instruments = 'all'

    # Handler config
    from qlib.contrib.data.handler import Alpha360

    # Use a date range: FIT_START to end_date (for normalization + inference)
    # The handler will use data from FIT_START for RobustZScoreNorm fitting,
    # but we only predict for the last date.
    handler_conf = {
        "start_time": FIT_START,
        "end_time": end_date,
        "fit_start_time": FIT_START,
        "fit_end_time": end_date,
        "instruments": instruments,
    }

    dataset = DatasetH(
        handler=Alpha360,
        handler_config=handler_conf,
        segments={
            "train": (FIT_START, "2024-12-31"),
            "valid": ("2025-01-01", "2025-12-31"),
            "test": ("2026-01-01", end_date),
        },
    )

    return dataset


def load_stock_info(target_codes):
    """批量查询股票信息（名称、行业）。只查需要的股票。"""
    try:
        from pyqlibdata import get_stock_info
        # Use efinance or akshare as fallback
        import akshare as ak
        df = ak.stock_info_em_code_name()
        info_map = {}
        for _, row in df.iterrows():
            code = row['代码']
            if code.startswith('6'):
                sym = 'sh' + code
            else:
                sym = 'sz' + code
            info_map[sym] = {'name': row['名称']}
        return info_map
    except Exception:
        return {code: {'name': code} for code in target_codes}


def get_industry(code):
    """获取股票行业分类。"""
    try:
        import akshare as ak
        symbol = code[2:]  # sh600000 -> 600000
        if code.startswith('sh'):
            full_code = f"sh{symbol}"
        else:
            full_code = f"sz{symbol}"
        # Try to get industry from akshare
        # Using a simpler approach: stock_board_industry_name_em
        pass
    except Exception:
        pass
    return "未知"


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='HIST 每日选股')
    parser.add_argument('--date', type=str, default=None, help='信号日期 YYYY-MM-DD（默认最新）')
    parser.add_argument('--model', type=str, default=None, help='模型路径')
    parser.add_argument('--market', type=str, default=MARKET, help='市场: csi500, csi300, all')
    parser.add_argument('--topk', type=int, default=TOPK, help='选股数量')
    args = parser.parse_args()

    global MARKET, TOPK, BEST_MODEL, MODEL_DIR, OUTPUT_DIR
    MARKET = args.market
    TOPK = args.topk
    if args.model:
        BEST_MODEL = args.model
        MODEL_DIR = os.path.dirname(os.path.dirname(BEST_MODEL))
    OUTPUT_DIR = os.path.join(MODEL_DIR, 'daily')
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Init qlib
    qlib.init(provider_uri=QLIB_PROVIDER, region=REG_CN)

    # Determine signal date
    if args.date:
        signal_date = args.date
    else:
        # Get latest calendar date
        cal = D.calendar(freq='day')
        signal_date = cal[-1].strftime('%Y-%m-%d')

    model_name = os.path.basename(BEST_MODEL).replace('.pt', '').replace('.ckpt', '')
    print(f'\n=================================================================')
    print(f'  HIST 每日选股信号')
    print(f'  信号日期: {signal_date}')
    print(f'  模型: {model_name}')
    print(f'  市场: {MARKET.upper()} | Top-K: {TOPK}')
    print(f'=================================================================\n')

    # [1/4] Build dataset
    print('[1/4] 构建推理数据集...')
    dataset = build_dataset(signal_date)
    print(f'  数据窗口: {FIT_START} ~ {signal_date} (含训练期归一化基准)')

    # [2/4] Load model and predict
    print(f'\n[2/4] 加载模型并预测...')

    # Load model config from checkpoint
    model = CustomHIST(
        loss='mse',
        metric='ic',
    )
    model.load(BEST_MODEL)
    model.to(device='cpu')
    model.eval()

    # Get predictions for the signal date
    # Use dataset's test segment to get features for the last date
    dataset.setup_data(
        handler_kwargs={
            "init_type": "qlib.data.dataset.handler.DataHandlerLP",
            "fit_start_time": FIT_START,
            "segments": dataset.segments,
        },
        segment="test",
    )
    handler = dataset.handler
    df = handler.fetch()

    # Filter to the signal date
    dates = df.index.get_level_values('datetime').unique()
    # Find the date closest to signal_date but not after
    target_date = pd.Timestamp(signal_date)
    valid_dates = dates[dates <= target_date]
    if len(valid_dates) == 0:
        print(f'  错误: 没有 <= {signal_date} 的数据')
        sys.exit(1)
    actual_date = valid_dates[-1]

    df_date = df[df.index.get_level_values('datetime') == actual_date]
    print(f'  特征股票数: {len(df_date)}')

    # Run prediction
    pred_scores = []
    instruments_list = []
    with torch.no_grad():
        for idx in df_date.index:
            inst = idx[0]
            row = df_date.loc[idx]
            # Get feature columns
            feat_cols = [c for c in row.index if c not in ('label', 'market_value', 'price', 'stock_index')]
            features = row[feat_cols].values.astype(np.float32)

            # Check for all NaN
            if np.all(np.isnan(features)):
                continue

            # Add stock_index if needed
            x = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)  # (1, 1, n_features)

            # For HIST, we need to provide stock_index
            # Try to get it from the data or use default
            stock_idx_val = DEFAULT_STOCK_INDEX

            try:
                pred = model(x)
                if pred.dim() > 2:
                    pred = pred[:, :, -1]
                score = pred.squeeze().item()
                pred_scores.append(score)
                instruments_list.append(inst)
            except Exception as e:
                pass

    if not pred_scores:
        print('  错误: 没有有效预测结果')
        sys.exit(1)

    pred_df = pd.DataFrame({
        'instrument': instruments_list,
        'score': pred_scores
    })
    print(f'  有效预测股票数: {len(pred_df)}')
    print(f'  分数分布: min={pred_df["score"].min():.4f}, median={pred_df["score"].median():.4f}, max={pred_df["score"].max():.4f}')

    # [3/4] Ranking
    print(f'\n[3/4] 生成排名...')
    pred_df = pred_df.sort_values('score', ascending=False).reset_index(drop=True)
    pred_df['rank'] = range(1, len(pred_df) + 1)

    topk = pred_df.head(TOPK).copy()

    # [4/4] Load stock info
    print(f'[4/4] 加载股票信息...')
    target_codes = topk['instrument'].tolist()
    info_map = load_stock_info(target_codes)

    topk['name'] = topk['instrument'].map(lambda x: info_map.get(x, {}).get('name', x))
    topk['change'] = 'NEW'  # Default, will be updated if history exists

    # Load history for change tracking
    history_path = os.path.join(OUTPUT_DIR, 'signal_history.csv')
    prev_topk = None
    if os.path.exists(history_path):
        hist = pd.read_csv(history_path)
        if 'date' in hist.columns:
            last_date = hist['date'].max()
            prev_topk = hist[hist['date'] == last_date].set_index('instrument')

    # Update change status
    for idx, row in topk.iterrows():
        inst = row['instrument']
        if prev_topk is not None and inst in prev_topk.index:
            topk.at[idx, 'change'] = 'HOLD'

    # Print results
    date_str = actual_date.strftime('%Y-%m-%d')
    print(f'\n===========================================================================')
    print(f'  HIST 每日选股信号 - {date_str}')
    print(f'  模型: {model_name} | 市场: {MARKET.upper()}')
    print(f'===========================================================================')
    print(f'    {"#":>3} | {"代码":<12} | {"名称":<10} | {"分数":>10} | 变动')
    print(f'  ---------------------------------------------------------------------')
    for _, row in topk.iterrows():
        print(f'    {row["rank"]:>3} | {row["instrument"]:<12} | {row["name"]:<10} | {row["score"]:>+10.4f} | {row["change"]:<6}')

    # Exit stocks
    if prev_topk is not None:
        exited = prev_topk.index.difference(topk['instrument'])
        if len(exited) > 0:
            print(f'  ---------------------------------------------------------------------')
            print(f'  退出 ({len(exited)} 只):')
            for inst in exited:
                prev_rank = prev_topk.loc[inst, 'rank']
                curr = topk[topk['instrument'] == inst]
                curr_rank = curr['rank'].iloc[0] if len(curr) > 0 else 'N/A'
                name = prev_topk.loc[inst, 'name']
                print(f'    {inst} ({name}, 原排名#{int(prev_rank)} -> 当前#{curr_rank})')

    # Stats
    new_count = (topk['change'] == 'NEW').sum()
    hold_count = (topk['change'] == 'HOLD').sum()
    if prev_topk is not None:
        exit_count = len(prev_topk.index.difference(topk['instrument']))
        print(f'\n  变动统计: NEW={new_count}, HOLD={hold_count}, EXIT={exit_count}')
        if exit_count > 0:
            print(f'  预估换手: {new_count}/{exit_count} = {new_count*100//max(exit_count,1)}%')
    else:
        print(f'\n  变动统计: NEW={new_count}, HOLD={hold_count} (无历史对比)')
    print(f'===========================================================================\n')

    # Save outputs
    picks_path = os.path.join(OUTPUT_DIR, f'picks_{date_str}.json')
    picks_data = topk[['rank', 'instrument', 'name', 'score', 'change']].to_dict(orient='records')
    with open(picks_path, 'w', encoding='utf-8') as f:
        json.dump(picks_data, f, ensure_ascii=False, indent=2)
    print(f'  Picks: {picks_path}')

    # Save full signal CSV
    csv_path = os.path.join(OUTPUT_DIR, f'signal_{date_str}.csv')
    pred_df.to_csv(csv_path, index=False)
    print(f'  Full signal: {csv_path}')

    # Update history
    topk_save = topk[['rank', 'instrument', 'name', 'score', 'change']].copy()
    topk_save['date'] = date_str
    if os.path.exists(history_path):
        hist_df = pd.read_csv(history_path)
        # Remove existing entries for this date
        hist_df = hist_df[hist_df['date'] != date_str]
        hist_df = pd.concat([hist_df, topk_save], ignore_index=True)
    else:
        hist_df = topk_save
    hist_df.to_csv(history_path, index=False)
    print(f'  History: {history_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
