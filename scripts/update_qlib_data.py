#!/usr/bin/env python3
"""
增量更新 qlib Yahoo CN 数据
===========================
1. 从 Yahoo Finance 下载增量行情
2. 追加到 source CSV
3. 计算归一化并追加到 normalize CSV
4. 用 fast_dump_bin 重建 qlib 二进制
5. 更新 index instruments 日期范围

使用：python3 update_qlib_data.py
"""
import os
import sys
import glob
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.expanduser('~/.qlib/stock_data/source/yahoo_cn')
NORMALIZE_DIR = os.path.expanduser('~/.qlib/stock_data/normalize/yahoo_cn')
QLIB_DIR = os.path.expanduser('~/.qlib/qlib_data/cn_data_yahoo')
FAST_DUMP_BIN = os.path.join(SCRIPT_DIR, 'fast_dump_bin.py')


def symbol_to_yahoo(sym):
    """qlib symbol -> Yahoo ticker: sh600000 -> 600000.SS"""
    if sym.startswith('sh'):
        return sym[2:] + '.SS'
    elif sym.startswith('sz'):
        return sym[2:] + '.SZ'
    return None


def get_all_symbols():
    """获取所有股票 symbol。"""
    files = glob.glob(os.path.join(SOURCE_DIR, '*.csv'))
    return sorted([os.path.basename(f).replace('.csv', '') for f in files])


def get_latest_source_date(sym):
    """获取 source CSV 最新日期。"""
    path = os.path.join(SOURCE_DIR, f'{sym}.csv')
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=['date'], dtype={'date': str})
    return df['date'].iloc[-1] if not df.empty else None


def download_batch(tickers, start_date, end_date):
    """批量下载 Yahoo 数据。"""
    import yfinance as yf
    try:
        df = yf.download(tickers, start=start_date, end=end_date, progress=False,
                         threads=True, group_by='ticker')
        return df
    except Exception as e:
        print(f'  [WARN] Batch download failed: {e}')
        return None


def append_source_csv(sym, new_rows_df):
    """追加新行到 source CSV。"""
    path = os.path.join(SOURCE_DIR, f'{sym}.csv')
    if new_rows_df.empty:
        return 0

    existing = pd.read_csv(path, usecols=['date'], dtype={'date': str})
    existing_dates = set(existing['date'].values)

    rows = []
    for idx, row in new_rows_df.iterrows():
        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10]
        if date_str in existing_dates:
            continue
        if pd.isna(row.get('Open')) or pd.isna(row.get('Close')):
            continue
        rows.append({
            'date': date_str,
            'open': row['Open'],
            'high': row['High'],
            'low': row['Low'],
            'close': row['Close'],
            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0,
            'adjclose': row['Adj Close'] if 'Adj Close' in row.index else row['Close'],
            'symbol': sym,
        })

    if not rows:
        return 0

    append_df = pd.DataFrame(rows)
    append_df.to_csv(path, mode='a', header=False, index=False)
    return len(rows)


def get_normalize_ratio(sym):
    """从已有 normalize CSV 获取归一化比率。"""
    src_path = os.path.join(SOURCE_DIR, f'{sym}.csv')
    norm_path = os.path.join(NORMALIZE_DIR, f'{sym}.csv')

    if not os.path.exists(norm_path):
        return None, None, None

    src_df = pd.read_csv(src_path)
    norm_df = pd.read_csv(norm_path)

    if src_df.empty or norm_df.empty:
        return None, None, None

    src_last = src_df.iloc[-1]
    norm_last = norm_df.iloc[-1]

    if str(src_last['date'])[:10] != str(norm_last['date'])[:10]:
        norm_date = str(norm_last['date'])[:10]
        match = src_df[src_df['date'].astype(str).str[:10] == norm_date]
        if match.empty:
            return None, None, None
        src_last = match.iloc[-1]

    src_close = float(src_last['close'])
    norm_close = float(norm_last['close'])
    src_vol = float(src_last['volume'])
    norm_vol = float(norm_last['volume'])

    if src_close == 0 or src_vol == 0:
        return None, None, None

    return norm_close / src_close, norm_vol / src_vol, norm_close


def append_normalize_csv(sym, new_source_rows, price_ratio, volume_ratio, last_norm_close):
    """归一化新数据并追加到 normalize CSV。"""
    norm_path = os.path.join(NORMALIZE_DIR, f'{sym}.csv')
    if not os.path.exists(norm_path):
        return 0

    existing = pd.read_csv(norm_path, usecols=['date'], dtype={'date': str})
    existing_dates = set(existing['date'].astype(str).str[:10].values)

    rows = []
    prev_close = last_norm_close

    for _, row in new_source_rows.iterrows():
        date_str = str(row['date'])[:10]
        if date_str in existing_dates:
            continue

        rows.append({
            'date': date_str,
            'open': float(row['open']) * price_ratio,
            'close': float(row['close']) * price_ratio,
            'high': float(row['high']) * price_ratio,
            'low': float(row['low']) * price_ratio,
            'volume': float(row['volume']) * volume_ratio,
            'change': (float(row['close']) * price_ratio / prev_close - 1) if prev_close != 0 else 0.0,
            'factor': 1.0,
            'symbol': sym,
        })
        prev_close = float(row['close']) * price_ratio

    if not rows:
        return 0

    append_df = pd.DataFrame(rows)
    append_df.to_csv(norm_path, mode='a', header=False, index=False)
    return len(rows)


def main():
    import yfinance as yf
    from datetime import datetime, timedelta
    import subprocess

    print('=' * 65)
    print('  Qlib Yahoo CN 数据增量更新')
    print('=' * 65)

    # 1. Discover stocks
    symbols = get_all_symbols()
    print(f'\n[1/6] 发现 {len(symbols)} 只股票')

    sample_date = get_latest_source_date(symbols[0])
    print(f'  当前数据最新: {sample_date}')
    end_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f'  下载区间: {sample_date} ~ {end_date}')

    # 2. Download
    tickers = [symbol_to_yahoo(s) for s in symbols if symbol_to_yahoo(s)]
    ticker_map = {t: s for s, t in zip(symbols, tickers) if t}
    print(f'\n[2/6] 下载 {len(tickers)} 只股票...')

    BATCH_SIZE = 500
    all_data = {}
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        t0 = time.time()
        df = download_batch(batch, sample_date, end_date)
        print(f'  批次 {batch_num}/{(len(tickers)+BATCH_SIZE-1)//BATCH_SIZE}: {len(batch)} 只, {time.time()-t0:.1f}s')

        if df is None or df.empty:
            continue

        if len(batch) == 1:
            ticker = batch[0]
            sym = ticker_map[ticker]
            valid = df.dropna(subset=['Open', 'Close'])
            if not valid.empty:
                all_data[sym] = valid
        else:
            for ticker in batch:
                sym = ticker_map.get(ticker)
                try:
                    stock_df = df.xs(ticker, level='Ticker', axis=1)
                    valid = stock_df.dropna(subset=['Open', 'Close'])
                    if not valid.empty:
                        all_data[sym] = valid
                except Exception:
                    pass

    print(f'  获取有效数据: {len(all_data)} 只')

    # 3. Append source CSV
    print(f'\n[3/6] 更新 source CSV...')
    source_updated = source_rows = 0
    for sym, df in all_data.items():
        n = append_source_csv(sym, df)
        if n > 0:
            source_updated += 1
            source_rows += n
    print(f'  更新 {source_updated} 只, 共 {source_rows} 行')

    # 4. Normalize
    print(f'\n[4/6] 更新 normalize CSV...')
    norm_updated = 0
    for sym in all_data.keys():
        src_path = os.path.join(SOURCE_DIR, f'{sym}.csv')
        src_df = pd.read_csv(src_path)

        price_ratio, volume_ratio, last_norm_close = get_normalize_ratio(sym)
        if price_ratio is None:
            continue

        norm_path = os.path.join(NORMALIZE_DIR, f'{sym}.csv')
        norm_existing = pd.read_csv(norm_path, usecols=['date'], dtype={'date': str})
        existing_dates = set(norm_existing['date'].astype(str).str[:10].values)
        new_rows = src_df[~src_df['date'].astype(str).str[:10].isin(existing_dates)]
        if new_rows.empty:
            continue

        n = append_normalize_csv(sym, new_rows, price_ratio, volume_ratio, last_norm_close)
        if n > 0:
            norm_updated += 1
    print(f'  更新 {norm_updated} 只')

    # 5. Rebuild binary
    print(f'\n[5/6] 重建 qlib 二进制...')
    result = subprocess.run([sys.executable, FAST_DUMP_BIN], capture_output=False, timeout=300)
    if result.returncode == 0:
        print('  二进制重建成功!')
    else:
        print(f'  重建失败, 返回码: {result.returncode}')

    # 6. Update instruments
    print(f'\n[6/6] 更新 index instruments...')
    src_df = pd.read_csv(os.path.join(SOURCE_DIR, f'{symbols[0]}.csv'))
    latest_date = str(src_df['date'].iloc[-1])[:10]
    inst_dir = os.path.join(QLIB_DIR, 'instruments')
    for idx_file in ['csi300.txt', 'csi500.txt', 'csi800.txt', 'csi100.txt']:
        idx_path = os.path.join(inst_dir, idx_file)
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                content = f.read()
            dates = re.findall(r'(\d{4}-\d{2}-\d{2})', content)
            if dates:
                max_date = max(dates)
                if max_date < latest_date:
                    content = content.replace(max_date, latest_date)
                    with open(idx_path, 'w') as f:
                        f.write(content)
                    print(f'  {idx_file}: {max_date} -> {latest_date}')

    print(f'\n{"=" * 65}')
    print(f'  更新完成! 最新日期: {latest_date}')
    print(f'{"=" * 65}')


if __name__ == '__main__':
    main()
