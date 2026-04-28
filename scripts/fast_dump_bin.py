#!/usr/bin/env python3
"""
快速 qlib 二进制数据重建
========================
从 normalize CSV 重建 qlib bin 文件（替代慢速 dump_bin）
用时约 90 秒（5193 只股票）。
"""
import os
import glob
import time
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

NORMALIZE_DIR = os.path.expanduser("~/.qlib/stock_data/normalize/yahoo_cn")
QLIB_DIR = os.path.expanduser("~/.qlib/qlib_data/cn_data_yahoo")
FEATURES = ["open", "close", "high", "low", "volume", "change", "factor"]
CALENDAR_FILE = os.path.join(QLIB_DIR, "calendars", "day.txt")
INSTRUMENTS_FILE = os.path.join(QLIB_DIR, "instruments", "all.txt")
FEATURES_DIR = os.path.join(QLIB_DIR, "features")


def build_calendar():
    """从所有 normalize CSV 构建交易日日历。"""
    print("[1/4] Building calendar from all CSVs...")
    all_dates = set()
    csv_files = glob.glob(os.path.join(NORMALIZE_DIR, "*.csv"))
    for i, f in enumerate(csv_files):
        try:
            df = pd.read_csv(f, usecols=["date"], dtype={"date": str})
            dates = df["date"].dropna().unique()
            all_dates.update(dates)
        except Exception:
            pass
        if (i + 1) % 1000 == 0:
            print(f"  Scanned {i+1}/{len(csv_files)} CSVs, {len(all_dates)} unique dates so far")

    calendar = sorted(all_dates)
    print(f"  Total: {len(calendar)} trading days, {calendar[0]} ~ {calendar[-1]}")
    return calendar


def write_calendar(calendar):
    """写入 calendar day.txt"""
    os.makedirs(os.path.dirname(CALENDAR_FILE), exist_ok=True)
    with open(CALENDAR_FILE, "w") as f:
        for d in calendar:
            f.write(d + "\n")
    print(f"  Calendar written: {len(calendar)} entries")


def process_stock(args):
    """处理单只股票：读 CSV -> 写 bin 文件。
    
    qlib bin 格式: [start_index as float32][data_0][data_1]...[data_n]
    """
    csv_path, date_to_idx, calendar_len = args
    symbol = os.path.basename(csv_path).replace(".csv", "")

    try:
        df = pd.read_csv(csv_path, dtype={"date": str, "symbol": str})
        df = df.dropna(subset=["date"])
        df = df[df["date"].isin(date_to_idx)]
        if df.empty:
            return None

        df["_idx"] = df["date"].map(date_to_idx)
        df = df.sort_values("_idx").drop_duplicates(subset=["_idx"], keep="last")

        start_idx = int(df["_idx"].min())
        end_idx = int(df["_idx"].max())
        length = end_idx - start_idx + 1

        stock_dir = os.path.join(FEATURES_DIR, symbol)
        os.makedirs(stock_dir, exist_ok=True)

        for feat in FEATURES:
            arr = np.full(length, np.nan, dtype=np.float32)
            if feat in df.columns:
                valid = df[["_idx", feat]].dropna(subset=[feat])
                if not valid.empty:
                    indices = (valid["_idx"].values - start_idx).astype(int)
                    arr[indices] = valid[feat].values.astype(np.float32)
            # qlib format: prepend start_index
            bin_data = np.hstack([np.float32(start_idx), arr]).astype("<f")
            bin_data.tofile(os.path.join(stock_dir, f"{feat}.day.bin"))

        start_date = df.loc[df["_idx"] == start_idx, "date"].iloc[0]
        end_date = df.loc[df["_idx"] == end_idx, "date"].iloc[0]
        return (symbol.upper(), start_date, end_date)

    except Exception as e:
        return None


def main():
    t0 = time.time()

    calendar = build_calendar()
    date_to_idx = {d: i for i, d in enumerate(calendar)}

    print("[2/4] Writing calendar...")
    write_calendar(calendar)

    print("[3/4] Processing stocks and writing bin files...")
    csv_files = sorted(glob.glob(os.path.join(NORMALIZE_DIR, "*.csv")))
    print(f"  {len(csv_files)} stocks to process")

    instruments = []
    done = 0
    errors = 0
    args_list = [(f, date_to_idx, len(calendar)) for f in csv_files]

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(process_stock, a): a[0] for a in args_list}
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result:
                instruments.append(result)
            else:
                errors += 1
            if done % 500 == 0 or done == len(csv_files):
                elapsed = time.time() - t0
                print(f"  Processed {done}/{len(csv_files)} stocks ({elapsed:.1f}s)")

    print(f"[4/4] Writing instruments file ({len(instruments)} stocks)...")
    instruments.sort(key=lambda x: x[0])
    os.makedirs(os.path.dirname(INSTRUMENTS_FILE), exist_ok=True)
    with open(INSTRUMENTS_FILE, "w") as f:
        for sym, start, end in instruments:
            f.write(f"{sym}\t{start}\t{end}\n")

    elapsed = time.time() - t0
    print(f"\nDone! {len(instruments)} stocks processed, {errors} skipped, {elapsed:.1f}s total")
    print(f"Calendar: {calendar[0]} ~ {calendar[-1]} ({len(calendar)} days)")
    print(f"Qlib dir: {QLIB_DIR}")


if __name__ == "__main__":
    main()
