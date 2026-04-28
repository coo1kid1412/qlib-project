#!/usr/bin/env python3
"""
HIST 模型训练与评估（带 NaN 修复）
===================================
功能：
1. 训练 HIST 模型（基于 qlib contrib）
2. 在测试集上评估 IC
3. 保存最佳模型权重

配置：
- 市场: csi500（可改 csi300）
- 训练期: 2008-01-01 ~ 2024-12-31
- 验证期: 2025-01-01 ~ 2025-12-31
- 测试期: 2026-01-01 ~ 最新
- 归一化基准: 2008-01-01（确保 RobustZScoreNorm 一致性）
"""
import os
import sys
import json
import pickle
import argparse
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Text, Union
import urllib.request

import torch
import torch.nn as nn
import torch.optim as optim

import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.model.base import Model
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.contrib.data.handler import Alpha360
from qlib.contrib.model.pytorch_hist import HISTModel
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.utils import get_or_create_path, init_instance_by_config
from qlib.log import get_module_logger

# ============================================================
# Configuration
# ============================================================
QLIB_PROVIDER = os.path.expanduser('~/.qlib/qlib_data/cn_data_yahoo')
HIST_DATA_DIR = os.path.join(QLIB_PROVIDER, 'hist_data')
MARKET = 'csi500'
DEFAULT_STOCK_INDEX = 1777
FIT_START = '2008-01-01'
REPORT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'hist_report')

# Training config
TRAIN_START = '2008-01-01'
TRAIN_END = '2024-12-31'
VALID_START = '2025-01-01'
VALID_END = '2025-12-31'
TEST_START = '2026-01-01'

N_EPOCHS = 100
LR = 0.0001
EARLY_STOP = 15
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.0
D_FEAT = 5  # Alpha360: OHLCV (5 features × time_window)


# ============================================================
# Custom HIST with NaN fixes
# ============================================================
class CustomHIST(Model):
    """HIST model with division-by-zero protection in metric_fn."""

    def __init__(
        self,
        d_feat=D_FEAT,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        n_epochs=N_EPOCHS,
        lr=LR,
        metric="ic",
        early_stop=EARLY_STOP,
        loss="mse",
        base_model="GRU",
        model_path=None,
        stock2concept=None,
        stock_index=None,
        optimizer="adam",
        GPU=-1,
        seed=None,
        default_stock_index=DEFAULT_STOCK_INDEX,
        resume_path=None,
        warmup_path=None,
    ):
        self.logger = get_module_logger("HIST")
        self.logger.info("HIST pytorch version...")

        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.lr = lr
        self.metric = metric
        self.early_stop = early_stop
        self.optimizer = optimizer.lower()
        self.loss = loss
        self.base_model = base_model
        self.model_path = model_path
        self.stock2concept = stock2concept
        self.stock_index = stock_index
        self.device = torch.device("cuda:%d" % (GPU) if torch.cuda.is_available() and GPU >= 0 else "cpu")
        self.seed = seed
        self._default_stock_index = default_stock_index
        self._resume_path = resume_path
        self._warmup_path = warmup_path

        self.logger.info("HIST parameters setting:\n" +
            "\nd_feat : {}\nhidden_size : {}\nnum_layers : {}\ndropout : {}\n"
            "n_epochs : {}\nlr : {}\nmetric : {}\nearly_stop : {}\noptimizer : {}\n"
            "loss_type : {}\nbase_model : {}\nmodel_path : {}\nstock2concept : {}\n"
            "stock_index : {}\nuse_GPU : {}\nseed : {}".format(
                d_feat, hidden_size, num_layers, dropout, n_epochs, lr, metric,
                early_stop, optimizer.lower(), loss, base_model, model_path,
                stock2concept, stock_index, GPU, seed))

        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.HIST_model = HISTModel(
            d_feat=self.d_feat, hidden_size=self.hidden_size,
            num_layers=self.num_layers, dropout=self.dropout, base_model=self.base_model)
        self.logger.info("model:\n{:}".format(self.HIST_model))
        self.logger.info("model size: {:.4f} MB".format(
            sum(p.numel() for p in self.HIST_model.parameters()) * 4 / 1024 / 1024))

        if optimizer.lower() == "adam":
            self.train_optimizer = optim.Adam(self.HIST_model.parameters(), lr=self.lr)
        elif optimizer.lower() == "gd":
            self.train_optimizer = optim.SGD(self.HIST_model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"optimizer {optimizer} is not supported!")

        self.fitted = False
        self.HIST_model.to(self.device)

        self.logger.info(f"CustomHIST: default_stock_index={default_stock_index}, "
                         f"resume_path={resume_path}, warmup_path={warmup_path}")

    @property
    def use_gpu(self):
        return self.device != torch.device("cpu")

    def mse(self, pred, label):
        loss = (pred - label) ** 2
        return torch.mean(loss)

    def loss_fn(self, pred, label):
        mask = ~torch.isnan(label)
        if self.loss == "mse":
            return self.mse(pred[mask], label[mask])
        raise ValueError("unknown loss `%s`" % self.loss)

    def metric_fn(self, pred, label):
        """IC (Pearson correlation) with division-by-zero protection."""
        mask = torch.isfinite(label)
        if self.metric == "ic":
            x = pred[mask]
            y = label[mask]
            vx = x - torch.mean(x)
            vy = y - torch.mean(y)
            denom = torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2))
            eps = 1e-12  # FIX: prevent division by zero
            return torch.sum(vx * vy) / (denom + eps)
        if self.metric in ("", "loss"):
            return -self.loss_fn(pred[mask], label[mask])
        raise ValueError("unknown metric `%s`" % self.metric)

    def get_daily_inter(self, df, shuffle=False):
        daily_count = df.groupby(level=0, group_keys=False).size().values
        daily_index = np.roll(np.cumsum(daily_count), 1)
        daily_index[0] = 0
        if shuffle:
            daily_shuffle = list(zip(daily_index, daily_count))
            np.random.shuffle(daily_shuffle)
            daily_index, daily_count = zip(*daily_shuffle)
        return daily_index, daily_count

    def train_epoch(self, x_train, y_train, stock_index):
        stock2concept_matrix = np.load(self.stock2concept)
        x_train_values = x_train.values
        y_train_values = np.squeeze(y_train.values)
        stock_index_vals = stock_index.values
        stock_index_vals[np.isnan(stock_index_vals)] = 733
        self.HIST_model.train()

        daily_index, daily_count = self.get_daily_inter(x_train, shuffle=True)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_train_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index_vals[batch]]).float().to(self.device)
            label = torch.from_numpy(y_train_values[batch]).float().to(self.device)
            pred = self.HIST_model(feature, concept_matrix)
            loss = self.loss_fn(pred, label)

            self.train_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.HIST_model.parameters(), 3.0)
            self.train_optimizer.step()

    def test_epoch(self, data_x, data_y, stock_index):
        stock2concept_matrix = np.load(self.stock2concept)
        x_values = data_x.values
        y_values = np.squeeze(data_y.values)
        stock_index_vals = stock_index.values
        stock_index_vals[np.isnan(stock_index_vals)] = 733
        self.HIST_model.eval()

        scores = []
        losses = []
        daily_index, daily_count = self.get_daily_inter(data_x, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            feature = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index_vals[batch]]).float().to(self.device)
            label = torch.from_numpy(y_values[batch]).float().to(self.device)
            with torch.no_grad():
                pred = self.HIST_model(feature, concept_matrix)
                loss = self.loss_fn(pred, label)
                losses.append(loss.item())
                score = self.metric_fn(pred, label)
                scores.append(score.item())

        return np.mean(losses), np.nanmean(scores)  # FIX: nanmean prevents NaN propagation

    def fit(self, dataset: DatasetH, evals_result=dict(), save_path=None):
        df_train, df_valid, df_test = dataset.prepare(
            ["train", "valid", "test"],
            col_set=["feature", "label"],
            data_key=DataHandlerLP.DK_L,
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("Empty data from dataset, please check your dataset config.")

        if not os.path.exists(self.stock2concept):
            url = "https://github.com/SunsetWolf/qlib_dataset/releases/download/v0/qlib_csi300_stock2concept.npy"
            urllib.request.urlretrieve(url, self.stock2concept)

        stock_index_map = np.load(self.stock_index, allow_pickle=True).item()
        df_train["stock_index"] = 733
        df_train["stock_index"] = df_train.index.get_level_values("instrument").map(stock_index_map)
        df_valid["stock_index"] = 733
        df_valid["stock_index"] = df_valid.index.get_level_values("instrument").map(stock_index_map)

        x_train, y_train, stock_index_train = df_train["feature"], df_train["label"], df_train["stock_index"]
        x_valid, y_valid, stock_index_valid = df_valid["feature"], df_valid["label"], df_valid["stock_index"]

        save_path = get_or_create_path(save_path)
        stop_steps = 0
        best_score = -np.inf
        best_epoch = 0
        evals_result["train"] = []
        evals_result["valid"] = []

        # Load pretrained base model
        if self.base_model == "LSTM":
            pretrained_model = LSTMModel(d_feat=self.d_feat, hidden_size=self.hidden_size,
                                          num_layers=self.num_layers, dropout=self.dropout)
        elif self.base_model == "GRU":
            pretrained_model = GRUModel(d_feat=self.d_feat, hidden_size=self.hidden_size,
                                         num_layers=self.num_layers, dropout=self.dropout)
        else:
            raise ValueError(f"unknown base model name `{self.base_model}`")

        model_dict = self.HIST_model.state_dict()
        pretrained_dict = {
            k: v for k, v in pretrained_model.state_dict().items() if k in model_dict
        }
        if pretrained_dict:
            model_dict.update(pretrained_dict)
            self.HIST_model.load_state_dict(model_dict)
            self.logger.info("Loading pretrained base model Done...")

        # Resume from checkpoint
        if self._resume_path and os.path.exists(self._resume_path):
            self.logger.info(f"Resuming from {self._resume_path}")
            checkpoint = torch.load(self._resume_path, map_location=self.device)
            self.HIST_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
            if 'optimizer_state_dict' in checkpoint:
                self.train_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'epoch' in checkpoint:
                self.logger.info(f"Resumed from epoch {checkpoint['epoch']}")

        # Warmup from another model
        if self._warmup_path and os.path.exists(self._warmup_path):
            self.logger.info(f"Warming up from {self._warmup_path}")
            warmup_state = torch.load(self._warmup_path, map_location=self.device)
            self.HIST_model.load_state_dict(warmup_state)

        # Train
        self.logger.info("training...")
        self.fitted = True

        for step in range(self.n_epochs):
            self.logger.info("Epoch %d:", step)
            self.logger.info("training...")
            self.train_epoch(x_train, y_train, stock_index_train)

            self.logger.info("evaluating...")
            train_loss, train_score = self.test_epoch(x_train, y_train, stock_index_train)
            val_loss, val_score = self.test_epoch(x_valid, y_valid, stock_index_valid)
            self.logger.info("train %.6f, valid %.6f" % (train_score, val_score))
            evals_result["train"].append(train_score)
            evals_result["valid"].append(val_score)

            if val_score > best_score:
                best_score = val_score
                stop_steps = 0
                best_epoch = step
                best_param = copy.deepcopy(self.HIST_model.state_dict())
            else:
                stop_steps += 1
                if stop_steps >= self.early_stop:
                    self.logger.info("early stop")
                    break

        self.logger.info("best score: %.6lf @ %d" % (best_score, best_epoch))
        self.HIST_model.load_state_dict(best_param)
        torch.save(best_param, save_path)

        # Save checkpoint
        checkpoint_path = save_path.replace('.pt', '_checkpoint.ckpt')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_param,
            'optimizer_state_dict': self.train_optimizer.state_dict(),
            'best_score': best_score,
            'evals_result': evals_result,
        }, checkpoint_path)

        return best_epoch, best_score

    def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
        if not self.fitted:
            raise ValueError("model is not fitted yet!")

        stock2concept_matrix = np.load(self.stock2concept)
        stock_index_map = np.load(self.stock_index, allow_pickle=True).item()
        df_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        df_test["stock_index"] = 733
        df_test["stock_index"] = df_test.index.get_level_values("instrument").map(stock_index_map)
        stock_index_test = df_test["stock_index"].values
        stock_index_test[np.isnan(stock_index_test)] = 733
        stock_index_test = stock_index_test.astype("int")
        df_test = df_test.drop(["stock_index"], axis=1)
        index = df_test.index

        self.HIST_model.eval()
        x_values = df_test.values
        preds = []

        daily_index, daily_count = self.get_daily_inter(df_test, shuffle=False)

        for idx, count in zip(daily_index, daily_count):
            batch = slice(idx, idx + count)
            x_batch = torch.from_numpy(x_values[batch]).float().to(self.device)
            concept_matrix = torch.from_numpy(stock2concept_matrix[stock_index_test[batch]]).float().to(self.device)

            with torch.no_grad():
                pred = self.HIST_model(x_batch, concept_matrix).detach().cpu().numpy()

            preds.append(pred)

        return pd.Series(np.concatenate(preds), index=index)


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='HIST 模型训练')
    parser.add_argument('--market', type=str, default=MARKET, help='市场: csi500, csi300')
    parser.add_argument('--epochs', type=int, default=N_EPOCHS, help='训练轮数')
    parser.add_argument('--lr', type=float, default=LR, help='学习率')
    parser.add_argument('--early-stop', type=int, default=EARLY_STOP, help='早停轮数')
    parser.add_argument('--resume', type=str, default=None, help='从 checkpoint 恢复')
    parser.add_argument('--output', type=str, default=None, help='输出目录')
    parser.add_argument('--gpu', type=int, default=-1, help='GPU ID (-1=CPU)')
    args = parser.parse_args()

    market = args.market
    output_dir = args.output or REPORT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Init qlib
    qlib.init(provider_uri=QLIB_PROVIDER, region=REG_CN)

    # Prepare stock2concept and stock_index paths
    stock2concept_path = os.path.join(HIST_DATA_DIR, f'stock2concept_{market}.npy')
    stock_index_path = os.path.join(HIST_DATA_DIR, f'stock_index_{market}.npy')
    if not os.path.exists(stock2concept_path):
        stock2concept_path = os.path.join(HIST_DATA_DIR, 'stock2concept.npy')
        stock_index_path = os.path.join(HIST_DATA_DIR, 'stock_index.npy')

    # Build dataset
    print(f'\n[1/3] 构建数据集 (market={market})...')
    instruments = market if market in ('csi300', 'csi500', 'csi800', 'all') else 'all'

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
                    "fit_end_time": "2024-12-31",
                    "instruments": instruments,
                },
            },
            "segments": {
                "train": (TRAIN_START, TRAIN_END),
                "valid": (VALID_START, VALID_END),
                "test": (TEST_START, "2026-12-31"),
            },
        },
    }
    dataset = init_instance_by_config(dataset_config)

    # Create model
    print(f'[2/3] 创建模型...')
    model = CustomHIST(
        d_feat=D_FEAT,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        n_epochs=args.epochs,
        lr=args.lr,
        metric="ic",
        early_stop=args.early_stop,
        loss="mse",
        base_model="GRU",
        stock2concept=stock2concept_path,
        stock_index=stock_index_path,
        optimizer="adam",
        GPU=args.gpu,
        seed=42,
        default_stock_index=DEFAULT_STOCK_INDEX,
        resume_path=args.resume,
    )

    # Train
    save_path = os.path.join(output_dir, 'hist_model_best.pt')
    evals_result = {}
    best_epoch, best_score = model.fit(dataset, evals_result, save_path)

    print(f'\n[3/3] 评估测试集...')
    pred_series = model.predict(dataset, segment="test")

    # Save results
    pd.Series(evals_result['train']).to_csv(os.path.join(output_dir, 'train_ic.csv'))
    pd.Series(evals_result['valid']).to_csv(os.path.join(output_dir, 'valid_ic.csv'))
    pred_series.to_pickle(os.path.join(output_dir, 'predictions.pkl'))
    with open(os.path.join(output_dir, 'training_history.pkl'), 'wb') as f:
        pickle.dump(evals_result, f)

    print(f'\n=================================================================')
    print(f'  训练完成!')
    print(f'  最佳轮次: Epoch {best_epoch} | Valid IC: {best_score:.6f}')
    print(f'  模型: {save_path}')
    print(f'  预测: {os.path.join(output_dir, "predictions.pkl")}')
    print(f'=================================================================\n')

    return model


if __name__ == '__main__':
    main()
