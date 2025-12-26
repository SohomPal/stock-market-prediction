#!/usr/bin/env python
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

warnings.filterwarnings("ignore", message="The given NumPy array is not writable")


# ============================
# Dataset
# ============================
class LSTMDataset(Dataset):
    def __init__(self, root: Path, split: str):
        root = Path(root)
        self.X = np.load(root / f"{split}_X.npy", mmap_mode="r")
        self.y = np.load(root / f"{split}_y.npy", mmap_mode="r")
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


# ============================
# Model
# ============================
class LSTMForecast(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim=512,
        num_layers=4,
        dropout=0.2,
        lr=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.fc = nn.Linear(hidden_dim, 1)
        self.loss_fn = nn.HuberLoss(delta=1.0)

    def forward(self, x):
        out, _ = self.lstm(x)
        y_hat = self.fc(out[:, -1]).squeeze(-1)

        # Hard clamp (aligned with clipped log-return labels)
        return torch.clamp(y_hat, -1.0, 1.0)

    # ============================
    # Training
    # ============================
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_fn(y_hat, y)

        # Absolute safety
        if not torch.isfinite(loss):
            self.log("nan_train_batch", 1, on_step=True)
            return None

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    # ============================
    # Validation
    # ============================
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
    
        diff = torch.abs(y_hat - y)
    
        # 🔥 FINAL FIX: force numerical stability
        diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        mae = diff.mean()
    
        dir_true = (y > 0).long()
        dir_pred = (y_hat > 0).long()
        dir_acc = (dir_pred == dir_true).float().mean()
    
        self.log("val_mae", mae, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_dir_acc", dir_acc, on_epoch=True, prog_bar=True, sync_dist=True)
    
        return mae


    # ============================
    # GPU usage logging
    # ============================
    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
            self.log("gpu_peak_gb", peak, sync_dist=True)
            torch.cuda.reset_peak_memory_stats()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-4,
        )


# ============================
# Main
# ============================
def main():
    torch.set_float32_matmul_precision("high")

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--val_batch_size", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train_ds = LSTMDataset(data_dir, "train")
    val_ds = LSTMDataset(data_dir, "val")

    input_dim = train_ds[0][0].shape[-1]

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        drop_last=False,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    model = LSTMForecast(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
    )

    logger = CSVLogger(save_dir=str(outdir / "logs"), name="lstm_final")

    ckpt = ModelCheckpoint(
        dirpath=str(outdir / "checkpoints"),
        monitor="val_mae",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="lstm-{epoch:02d}-{val_mae:.6f}",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",   # safe now
        max_epochs=args.epochs,
        logger=logger,
        callbacks=[ckpt, LearningRateMonitor("epoch")],
        log_every_n_steps=50,
        gradient_clip_val=1.0,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
