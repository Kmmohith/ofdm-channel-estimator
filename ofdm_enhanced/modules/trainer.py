"""
trainer.py
Training pipeline for deep learning channel estimators.

Features
--------
- Adam optimizer with CosineAnnealingLR scheduler
- MSE loss (normalized)
- Train/validation tracking
- Best-model checkpointing
- GPU-ready (auto device detection)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Generic trainer for PyTorch channel estimation models.

    Parameters
    ----------
    model      : nn.Module
    lr         : learning rate (default 1e-3)
    batch_size : mini-batch size
    device     : 'cpu' or 'cuda' (auto-detected if None)
    save_path  : where to checkpoint the best model
    """

    def __init__(self, model: nn.Module,
                 lr: float = 1e-3,
                 batch_size: int = 32,
                 device: str = None,
                 save_path: str = "models_saved/best_model.pth"):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.model      = model.to(device)
        self.lr         = lr
        self.batch_size = batch_size
        self.save_path  = save_path

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=1e-5)
        # Smooth cosine decay from lr → lr/100 over training
        self.scheduler = None       # set in train()

        self.train_losses: list = []
        self.val_losses:   list = []
        self.best_val     = float("inf")

    # ---------------------------------------------------------------- #
    #  Main training loop                                               #
    # ---------------------------------------------------------------- #

    def train(self, train_ds, val_ds,
              epochs: int = 50,
              progress_callback=None):
        """
        Run the full training loop.

        Parameters
        ----------
        train_ds          : PyTorch Dataset (training subset)
        val_ds            : PyTorch Dataset (validation subset)
        epochs            : number of training epochs
        progress_callback : optional callable(epoch, train_loss, val_loss)
                            called after each epoch (e.g. for Streamlit progress)

        Returns
        -------
        (train_losses, val_losses) : lists of per-epoch losses
        """
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True,  num_workers=0)
        val_loader   = DataLoader(val_ds,   batch_size=self.batch_size,
                                  shuffle=False, num_workers=0)

        # Scheduler spans the full training run
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=self.lr / 100
        )

        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

        print(f"Training on {self.device}  |  "
              f"{len(train_ds)} train / {len(val_ds)} val  |  "
              f"epochs={epochs}  batch={self.batch_size}")

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            tr_loss  = self._run_epoch(train_loader, training=True)
            val_loss = self._run_epoch(val_loader,   training=False)

            self.scheduler.step()
            self.train_losses.append(tr_loss)
            self.val_losses.append(val_loss)

            # Checkpoint best model
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.save(self.save_path)
                tag = " ✓"
            else:
                tag = ""

            print(f"Epoch {epoch:3d}/{epochs}  "
                  f"train={tr_loss:.5f}  val={val_loss:.5f}"
                  f"  lr={self._lr():.2e}{tag}")

            if progress_callback:
                progress_callback(epoch, tr_loss, val_loss)

        elapsed = time.time() - t0
        print(f"Training complete in {elapsed:.1f}s  "
              f"| best val loss = {self.best_val:.6f}")

        return self.train_losses, self.val_losses

    # ---------------------------------------------------------------- #
    #  Internal                                                         #
    # ---------------------------------------------------------------- #

    def _run_epoch(self, loader: DataLoader, training: bool) -> float:
        self.model.train(training)
        total = 0.0
        ctx   = torch.enable_grad() if training else torch.no_grad()

        with ctx:
            for X, Y in loader:
                X, Y = X.to(self.device), Y.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                pred = self.model(X)
                loss = self.criterion(pred, Y)

                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total += loss.item() * len(X)

        return total / len(loader.dataset)

    def _lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    # ---------------------------------------------------------------- #
    #  Save / Load                                                      #
    # ---------------------------------------------------------------- #

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model_state":   self.model.state_dict(),
            "train_losses":  self.train_losses,
            "val_losses":    self.val_losses,
            "best_val":      self.best_val,
        }, path)

    @staticmethod
    def load_model(model: nn.Module, path: str, device: str = "cpu"):
        """Load model weights from checkpoint."""
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        return model, ckpt.get("train_losses", []), ckpt.get("val_losses", [])
