from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Tuple

import mlflow as mlf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from loguru import logger
from torch.utils.data import DataLoader

from mads_datasets import DatasetFactoryProvider, DatasetType
from mltrainer import ReportTypes, Trainer, TrainerSettings, metrics
from mltrainer.preprocessors import BasePreprocessor


# -----------------------------
# Data helpers: build DataLoaders from Fashion factory
# -----------------------------
def get_fashion_dataloaders(batchsize: int) -> tuple[DataLoader, DataLoader]:
    factory = DatasetFactoryProvider.create_factory(DatasetType.FASHION)
    preprocessor = BasePreprocessor()
    streamers = factory.create_datastreamer(batchsize=batchsize, preprocessor=preprocessor)

    train_obj = streamers["train"]
    valid_obj = streamers["valid"]

    # If already DataLoaders, return them
    if isinstance(train_obj, DataLoader) and isinstance(valid_obj, DataLoader):
        return train_obj, valid_obj

    # Else unwrap underlying datasets and wrap in DataLoader
    train_ds = getattr(train_obj, "dataset", None)
    valid_ds = getattr(valid_obj, "dataset", None)
    if train_ds is None or valid_ds is None:
        raise RuntimeError("Could not obtain .dataset from Fashion streamers to build DataLoaders.")

    traindl = DataLoader(train_ds, batch_size=batchsize, shuffle=True)
    validdl = DataLoader(valid_ds, batch_size=batchsize, shuffle=False)
    return traindl, validdl


def get_device() -> str:
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _to_py(v):
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return v


def _infer_input_and_classes_from_loader(loader: DataLoader) -> tuple[Tuple[int, int, int], int]:
    x, y = next(iter(loader))
    if x.ndim == 4:
        C, H, W = int(x.shape[1]), int(x.shape[2]), int(x.shape[3])
    elif x.ndim == 3:
        C, H, W = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    else:
        side = int((x.shape[-1]) ** 0.5)
        C, H, W = 1, side, side
    n_classes = int(y.max().item()) + 1 if torch.is_tensor(y) else 10
    return (C, H, W), n_classes


# -----------------------------
# Model (with norm+dropout and optional conv blocks)
# -----------------------------
class CNN(nn.Module):
    def __init__(
        self,
        filters: int,
        units1: int,
        units2: int,
        input_shape: Tuple[int, int, int],
        n_classes: int,
        norm_type: str = "none",            # "none" | "batch" | "layer"
        dropout_p: float = 0.0,
        activation: str = "relu",           # "relu" | "gelu" | "tanh" | "leaky_relu"
        num_conv_blocks: int = 0,
        conv_out_channels: Optional[int] = None,
        conv_kernel_size: int = 3,
        pool_type: str = "max",             # "max" | "avg" | "none"
        pool_kernel_size: int = 2,
    ):
        super().__init__()
        C, H, W = input_shape
        self.input_shape = input_shape

        act_cls = {
            "relu": nn.ReLU,
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
            "leaky_relu": nn.LeakyReLU,
        }.get(activation.lower(), nn.ReLU)

        # Conv stack
        if num_conv_blocks == 0:
            conv_layers = []
            conv_layers.append(nn.Conv2d(C, filters, kernel_size=3, stride=1, padding=1))
            if norm_type == "batch":
                conv_layers.append(nn.BatchNorm2d(filters))  # norm BEFORE activation
            conv_layers.append(act_cls())
            if dropout_p > 0.0:
                conv_layers.append(nn.Dropout2d(dropout_p))  # dropout AFTER activation
            conv_layers.append(nn.MaxPool2d(kernel_size=2))

            conv_layers.append(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0))
            if norm_type == "batch":
                conv_layers.append(nn.BatchNorm2d(filters))
            conv_layers.append(act_cls())
            if dropout_p > 0.0:
                conv_layers.append(nn.Dropout2d(dropout_p))
            conv_layers.append(nn.MaxPool2d(kernel_size=2))

            conv_layers.append(nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=0))
            if norm_type == "batch":
                conv_layers.append(nn.BatchNorm2d(filters))
            conv_layers.append(act_cls())
            if dropout_p > 0.0:
                conv_layers.append(nn.Dropout2d(dropout_p))
            conv_layers.append(nn.MaxPool2d(kernel_size=2))

            self.convolutions = nn.Sequential(*conv_layers)
            self.conv_blocks = None
        else:
            out_ch = conv_out_channels if conv_out_channels is not None else filters
            self.conv_blocks = nn.ModuleList()
            in_ch = C
            for _ in range(num_conv_blocks):
                block = []
                block.append(nn.Conv2d(in_ch, out_ch, kernel_size=conv_kernel_size,
                                       padding=conv_kernel_size // 2))
                if norm_type == "batch":
                    block.append(nn.BatchNorm2d(out_ch))
                block.append(act_cls())
                if dropout_p > 0.0:
                    block.append(nn.Dropout2d(dropout_p))
                if pool_type == "max":
                    block.append(nn.MaxPool2d(kernel_size=pool_kernel_size))
                elif pool_type == "avg":
                    block.append(nn.AvgPool2d(kernel_size=pool_kernel_size))
                self.conv_blocks.append(nn.Sequential(*block))
                in_ch = out_ch
            self.convolutions = None

        # Flatten dim via dummy
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            if self.conv_blocks is None:
                z = self.convolutions(dummy)
            else:
                z = dummy
                for blk in self.conv_blocks:
                    z = blk(z)
            fdim = z.view(1, -1).shape[1]

        # Dense stack
        fc = [nn.Flatten(), nn.Linear(fdim, units1)]
        if norm_type == "batch":
            fc.append(nn.BatchNorm1d(units1))
        elif norm_type == "layer":
            fc.append(nn.LayerNorm(units1))
        fc.append(act_cls())
        if dropout_p > 0.0:
            fc.append(nn.Dropout(dropout_p))

        fc.append(nn.Linear(units1, units2))
        if norm_type == "batch":
            fc.append(nn.BatchNorm1d(units2))
        elif norm_type == "layer":
            fc.append(nn.LayerNorm(units2))
        fc.append(act_cls())
        if dropout_p > 0.0:
            fc.append(nn.Dropout(dropout_p))

        fc.append(nn.Linear(units2, n_classes))
        self.dense = nn.Sequential(*fc)

    def forward(self, x):
        if x.ndim == 2:
            B = x.shape[0]
            x = x.view(B, *self.input_shape)
        if self.conv_blocks is None:
            x = self.convolutions(x)
        else:
            for blk in self.conv_blocks:
                x = blk(x)
        return self.dense(x)


# -----------------------------
# Objective (one trial)
# -----------------------------
def objective(hparams):
    device_str = get_device()
    device = torch.device(device_str)

    # DataLoaders (required by your Trainer)
    traindl, validdl = get_fashion_dataloaders(batchsize=int(hparams["batchsize"]))
    input_shape, n_classes = _infer_input_and_classes_from_loader(traindl)

    # Model
    model = CNN(
        filters=int(hparams["filters"]),
        units1=int(hparams["units1"]),
        units2=int(hparams["units2"]),
        input_shape=input_shape,
        n_classes=n_classes,
        norm_type=hparams.get("norm_type", "none"),
        dropout_p=float(_to_py(hparams.get("dropout_p", 0.0))),
        activation=hparams.get("activation", "relu"),
        num_conv_blocks=int(hparams.get("num_conv_blocks", 0)),
        conv_out_channels=int(hparams.get("conv_out_channels", hparams["filters"])),
        conv_kernel_size=int(hparams.get("conv_kernel_size", 3)),
        pool_type=hparams.get("pool_type", "max"),
        pool_kernel_size=int(hparams.get("pool_kernel_size", 2)),
    ).to(device)

    # Optimizer/scheduler factories that accept **kwargs (Trainer passes lr, etc.)
    opt_name = hparams.get("optimizer", "adam").lower()
    lr = float(_to_py(hparams["lr"]))
    wd = float(_to_py(hparams.get("weight_decay", 0.0)))

    def optimizer_factory(params, **kwargs):
        lr_local = kwargs.get("lr", lr)
        wd_local = kwargs.get("weight_decay", wd)
        if opt_name == "adamw":
            return optim.AdamW(params, lr=lr_local, weight_decay=wd_local)
        elif opt_name == "sgd":
            momentum_local = kwargs.get("momentum", 0.9)
            return optim.SGD(params, lr=lr_local, momentum=momentum_local, weight_decay=wd_local)
        else:
            return optim.Adam(params, lr=lr_local, weight_decay=wd_local)

    def scheduler_factory(opt, **kwargs):
        step_size = kwargs.get("step_size", 2)
        gamma = kwargs.get("gamma", 0.8)
        return optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=gamma)

    criterion = nn.CrossEntropyLoss()

    # TrainerSettings (epochs + logdir REQUIRED)
    logdir = Path(__file__).parent / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir.mkdir(parents=True, exist_ok=True)

    settings = TrainerSettings(
        epochs=int(hparams["epochs"]),
        batch_size=int(hparams["batchsize"]),
        train_steps=100,
        valid_steps=100,
        logdir=str(logdir),
        reporttypes=[ReportTypes.MLFLOW],
        metrics=[metrics.Accuracy()],
        device=device_str,
    )

    with mlf.start_run(nested=True):
        mlf.set_tag("script", "03_mlflow.py")
        mlf.set_tag("dataset", "FASHION")

        mlf.log_params({
            "batchsize": int(hparams["batchsize"]),
            "epochs": int(hparams["epochs"]),
            "optimizer": opt_name,
            "lr": lr,
            "weight_decay": wd,
            "filters": int(hparams["filters"]),
            "units1": int(hparams["units1"]),
            "units2": int(hparams["units2"]),
            "activation": hparams.get("activation", "relu"),
            "norm_type": hparams.get("norm_type", "none"),
            "dropout_p": float(_to_py(hparams.get("dropout_p", 0.0))),
            "num_conv_blocks": int(hparams.get("num_conv_blocks", 0)),
            "conv_out_channels": int(hparams.get("conv_out_channels", hparams["filters"])),
            "conv_kernel_size": int(hparams.get("conv_kernel_size", 3)),
            "pool_type": hparams.get("pool_type", "max"),
            "pool_kernel_size": int(hparams.get("pool_kernel_size", 2)),
        })

        # Build Trainer with loaders + factories
        trainer = Trainer(
            model=model,
            optimizer=optimizer_factory,      # callable
            loss_fn=criterion,
            traindataloader=traindl,
            validdataloader=validdl,
            scheduler=scheduler_factory,      # callable
            settings=settings,
        )

        # Compatibility shim: fit() → train() → run()
        try:
            history = trainer.fit()
        except AttributeError:
            try:
                history = trainer.train()
            except AttributeError:
                history = trainer.run()

        val_acc = float(history.get("val_accuracy_best", history.get("val_accuracy_last", 0.0)))
        mlf.log_metric("val_accuracy", val_acc)

        loss = -val_acc if val_acc == val_acc else 1e9
        return {"status": STATUS_OK, "loss": loss, "val_acc": val_acc}


# -----------------------------
# Search space
# -----------------------------
search_space = {
    "optimizer": hp.choice("optimizer", ["adam", "adamw", "sgd"]),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(5e-2)),
    "weight_decay": hp.loguniform("weight_decay", np.log(1e-6), np.log(1e-2)),
    "batchsize": scope.int(hp.quniform("batchsize", 32, 256, 32)),
    "epochs": scope.int(hp.quniform("epochs", 5, 12, 1)),
    "filters": scope.int(hp.quniform("filters", 16, 128, 8)),
    "units1": scope.int(hp.quniform("units1", 32, 256, 16)),
    "units2": scope.int(hp.quniform("units2", 32, 256, 16)),
    "norm_type": hp.choice("norm_type", ["none", "batch", "layer"]),
    "dropout_p": hp.uniform("dropout_p", 0.0, 0.6),
    "activation": hp.choice("activation", ["relu", "gelu", "tanh", "leaky_relu"]),
    "num_conv_blocks": scope.int(hp.quniform("num_conv_blocks", 0, 3, 1)),
    "conv_out_channels": scope.int(hp.quniform("conv_out_channels", 16, 128, 16)),
    "conv_kernel_size": scope.int(hp.quniform("conv_kernel_size", 3, 5, 2)),
    "pool_type": hp.choice("pool_type", ["max", "avg", "none"]),
    "pool_kernel_size": hp.choice("pool_kernel_size", [2]),
}


# -----------------------------
# Main
# -----------------------------
def main():
    mlf.set_experiment("03_mlflow_with_norm_dropout_conv")
    try:
        import mlflow.pytorch
        mlf.pytorch.autolog(log_models=False)
    except Exception:
        pass

    best_result = fmin(
        fn=objective, space=search_space, algo=tpe.suggest, max_evals=6, trials=Trials()
    )
    logger.info(f"Best result: {best_result}")


if __name__ == "__main__":
    main()