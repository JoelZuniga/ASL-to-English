from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.config import MOTION_CLASSES, RANDOM_STATE, SEQUENCE_WINDOW
from src.ml.motion_model import FEATURE_SIZE, MotionLSTM
from src.ml.sequence_features import load_sequences_from_dir
from src.utils.paths import MODELS_DIR, SEQUENCES_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the bidirectional LSTM for J/Z motion recognition."
    )
    parser.add_argument("--input-dir", type=Path, default=SEQUENCES_DIR)
    parser.add_argument("--output-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def build_dataloaders(
    x: npt.NDArray[np.float32],
    y: npt.NDArray[np.int64],
    batch_size: int,
    val_fraction: float = 0.2,
) -> tuple[DataLoader[tuple[torch.Tensor, ...]], DataLoader[tuple[torch.Tensor, ...]]]:
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=val_fraction, stratify=y, random_state=RANDOM_STATE
    )
    train_ds: TensorDataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds: TensorDataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(
        val_ds, batch_size=batch_size
    )
    return train_loader, val_loader


def train_one_epoch(
    model: MotionLSTM,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * len(x_batch)
        n += len(x_batch)
    return total_loss / n


def evaluate(
    model: MotionLSTM,
    loader: DataLoader[tuple[torch.Tensor, ...]],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(x_batch)
            total_loss += float(criterion(logits, y_batch).item()) * len(x_batch)
            correct += int((logits.argmax(dim=1) == y_batch).sum().item())
            n += len(x_batch)
    return total_loss / n, correct / n


def save_checkpoint(
    model: MotionLSTM,
    label_to_int: dict[str, int],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "asl_motion_lstm.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_int": label_to_int,
            "config": {
                "hidden_size": model.lstm.hidden_size,
                "num_layers": model.lstm.num_layers,
                "bidirectional": model.lstm.bidirectional,
                "num_classes": cast(nn.Linear, model.classifier[-1]).out_features,
            },
        },
        path,
    )
    return path


def export_to_onnx(
    model: MotionLSTM,
    output_dir: Path,
    window: int,
    input_size: int,
) -> Path:
    model.eval()
    dummy = torch.zeros(1, window, input_size)
    path = output_dir / "asl_motion_lstm.onnx"

    torch.onnx.export(
        model,
        (dummy,),
        str(path),
        opset_version=14,
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={
            "sequence": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        do_constant_folding=True,
    )

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    ort_out: npt.NDArray[np.float32] = session.run(
        ["logits"], {"sequence": dummy.numpy()}
    )[0]
    assert np.allclose(pt_out, ort_out, atol=1e-4), "ONNX output diverges from PyTorch."
    print("  ONNX verification passed.")

    return path


def main() -> None:
    args = parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    label_to_int: dict[str, int] = {label: i for i, label in enumerate(MOTION_CLASSES)}
    print(f"Classes: {label_to_int}")

    print("Loading sequences...")
    x, y = load_sequences_from_dir(
        args.input_dir, MOTION_CLASSES, window=SEQUENCE_WINDOW
    )
    print(f"  Dataset: {x.shape}  class counts: {np.bincount(y)}")

    train_loader, val_loader = build_dataloaders(x, y, args.batch_size)

    model = MotionLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    patience_counter = 0
    patience = 20
    best_state: dict[str, torch.Tensor] = {}

    print(f"\nTraining up to {args.epochs} epochs (early-stop patience={patience})...")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}  "
                f"train={train_loss:.4f}  "
                f"val={val_loss:.4f}  "
                f"acc={val_acc:.2%}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    _, final_acc = evaluate(model, val_loader, criterion, device)
    print(f"\nFinal val accuracy: {final_acc:.2%}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = save_checkpoint(model, label_to_int, args.output_dir)
    print(f"Checkpoint: {ckpt_path}")

    labels_path = args.output_dir / "asl_motion_lstm_labels.json"
    labels_path.write_text(json.dumps(label_to_int))
    print(f"Label map: {labels_path}")

    onnx_path = export_to_onnx(
        model.to("cpu"),
        args.output_dir,
        window=SEQUENCE_WINDOW,
        input_size=FEATURE_SIZE,
    )
    print(f"ONNX model: {onnx_path}")


if __name__ == "__main__":
    main()
