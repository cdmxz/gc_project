"""训练脚本：PyTorch + TensorBoard，支持多组对照实验。

示例：
    python src/train.py --model resnet18 --epochs 20 --lr 3e-4 --exp-name E3_resnet18_lr3e-4_aug
    python src/train.py --model resnet18 --epochs 20 --lr 1e-3 --exp-name E2_resnet18_lr1e-3_aug
    python src/train.py --model smallcnn --epochs 20 --lr 1e-3 --no-aug --scratch --exp-name E1_smallcnn_scratch_lr1e-3

训练过程记录：Loss/train、Loss/val、Accuracy/train、Accuracy/val、F1_macro/val、LearningRate。
最佳权重按验证集 Macro-F1 选择，保存为 models/experiments/<exp-name>_best.pth。
"""
import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter

import config
from data import build_loaders
from model import build_model, count_parameters, set_backbone_frozen


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr, weight_decay):
    # 包含全部参数：冻结阶段这些参数没有梯度，优化器自动跳过；
    # 解冻后无需重建优化器即可参与更新（否则微调时主干永远不会被更新）
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.item() * images.size(0)
        all_preds.append(logits.detach().argmax(1).cpu())
        all_labels.append(labels.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return (total_loss / len(loader.dataset),
            accuracy_score(labels, preds),
            f1_score(labels, preds, average="macro"))


def main():
    ap = argparse.ArgumentParser(description="垃圾分类图像分类训练")
    ap.add_argument("--model", default="resnet18", choices=["smallcnn", "resnet18", "mobilenet_v3"])
    ap.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--weight-decay", type=float, default=config.WEIGHT_DECAY)
    ap.add_argument("--label-smoothing", type=float, default=config.LABEL_SMOOTHING)
    ap.add_argument("--scheduler", default="cosine", choices=["cosine", "step", "none"])
    ap.add_argument("--freeze-epochs", type=int, default=0, help="前 N 个 Epoch 冻结主干只训练分类头")
    ap.add_argument("--aug", dest="augment", action="store_true", default=True)
    ap.add_argument("--no-aug", dest="augment", action="store_false")
    ap.add_argument("--scratch", action="store_true", help="不使用 ImageNet 预训练权重")
    ap.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--exp-name", default=None)
    args = ap.parse_args()

    exp_name = args.exp_name or f"{args.model}_lr{args.lr:g}_{'aug' if args.augment else 'noaug'}"
    log_dir = config.RUNS_DIR / exp_name
    ckpt_dir = config.MODELS_DIR / "experiments"
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device} torch={torch.__version__} 实验={exp_name}")

    train_loader, val_loader, _ = build_loaders(args.batch_size, args.workers, augment=args.augment)
    print(f"[数据] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"batch={args.batch_size} steps/epoch={len(train_loader)}")

    model = build_model(args.model, config.NUM_CLASSES, pretrained=not args.scratch).to(device)
    print(f"[模型] {args.model} 参数量={count_parameters(model):,} "
          f"可训练={count_parameters(model, True):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    if args.freeze_epochs > 0:
        set_backbone_frozen(model, args.model, True)
        print(f"[迁移学习] 前 {args.freeze_epochs} 个 Epoch 冻结主干，仅训练分类头")

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    elif args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=max(1, args.epochs // 3), gamma=0.3)
    else:
        scheduler = None
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    config_dict = {
        "exp_name": exp_name, "model": args.model, "epochs": args.epochs, "lr": args.lr,
        "batch_size": args.batch_size, "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing, "scheduler": args.scheduler,
        "freeze_epochs": args.freeze_epochs, "augment": args.augment,
        "pretrained": not args.scratch, "seed": args.seed, "img_size": config.IMG_SIZE,
        "optimizer": "AdamW", "loss": "CrossEntropyLoss",
        "num_classes": config.NUM_CLASSES, "class_names": config.CLASS_NAMES,
        "device": str(device), "torch": torch.__version__,
        "params": count_parameters(model), "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
    }
    (log_dir / "config.json").write_text(json.dumps(config_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    writer = SummaryWriter(str(log_dir))
    writer.add_text("config", "```json\n" + json.dumps(config_dict, ensure_ascii=False, indent=2) + "\n```", 0)

    history_path = log_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "epoch_time_s"])

    best_f1, best_acc, best_epoch = -1.0, 0.0, -1
    best_path = ckpt_dir / f"{exp_name}_best.pth"
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # 冻结阶段结束后解冻主干，并用更小的学习率微调
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs + 1:
            set_backbone_frozen(model, args.model, False)
            for g in optimizer.param_groups:
                g["lr"] = args.lr * 0.1
            print(f"[迁移学习] Epoch {epoch}: 解冻主干，学习率降为 {args.lr * 0.1:g}")

        current_lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer, scaler)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, device)

        if scheduler is not None and epoch > args.freeze_epochs:
            scheduler.step()

        dt = time.time() - t0
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("F1_macro/val", val_f1, epoch)
        writer.add_scalar("LearningRate", current_lr, epoch)
        writer.flush()

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, current_lr, f"{train_loss:.6f}", f"{train_acc:.6f}",
                                    f"{val_loss:.6f}", f"{val_acc:.6f}", f"{val_f1:.6f}", f"{dt:.1f}"])

        print(f"Epoch {epoch:3d}/{args.epochs} lr={current_lr:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} ({dt:.1f}s)")

        if val_f1 > best_f1:
            best_f1, best_acc, best_epoch = val_f1, val_acc, epoch
            torch.save({
                "state_dict": model.state_dict(),
                "model_name": args.model,
                "num_classes": config.NUM_CLASSES,
                "class_names": config.CLASS_NAMES,
                "img_size": config.IMG_SIZE,
                "mean": config.MEAN,
                "std": config.STD,
                "epoch": epoch,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "exp_name": exp_name,
            }, best_path)
            print(f"    ↳ 新的最佳模型（val_f1={best_f1:.4f}, val_acc={val_acc:.4f}），已保存 {best_path.name}")

    total_time = time.time() - t_start
    writer.add_hparams(
        {"model": args.model, "lr": args.lr, "batch_size": args.batch_size,
         "augment": int(args.augment), "epochs": args.epochs},
        {"hparam/val_acc": val_acc, "hparam/val_f1": best_f1},
    )
    writer.close()

    summary = {"exp_name": exp_name, "best_epoch": best_epoch, "best_val_f1": best_f1,
               "best_val_acc": best_acc, "last_val_acc": val_acc, "total_time_s": round(total_time, 1),
               "checkpoint": str(best_path.relative_to(config.PROJECT_ROOT))}
    (log_dir / "result.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] 最佳 Epoch={best_epoch} val_f1={best_f1:.4f} 用时={total_time / 60:.1f} min")
    print(f"[日志] {log_dir}")
    print(f"[权重] {best_path}")


if __name__ == "__main__":
    main()
