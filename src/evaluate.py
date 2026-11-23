"""测试集评估：整体指标、每类精确率/召回率/F1、混淆矩阵、正确与错误案例。

用法：
    python src/evaluate.py --checkpoint models/best_model.pth
输出（默认写入 outputs/）：
    metrics.json、classification_report.txt、confusion_matrix.png、
    case_correct.png、case_wrong.png、predictions.csv
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support, top_k_accuracy_score)
from torch.utils.data import DataLoader

import config
from data import build_datasets
from model import build_model, count_parameters

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

CLASS_LABELS_ZH = [f"{config.CLASS_NAMES_ZH[c]}\n{c}" for c in config.CLASS_NAMES]


def load_checkpoint(path, device):
    """兼容两种保存格式：纯 state_dict 或带元信息的 dict。"""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        state, meta = obj["state_dict"], {k: v for k, v in obj.items() if k != "state_dict"}
    else:
        state, meta = obj, {}
    model_name = meta.get("model_name")
    if not model_name:
        meta_file = config.MODELS_DIR / "model_meta.json"
        if meta_file.exists():
            import json as _json
            model_name = _json.loads(meta_file.read_text(encoding="utf-8")).get("model_name")
    if not model_name:
        if any(k.startswith("features.0.0.") for k in state):
            model_name = "mobilenet_v3"
        elif "features.0.weight" in state:
            model_name = "smallcnn"
        else:
            model_name = "resnet18"
    model = build_model(model_name, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(state)
    return model.to(device).eval(), model_name, meta


@torch.no_grad()
def infer(model, loader, device):
    ys, ps, probs = [], [], []
    for images, labels in loader:
        logits = model(images.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu())
        ps.append(logits.argmax(1).cpu())
        ys.append(labels)
    return (torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy())


def plot_confusion(cm, out_path, title="测试集混淆矩阵"):
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=200)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASS_NAMES)))
    ax.set_yticks(range(len(config.CLASS_NAMES)))
    ax.set_xticklabels(CLASS_LABELS_ZH, fontsize=9)
    ax.set_yticklabels(CLASS_LABELS_ZH, fontsize=9)
    ax.set_xlabel("预测类别", fontsize=11)
    ax.set_ylabel("真实类别", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > thresh else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_cases(samples, out_path, title):
    n = len(samples)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows), dpi=170)
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, s in zip(axes, samples):
        from PIL import Image
        ax.imshow(Image.open(s["path"]).convert("RGB"))
        true_zh = config.CLASS_NAMES_ZH[config.CLASS_NAMES[s["true"]]]
        pred_zh = config.CLASS_NAMES_ZH[config.CLASS_NAMES[s["pred"]]]
        color = "#1a7f37" if s["true"] == s["pred"] else "#c62828"
        ax.set_title(f"真实: {true_zh}\n预测: {pred_zh} ({s['conf']:.2%})",
                     fontsize=9, color=color)
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(config.BEST_MODEL_PATH))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--output-dir", default=str(config.OUTPUTS_DIR))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_name, meta = load_checkpoint(args.checkpoint, device)
    print(f"[模型] {model_name} 参数量={count_parameters(model):,} 设备={device}")

    ds = {"train": 0, "val": 1, "test": 2}
    datasets = build_datasets()
    dataset = datasets[ds[args.split]]
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    y_true, y_pred, probs = infer(model, loader, device)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    top3 = top_k_accuracy_score(y_true, probs, k=3, labels=list(range(config.NUM_CLASSES)))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(config.NUM_CLASSES)), zero_division=0)

    print(f"[结果] {args.split} 样本数={len(y_true)} accuracy={acc:.4f} macro_f1={macro_f1:.4f} top3_acc={top3:.4f}")
    print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4, zero_division=0))

    metrics = {
        "checkpoint": str(args.checkpoint),
        "model": model_name,
        "split": args.split,
        "num_samples": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "top3_accuracy": float(top3),
        "per_class": {
            config.CLASS_NAMES[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i]),
            } for i in range(config.NUM_CLASSES)
        },
        "checkpoint_meta": {k: v for k, v in meta.items() if isinstance(v, (int, float, str, list))},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, target_names=config.CLASS_NAMES, digits=4, zero_division=0),
        encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(config.NUM_CLASSES)))
    plot_confusion(cm, out_dir / "confusion_matrix.png")

    # 单张图片推理耗时（含预处理，预热 5 次取 20 次均值）
    import time
    sample_img, _ = dataset[0]
    with torch.no_grad():
        for _ in range(5):
            model(sample_img.unsqueeze(0).to(device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            model(sample_img.unsqueeze(0).to(device))
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    metrics["inference_ms"] = round(float(np.mean(times)), 2)
    metrics["confusion_matrix"] = cm.tolist()
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # 逐样本记录，便于错误案例分析
    paths = [p for p, _ in dataset.samples]
    df = pd.DataFrame({
        "path": paths,
        "file_name": [Path(p).name for p in paths],
        "true": [config.CLASS_NAMES[i] for i in y_true],
        "pred": [config.CLASS_NAMES[i] for i in y_pred],
        "confidence": probs.max(1),
        "top3": ["|".join(f"{config.CLASS_NAMES[j]}:{probs[k, j]:.4f}"
                          for j in np.argsort(-probs[k])[:3]) for k in range(len(y_true))],
        "correct": y_true == y_pred,
    })
    df.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    # 合并案例图：6 个正确 + 2 个错误（4 列 × 2 行，边框颜色区分对错）
    pick_correct = df[df.correct].sort_values("confidence", ascending=False).head(6)
    pick_wrong = df[~df.correct].sort_values("confidence", ascending=False).head(2)
    merged = [(r, True) for r in pick_correct.itertuples()] + [(r, False) for r in pick_wrong.itertuples()]
    if merged:
        from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont
        cols, cell_w, cell_h = 4, 360, 320
        rows_n = 2
        canvas = _Image.new("RGB", (cols * cell_w, rows_n * cell_h + 46), "white")
        draw = _ImageDraw.Draw(canvas)
        try:
            f_title = _ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", 22)
            f_text = _ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 15)
        except Exception:
            f_title = f_text = _ImageFont.load_default()
        draw.text((14, 10), "测试集案例：绿色边框为正确预测，红色边框为错误预测", font=f_title, fill=(30, 40, 35))
        for i, (r, ok) in enumerate(merged):
            row, col = divmod(i, cols)
            x0, y0 = col * cell_w, 46 + row * cell_h
            im = _Image.open(r.path).convert("RGB")
            im.thumbnail((cell_w - 24, cell_h - 62))
            canvas.paste(im, (x0 + (cell_w - im.width) // 2, y0 + 6))
            color = (26, 127, 55) if ok else (198, 40, 40)
            draw.rectangle([x0 + 8, y0 + 2, x0 + cell_w - 8, y0 + cell_h - 34], outline=color, width=3)
            true_zh = config.CLASS_NAMES_ZH[r.true]
            pred_zh = config.CLASS_NAMES_ZH[r.pred]
            label = f"真:{true_zh} 预:{pred_zh} {r.confidence:.0%}"
            tw = draw.textlength(label, font=f_text)
            draw.text((x0 + (cell_w - tw) / 2, y0 + cell_h - 28), label, font=f_text, fill=color)
        canvas.save(out_dir / "case_gallery.png")

    correct = df[df.correct].sort_values("confidence", ascending=False).head(8)
    wrong = df[~df.correct].sort_values("confidence", ascending=False).head(6)
    if len(correct):
        plot_cases([{"path": r.path, "true": config.CLASS_NAMES.index(r.true),
                     "pred": config.CLASS_NAMES.index(r.pred), "conf": r.confidence}
                    for r in correct.itertuples()], out_dir / "case_correct.png",
                   "正确预测案例（按置信度排序）")
    if len(wrong):
        plot_cases([{"path": r.path, "true": config.CLASS_NAMES.index(r.true),
                     "pred": config.CLASS_NAMES.index(r.pred), "conf": r.confidence}
                    for r in wrong.itertuples()], out_dir / "case_wrong.png",
                   "错误预测案例（按置信度排序）")
        wrong[["file_name", "true", "pred", "confidence", "top3"]].to_csv(
            out_dir / "error_cases.csv", index=False, encoding="utf-8-sig")

    print(f"[输出] 已写入 {out_dir}")


if __name__ == "__main__":
    main()
