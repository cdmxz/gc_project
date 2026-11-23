"""把 runs/*/history.csv 画成便于对比的 Loss / 准确率曲线（补充 TensorBoard 截图，颜色更易区分）。"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = {
    "E1_smallcnn_scratch_lr1e-3_noaug": ("#8a8f8c", "E1 SmallCNN 从零训练（无增强）"),
    "E2_resnet18_lr1e-3_aug": ("#c0392b", "E2 ResNet18 lr=1e-3"),
    "E3_resnet18_lr3e-4_aug": ("#1f7a4d", "E3 ResNet18 lr=3e-4 冻结3轮"),
    "E4_mobilenet_v3_lr5e-4_aug": ("#2a6fb0", "E4 MobileNetV3 lr=5e-4"),
}


def load(exp):
    p = ROOT / "runs" / exp / "history.csv"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {k: [float(r[k]) for r in rows] for k in rows[0]}


def main():
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.3), dpi=200)

    for exp, (color, label) in COLORS.items():
        h = load(exp)
        if not h:
            continue
        ax = axes[0]
        ax.plot(h["epoch"], h["train_loss"], color=color, lw=1.5, alpha=.55, ls="--")
        ax.plot(h["epoch"], h["val_loss"], color=color, lw=2.0, label=label)
        axes[1].plot(h["epoch"], h["val_acc"], color=color, lw=2.0, label=label)

    ax = axes[0]
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("交叉熵损失", fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_title("(a) 训练集（虚线）与验证集（实线）Loss", fontsize=11.5)
    ax.grid(linestyle=":", alpha=.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("验证集准确率", fontsize=10)
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.set_title("(b) 验证集准确率", fontsize=11.5)
    ax.set_ylim(0.3, 1.0)
    ax.grid(linestyle=":", alpha=.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # 标注最终模型的最佳轮次
    best = json.loads((ROOT / "runs" / "E4_mobilenet_v3_lr5e-4_aug" / "result.json").read_text(encoding="utf-8"))
    h4 = load("E4_mobilenet_v3_lr5e-4_aug")
    if h4:
        ep = best["best_epoch"]
        idx = int(ep) - 1
        axes[1].annotate(f"E4 最佳轮次 {ep}\nacc={h4['val_acc'][idx]:.3f}",
                         xy=(ep, h4["val_acc"][idx]), xytext=(ep + 1.5, h4["val_acc"][idx] - 0.16),
                         fontsize=8, color="#2a6fb0",
                         arrowprops=dict(arrowstyle="->", color="#2a6fb0", lw=1))

    fig.tight_layout()
    path = OUT / "fig_curves.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("[saved]", path)


if __name__ == "__main__":
    main()
