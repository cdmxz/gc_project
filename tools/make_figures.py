"""生成报告所需静态图表：类别分布、数据样例图版、系统流程图、网络结构示意图。

用法：python tools/make_figures.py
输出：outputs/fig_class_dist.png、fig_samples.png、fig_system_flow.png、fig_model_arch.png
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

GREEN = "#1f7a4d"
GREEN_L = "#7ab596"
GREEN_LL = "#cfe3d7"
INK = "#25322c"


def read_manifest():
    rows = []
    with open(config.MANIFEST_PATH, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fig_class_dist(rows):
    stat = defaultdict(Counter)
    for r in rows:
        stat[r["class_name"]][r["split"]] += 1
    classes = config.CLASS_NAMES
    labels = [f"{config.CLASS_NAMES_ZH[c]}\n{c}" for c in classes]
    tr = [stat[c]["train"] for c in classes]
    va = [stat[c]["val"] for c in classes]
    te = [stat[c]["test"] for c in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), dpi=200,
                             gridspec_kw={"width_ratios": [1.55, 1]})
    ax = axes[0]
    x = range(len(classes))
    b1 = ax.bar(x, tr, 0.62, label="训练集", color=GREEN)
    b2 = ax.bar(x, va, 0.62, bottom=tr, label="验证集", color=GREEN_L)
    b3 = ax.bar(x, te, 0.62, bottom=[a + b for a, b in zip(tr, va)], label="测试集", color=GREEN_LL)
    for i, c in enumerate(classes):
        total = tr[i] + va[i] + te[i]
        ax.text(i, total + 12, str(total), ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("图片数量（张）", fontsize=10)
    ax.set_title("(a) 各类别样本数量与集合划分（总计 2524 张）", fontsize=11.5)
    ax.set_ylim(0, max(tr[i] + va[i] + te[i] for i in range(len(classes))) * 1.16)
    ax.legend(frameon=False, fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.16))
    ax.grid(axis="y", linestyle=":", alpha=.45)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax2 = axes[1]
    totals = [tr[i] + va[i] + te[i] for i in range(len(classes))]
    order = sorted(range(len(classes)), key=lambda i: totals[i])
    ax2.barh([labels[i].replace("\n", " ") for i in order], [totals[i] for i in order],
             0.62, color=[GREEN if totals[i] > 300 else "#d08a3a" for i in order])
    for i, idx in enumerate(order):
        ax2.text(totals[idx] + 8, i, str(totals[idx]), va="center", fontsize=9, color=INK)
    ax2.set_xlabel("图片数量（张）", fontsize=10)
    ax2.set_title("(b) 类别不均衡情况（trash 仅 137 张）", fontsize=11.5)
    ax2.set_xlim(0, max(totals) * 1.15)
    ax2.grid(axis="x", linestyle=":", alpha=.45)
    ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig_class_dist.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig_class_dist.png")


def fig_samples(rows, per_class=2):
    """输出 12 张（6 类 × 2）真实样例图版，每张标注类别与文件名。"""
    from PIL import Image
    picked = defaultdict(list)
    for r in rows:
        if r["split"] == "test" and len(picked[r["class_name"]]) < per_class:
            picked[r["class_name"]].append(r)
    fig, axes = plt.subplots(2, 6, figsize=(15.6, 6.6), dpi=180)
    axes = axes.reshape(-1)
    items = []
    for c in config.CLASS_NAMES:
        for r in picked[c]:
            items.append((c, r))
    for ax in axes:
        ax.axis("off")
    for ax, (cls, r) in zip(axes, items):
        path = ROOT / "dataset" / r["split"] / cls / r["file_name"]
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(f"{config.CLASS_NAMES_ZH[cls]} / {cls}\n{r['file_name']}",
                     fontsize=9, color=INK)
        ax.axis("on")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("数据集真实样例（均为测试集图片，未参与训练）", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / "fig_samples.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig_samples.png")


def _box(ax, x, y, w, h, text, fc="#ffffff", ec=GREEN, fs=9.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.003,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.5)


def _arrow(ax, p1, p2, color="#6b8f7d"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                                 lw=1.2, color=color, shrinkA=2, shrinkB=2))


def fig_system_flow():
    fig, ax = plt.subplots(figsize=(11.6, 6.6), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, "系统总体流程图：离线训练 + 在线推理", ha="center", fontsize=13, fontweight="bold", color=INK)

    # 离线
    ax.text(0.03, 0.90, "离线训练链路", fontsize=11, color=GREEN, fontweight="bold")
    ax.add_patch(FancyBboxPatch((0.018, 0.695), 0.305, 0.215, boxstyle="round,pad=0.004",
                                fc="#f7fbf8", ec="#cfe3d7", lw=1))
    _box(ax, 0.035, 0.83, 0.27, 0.055, "AI Studio 公开数据集（6 类 2527 张）", fc="#eef7f1")
    _box(ax, 0.035, 0.755, 0.27, 0.06, "清洗 + MD5 去重 + 分层划分 7:1.5:1.5", fc="#eef7f1")
    _box(ax, 0.035, 0.705, 0.27, 0.035, "dataset_manifest.csv（2524 张）", fc="#eef7f1", fs=8.8)

    _box(ax, 0.375, 0.80, 0.24, 0.10, "torchvision transforms\n随机裁剪/翻转/旋转/颜色抖动\n+ ImageNet 归一化")
    _box(ax, 0.375, 0.64, 0.24, 0.10, "ResNet18 迁移学习\n冻结主干 → 全参微调\nAdamW + 余弦退火")
    _box(ax, 0.375, 0.48, 0.24, 0.10, "训练循环\n前向 → 交叉熵 → 反向传播\n→ 参数更新（AMP）")

    _box(ax, 0.68, 0.80, 0.29, 0.10, "TensorBoard 日志\nruns/<exp>/events.out.tfevents\nLoss / Accuracy / LR", fc="#fffaf0", ec="#e3c88f")
    _box(ax, 0.68, 0.64, 0.29, 0.10, "验证集评估\n按 Macro-F1 选择最佳 Epoch\n保存权重", fc="#fffaf0", ec="#e3c88f")
    _box(ax, 0.68, 0.48, 0.29, 0.10, "models/best_model.pth\n+ config.json / metrics.json", fc="#fdeeec", ec="#e8b4ac", bold=True)

    _arrow(ax, (0.305, 0.845), (0.375, 0.845))
    _arrow(ax, (0.495, 0.80), (0.495, 0.74))
    _arrow(ax, (0.495, 0.64), (0.495, 0.58))
    _arrow(ax, (0.615, 0.845), (0.68, 0.845))
    _arrow(ax, (0.825, 0.80), (0.825, 0.74))
    _arrow(ax, (0.825, 0.64), (0.825, 0.58))

    # 在线
    ax.text(0.03, 0.40, "在线推理链路", fontsize=11, color=GREEN, fontweight="bold")
    _box(ax, 0.03, 0.26, 0.17, 0.09, "浏览器前端\n上传 / 摄像头\n类型与大小校验")
    _box(ax, 0.245, 0.26, 0.17, 0.09, "POST /api/predict\nmultipart 图片")
    _box(ax, 0.46, 0.26, 0.19, 0.09, "Flask 服务\nPIL 解码 → RGB\n统一预处理 224×224")
    _box(ax, 0.695, 0.26, 0.19, 0.09, "best_model.pth\nsoftmax 推理\nTop-3 + 耗时")
    _box(ax, 0.03, 0.09, 0.855, 0.09, "前端结果区：类别（中/英）、置信度、Top-3 概率条、推理耗时、投放建议；低置信度提示人工复核",
         fc="#eef7f1", fs=9)

    _arrow(ax, (0.20, 0.305), (0.245, 0.305))
    _arrow(ax, (0.415, 0.305), (0.46, 0.305))
    _arrow(ax, (0.65, 0.305), (0.695, 0.305))
    _arrow(ax, (0.79, 0.26), (0.79, 0.185))
    _arrow(ax, (0.695, 0.545), (0.695, 0.355))

    fig.tight_layout()
    fig.savefig(OUT / "fig_system_flow.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig_system_flow.png")


def fig_model_arch():
    fig, ax = plt.subplots(figsize=(11.6, 4.4), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.95, "ResNet18 迁移学习网络结构（输入 224×224×3，输出 6 类）",
            ha="center", fontsize=12.5, fontweight="bold", color=INK)

    blocks = [
        ("Input\n224×224×3", "#eef2f0"),
        ("conv1 7×7/2\n64×112×112", "#eef7f1"),
        ("maxpool\n64×56×56", "#eef7f1"),
        ("layer1\n2×BasicBlock\n64×56×56", "#dcefe4"),
        ("layer2\n2×BasicBlock\n128×28×28", "#dcefe4"),
        ("layer3\n2×BasicBlock\n256×14×14", "#dcefe4"),
        ("layer4\n2×BasicBlock\n512×7×7", "#dcefe4"),
        ("AvgPool\n512×1×1", "#eef7f1"),
        ("Dropout 0.4\nFC 512→6\nSoftmax", "#fdeeec"),
    ]

    w, gap, x0 = 0.094, 0.014, 0.015
    xs = [x0 + i * (w + gap) for i in range(len(blocks))]
    for (text, color), x in zip(blocks, xs):
        _box(ax, x, 0.44, w, 0.28, text, fc=color, fs=8.4)
    for i in range(len(blocks) - 1):
        _arrow(ax, (xs[i] + w + 0.002, 0.58), (xs[i + 1] - 0.002, 0.58))

    ax.add_patch(FancyBboxPatch((0.10, 0.14), 0.72, 0.16, boxstyle="round,pad=0.01",
                                fc="#fffaf0", ec="#e3c88f"))
    ax.text(0.46, 0.22, "迁移学习：主干加载 ImageNet 预训练权重，替换最后全连接层为 6 类（Dropout + FC 512→6）\n"
                        "最终模型 E2 直接全参微调（lr=1e-3）；对照实验 E3 额外采用“前 3 轮冻结主干、再解冻微调”的两阶段策略\n"
                        "总参数量 11,179,590（约 11.18M），输入 224×224，输出 6 类",
            ha="center", va="center", fontsize=8.6, color="#6d5522", linespacing=1.7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_model_arch.png", bbox_inches="tight")
    plt.close(fig)
    print("saved", OUT / "fig_model_arch.png")


if __name__ == "__main__":
    rows = read_manifest()
    fig_class_dist(rows)
    fig_samples(rows)
    fig_system_flow()
    fig_model_arch()
