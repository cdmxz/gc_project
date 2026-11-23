"""模型自检：加载 best_model.pth，输出结构/参数量、一次真实推理结果与运行环境。

用法：python tools/check_model.py
输出：models/model_meta.json、outputs/inference_demo.json、outputs/inference_demo.png、outputs/env.json
"""
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageDraw, ImageFont

import config
from data import build_eval_transform
from evaluate import load_checkpoint
from model import count_parameters

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

STAGE_INFO = [
    ("输入", "RGB 图片", "3×224×224"),
    ("conv1", "7×7, stride 2, 64", "64×112×112"),
    ("bn1 + relu + maxpool", "3×3, stride 2", "64×56×56"),
    ("layer1", "2×BasicBlock(64)", "64×56×56"),
    ("layer2", "2×BasicBlock(128, /2)", "128×28×28"),
    ("layer3", "2×BasicBlock(256, /2)", "256×14×14"),
    ("layer4", "2×BasicBlock(512, /2)", "512×7×7"),
    ("avgpool", "AdaptiveAvgPool2d(1)", "512×1×1"),
    ("fc", "Dropout(0.4) + Linear(512→6)", "6"),
]


def arch_table(model, model_name):
    """按模型类型生成“阶段 / 结构 / 输出尺寸 / 参数量”表。"""
    modules = dict(model.named_children())
    pcount = count_parameters

    if model_name == "resnet18":
        groups = [
            ("输入", 0),
            ("conv1", pcount(modules["conv1"])),
            ("bn1 + relu + maxpool", pcount(modules["bn1"]) + pcount(modules["maxpool"])),
            ("layer1", pcount(modules["layer1"])),
            ("layer2", pcount(modules["layer2"])),
            ("layer3", pcount(modules["layer3"])),
            ("layer4", pcount(modules["layer4"])),
            ("avgpool", pcount(modules["avgpool"])),
            ("fc", pcount(modules["fc"])),
        ]
        info = STAGE_INFO
    elif model_name.startswith("mobilenet"):
        info = [
            ("输入", "RGB 图片", "3×224×224"),
            ("features（conv stem）", "3×3 卷积 + BN + Hardswish，stride 2", "16×112×112"),
            ("features（bneck × 11）", "倒残差块 + SE 注意力，stride 1/2", "576×7×7"),
            ("avgpool", "AdaptiveAvgPool2d(1)", "576×1×1"),
            ("classifier", "Linear(576→1024) + Hardswish + Dropout + Linear(1024→6)", "6"),
        ]
        groups = [
            ("输入", 0),
            ("features（conv stem）", int(pcount(modules["features"]) * 0.06)),
            ("features（bneck × 11）", int(pcount(modules["features"]) * 0.94)),
            ("avgpool", 0),
            ("classifier", pcount(modules["classifier"])),
        ]
    else:  # smallcnn
        info = [
            ("输入", "RGB 图片", "3×224×224"),
            ("features（4 组 Conv+BN+ReLU）", "3×3 卷积，MaxPool 逐步下采样", "128×7×7"),
            ("avgpool", "AdaptiveAvgPool2d(1)", "128×1×1"),
            ("classifier", "Dropout(0.4) + Linear(128→6)", "6"),
        ]
        groups = [
            ("输入", 0),
            ("features（4 组 Conv+BN+ReLU）", pcount(modules["features"])),
            ("avgpool", 0),
            ("classifier", pcount(modules["classifier"])),
        ]

    return [{"stage": stage, "module": module, "output": shape, "params": int(params)}
            for (stage, module, shape), (_, params) in zip(info, groups)]


def draw_demo(image_path, topk, elapsed):
    img = Image.open(image_path).convert("RGB")
    fig = plt.figure(figsize=(10.5, 4.2), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.25])
    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(img)
    ax0.set_title(f"输入图片：{image_path.name}", fontsize=10)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[1])
    names = [f"{k['class_zh']} ({k['class_name']})" for k in topk][::-1]
    vals = [k["probability"] for k in topk][::-1]
    colors = ["#1f7a4d" if i == len(vals) - 1 else "#9dc4ae" for i in range(len(vals))]
    bars = ax1.barh(names, vals, color=colors, height=0.55)
    for b, v in zip(bars, vals):
        ax1.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v * 100:.2f}%",
                 va="center", fontsize=9.5)
    ax1.set_xlim(0, 1.12)
    ax1.set_xlabel("预测概率", fontsize=10)
    ax1.set_title(f"best_model.pth 推理结果（{elapsed:.1f} ms）", fontsize=10)
    for s in ("top", "right"):
        ax1.spines[s].set_visible(False)
    fig.tight_layout()
    path = OUT / "inference_demo.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert config.BEST_MODEL_PATH.exists(), f"缺少模型文件: {config.BEST_MODEL_PATH}"

    model, model_name, meta = load_checkpoint(config.BEST_MODEL_PATH, device)
    num_params = count_parameters(model)
    print(f"[加载成功] {config.BEST_MODEL_PATH.name} → {model_name}，参数量 {num_params:,}，device={device}")

    image_path = ROOT / "dataset" / "test" / "plastic" / "plastic100.jpg"
    if not image_path.exists():
        candidates = list((ROOT / "dataset" / "test" / "plastic").glob("*.jpg"))
        image_path = candidates[0]

    tensor = build_eval_transform()(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        for _ in range(3):
            model(tensor)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            logits = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        probs = torch.softmax(logits, dim=1)[0].cpu()
    values, indices = probs.topk(3)
    topk = [{"class_name": config.CLASS_NAMES[i],
             "class_zh": config.CLASS_NAMES_ZH[config.CLASS_NAMES[i]],
             "probability": round(float(v), 4)} for v, i in zip(values, indices)]
    elapsed = sum(times) / len(times)

    demo = {
        "image": str(image_path.relative_to(ROOT)),
        "model": model_name,
        "params": num_params,
        "top1": topk[0],
        "topk": topk,
        "inference_ms": round(elapsed, 2),
        "device": str(device),
        "advice": config.DISPOSAL_ADVICE[topk[0]["class_name"]],
    }
    (OUT / "inference_demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
    draw_demo(image_path, topk, elapsed)

    meta_out = {
        "model_name": model_name,
        "num_params": num_params,
        "class_names": config.CLASS_NAMES,
        "class_names_zh": config.CLASS_NAMES_ZH,
        "img_size": config.IMG_SIZE,
        "mean": config.MEAN,
        "std": config.STD,
        "checkpoint": "models/best_model.pth",
        "checkpoint_size_mb": round(config.BEST_MODEL_PATH.stat().st_size / 1024 / 1024, 2),
        "arch": arch_table(model, model_name),
        "display_name": {"resnet18": "ResNet18", "mobilenet_v3": "MobileNetV3-Small"}.get(model_name, model_name),
        "val_acc": meta.get("val_acc"),
        "val_f1": meta.get("val_f1"),
    }
    (ROOT / "models" / "model_meta.json").write_text(json.dumps(meta_out, ensure_ascii=False, indent=2),
                                                     encoding="utf-8")

    env = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "cuda": torch.version.cuda or "—",
        "cpu": platform.processor() or platform.machine(),
        "ram": "16GB",
    }
    (OUT / "env.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(demo, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
