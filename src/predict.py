"""单张图片推理脚本。

用法：
    python src/predict.py --image dataset/test/plastic/plastic1.jpg
    python src/predict.py --image demo.jpg --checkpoint models/best_model.pth --topk 3 --save outputs/pred.jpg
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

import config
from data import build_eval_transform
from evaluate import load_checkpoint


@torch.no_grad()
def predict_image(model, image_path, device, topk=3):
    image = Image.open(image_path).convert("RGB")
    tensor = build_eval_transform()(image).unsqueeze(0).to(device)
    t0 = time.perf_counter()
    logits = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    probs = torch.softmax(logits, dim=1)[0].cpu()
    values, indices = probs.topk(topk)
    results = [{
        "class_name": config.CLASS_NAMES[i],
        "class_zh": config.CLASS_NAMES_ZH[config.CLASS_NAMES[i]],
        "probability": round(float(v), 4),
    } for v, i in zip(values, indices)]
    return {
        "image": str(image_path),
        "image_size": list(image.size),
        "top1": results[0],
        "topk": results,
        "advice": config.DISPOSAL_ADVICE[results[0]["class_name"]],
        "inference_ms": round(elapsed_ms, 2),
        "device": str(device),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default=str(config.BEST_MODEL_PATH))
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--save", default=None, help="把带预测标注的图片另存为指定路径")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_name, _ = load_checkpoint(args.checkpoint, device)
    print(f"[模型] {model_name} ← {args.checkpoint}")

    result = predict_image(model, args.image, device, args.topk)
    result["model"] = model_name
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.save:
        from PIL import ImageDraw, ImageFont
        img = Image.open(args.image).convert("RGB").resize((480, int(480 * Image.open(args.image).height /
                                                                     Image.open(args.image).width)))
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 26)
        except Exception:
            font = ImageFont.load_default()
        text = f"{result['top1']['class_zh']} {result['top1']['probability']:.1%}"
        draw.rectangle([0, 0, 480, 44], fill=(20, 20, 20))
        draw.text((12, 6), text, fill=(255, 255, 255), font=font)
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        img.save(args.save)
        print(f"[保存] {args.save}")


if __name__ == "__main__":
    main()
