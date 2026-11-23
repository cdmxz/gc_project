"""Flask 后端：加载 best_model.pth，提供真实推理接口，并托管前端页面。

启动：
    python src/app.py                 # 默认 http://127.0.0.1:5000
    python src/app.py --port 8000

接口：
    GET  /                    前端页面
    GET  /api/model-info      当前模型名称、参数量、类别、设备、状态
    POST /api/predict         上传图片（字段名 file），返回类别、置信度、Top-3、耗时
"""
import argparse
import io
import time
from pathlib import Path

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

import config
from data import build_eval_transform
from evaluate import load_checkpoint
from model import count_parameters

app = Flask(__name__, static_folder=str(config.FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

STATE = {"model": None, "model_name": None, "meta": {}, "device": None,
         "loaded_at": None, "version": "1.0.0", "requests": 0, "params": 0}


def init_model(checkpoint: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint.exists():
        raise FileNotFoundError(f"模型文件不存在: {checkpoint}")
    model, model_name, meta = load_checkpoint(checkpoint, device)
    STATE.update(model=model, model_name=model_name, meta=meta, device=device,
                 loaded_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                 params=count_parameters(model))
    print(f"[启动] 已加载 {checkpoint}（{model_name}, {STATE['params']:,} 参数, device={device}）")


@app.get("/")
def index():
    return send_from_directory(config.FRONTEND_DIR, "index.html")


@app.get("/samples/<path:filename>")
def samples(filename):
    return send_from_directory(config.DATASET_DIR / "samples", filename)


@app.get("/api/samples")
def api_samples():
    """返回数据集样例图列表，供前端展示与一键演示。"""
    items = []
    for f in sorted((config.DATASET_DIR / "samples").glob("*")):
        if f.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        cls = f.name.split("_")[0]
        items.append({"file": f.name, "class_name": cls,
                      "class_zh": config.CLASS_NAMES_ZH.get(cls, cls)})
    return jsonify({"ok": True, "samples": items})


@app.get("/api/model-info")
def model_info():
    return jsonify({
        "model_name": STATE["model_name"],
        "checkpoint": str(config.BEST_MODEL_PATH.relative_to(config.PROJECT_ROOT)),
        "num_params": STATE["params"],
        "num_classes": config.NUM_CLASSES,
        "class_names": config.CLASS_NAMES,
        "class_names_zh": config.CLASS_NAMES_ZH,
        "disposal_advice": config.DISPOSAL_ADVICE,
        "img_size": config.IMG_SIZE,
        "device": str(STATE["device"]),
        "loaded_at": STATE["loaded_at"],
        "torch_version": torch.__version__,
        "version": STATE["version"],
        "status": "ready" if STATE["model"] is not None else "not_loaded",
    })


@app.post("/api/predict")
def api_predict():
    if STATE["model"] is None:
        return jsonify({"ok": False, "error": "模型尚未加载"}), 503

    file = request.files.get("file")
    if file is None or file.filename == "":
        return jsonify({"ok": False, "error": "未接收到图片文件"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": f"不支持的文件类型 {ext}，请上传 JPG / PNG / BMP / WEBP"}), 400

    raw = file.read()
    if len(raw) == 0:
        return jsonify({"ok": False, "error": "文件为空"}), 400
    if len(raw) > config.MAX_UPLOAD_MB * 1024 * 1024:
        return jsonify({"ok": False, "error": f"图片超过 {config.MAX_UPLOAD_MB}MB 限制"}), 400

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        image.load()
    except Exception:
        return jsonify({"ok": False, "error": "图片无法解析，请确认文件未损坏"}), 400

    tensor = build_eval_transform()(image).unsqueeze(0).to(STATE["device"])
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = STATE["model"](tensor)
        if STATE["device"].type == "cuda":
            torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    probs = torch.softmax(logits, dim=1)[0].cpu()
    values, indices = probs.topk(min(3, config.NUM_CLASSES))
    topk = [{"class_name": config.CLASS_NAMES[i], "class_zh": config.CLASS_NAMES_ZH[config.CLASS_NAMES[i]],
             "probability": round(float(v), 4)} for v, i in zip(values, indices)]

    STATE["requests"] += 1
    return jsonify({
        "ok": True,
        "top1": topk[0],
        "topk": topk,
        "advice": config.DISPOSAL_ADVICE[topk[0]["class_name"]],
        "inference_ms": round(elapsed_ms, 2),
        "image_size": list(image.size),
        "model_name": STATE["model_name"],
        "device": str(STATE["device"]),
        "request_id": STATE["requests"],
    })


@app.errorhandler(413)
def too_large(_):
    return jsonify({"ok": False, "error": f"图片超过 {config.MAX_UPLOAD_MB}MB 限制"}), 413


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(config.BEST_MODEL_PATH))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    init_model(Path(args.checkpoint))
    print(f"[启动] 打开浏览器访问 http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
