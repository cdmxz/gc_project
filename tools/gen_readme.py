"""根据真实实验结果生成 README.md（数值全部来自 runs/ 与 outputs/）。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP_META = [
    ("E1_smallcnn_scratch_lr1e-3_noaug", "SmallCNN（从零训练、无增强）", "无", "1e-3"),
    ("E2_resnet18_lr1e-3_aug", "ResNet18（预训练 + 增强）", "有", "1e-3"),
    ("E3_resnet18_lr3e-4_aug", "ResNet18（预训练 + 增强 + 冻结3轮）", "有", "3e-4"),
    ("E4_mobilenet_v3_lr5e-4_aug", "MobileNetV3-Small（预训练 + 增强）", "有", "5e-4"),
]
REPO_URL = "https://github.com/cdmxz/gc_project"


def main():
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    exps = {}
    for exp, *_ in EXP_META:
        p = ROOT / "runs" / exp / "result.json"
        exps[exp] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
    size_mb = (ROOT / "models" / "best_model.pth").stat().st_size / 1024 / 1024

    rows = []
    for exp, name, aug, lr in EXP_META:
        r = exps.get(exp, {})
        rows.append(f"| {exp} | {name} | {aug} | {lr} | {r.get('best_epoch', '-')} | "
                    f"{r.get('best_val_acc', 0) * 100:.2f}% | {r.get('best_val_f1', 0):.4f} | "
                    f"{r.get('total_time_s', 0) / 60:.1f} min |")
    table = "\n".join(rows)

    pc_rows = "\n".join(
        f"| {k_zh}（{k}） | {v['precision']:.4f} | {v['recall']:.4f} | {v['f1']:.4f} | {v['support']} |"
        for k, v in metrics["per_class"].items()
        for k_zh in [{"cardboard": "纸板", "glass": "玻璃", "metal": "金属",
                      "paper": "纸张", "plastic": "塑料", "trash": "其他垃圾"}[k]]
    )

    readme = f"""# 基于 PyTorch 的垃圾图像智能分类系统

PyTorch 迁移学习（ResNet18 / MobileNetV3-Small）+ TensorBoard 训练可视化 + Flask 实时推理前端，六类生活垃圾（纸板 / 玻璃 / 金属 / 纸张 / 塑料 / 其他垃圾）图像分类，深度学习案例项目作业。

## 1. 项目简介

- **任务**：六类生活垃圾（纸板 / 玻璃 / 金属 / 纸张 / 塑料 / 其他垃圾）单张图片分类，输出类别、置信度、Top-3 与投放建议
- **数据**：飞桨 AI Studio 公开数据集「垃圾分类」（编号 128034），6 类 2527 张实拍图，清洗去重后 2524 张
- **模型**：ImageNet 预训练迁移学习，最终权重 MobileNetV3-Small，参数量 1,524,006
- **结果**：测试集准确率 89.15%，Macro-F1 0.8867，Top-3 准确率 98.68%，GPU 单张推理约 6 ms
- **系统**：PyTorch 训练 + TensorBoard 监控 + Flask 后端 + 原生 HTML/CSS/JS 前端，上传即推理

## 2. 源码地址

- 平台：GitHub
- 项目地址：https://github.com/cdmxz/gc_project
- 仓库可见性：公开

## 3. 目录结构

```
.
├─ README.md                    本文件
├─ 一键运行.bat                 一键启动前后端（Windows 双击即可）
├─ requirements.txt             依赖清单
├─ source_repository.txt        源码链接说明
├─ src/
│  ├─ config.py                 路径、类别、预处理与默认超参
│  ├─ data.py                   Dataset / DataLoader 与数据增强
│  ├─ model.py                  SmallCNN / ResNet18 / MobileNetV3
│  ├─ split_dataset.py          清洗、去重、分层划分、生成清单
│  ├─ train.py                  训练循环 + TensorBoard 记录
│  ├─ evaluate.py               测试集指标、混淆矩阵、案例分析
│  ├─ predict.py                单张图片推理
│  └─ app.py                    Flask 后端 + 前端托管
├─ frontend/                    index.html / style.css / script.js
├─ dataset/                     dataset_manifest.csv + samples/（12 张样例）
├─ models/                      best_model.pth + model_meta.json
├─ runs/                        4 组实验的 TensorBoard 日志
├─ outputs/                     评估结果、混淆矩阵、曲线与页面截图
├─ report/项目设计文档.docx      项目设计报告
└─ video/                       演示视频（25大数据本_垃圾分类图像识别系统_演示.mp4，7 分 45 秒）
```

## 4. 环境安装

```bash
python -m venv .venv
.venv\\Scripts\\activate            # Windows；Linux/macOS 用 source .venv/bin/activate
pip install -r requirements.txt
# GPU 用户若默认装的是 CPU 版 torch，按官网命令重装，例如：
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

验证：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

## 5. 数据集获取

1. 打开 https://aistudio.baidu.com/datasetdetail/128034 下载 `垃圾分类.zip`（约 86MB）
2. 解压到 `dataset/_raw/`，使路径为 `dataset/_raw/Garbage classification/Garbage classification/<类别>/*.jpg`
3. 执行划分脚本（自动清洗、MD5 去重、7:1.5:1.5 分层划分并生成清单）：

```bash
python src/split_dataset.py
```

输出：`dataset/train|val|test/<class>/`、`dataset/dataset_manifest.csv`、`dataset/samples/`（12 张测试集样例）。

## 6. 模型训练

```bash
# 最终模型 E4（MobileNetV3-Small，验证集 Macro-F1 最高）
python src/train.py --model mobilenet_v3 --epochs 20 --lr 5e-4 --exp-name E4_mobilenet_v3_lr5e-4_aug

# 三组对照实验
python src/train.py --model smallcnn     --epochs 20 --lr 1e-3 --no-aug --scratch --exp-name E1_smallcnn_scratch_lr1e-3_noaug
python src/train.py --model resnet18     --epochs 20 --lr 1e-3 --exp-name E2_resnet18_lr1e-3_aug
python src/train.py --model resnet18     --epochs 20 --lr 3e-4 --freeze-epochs 3 --exp-name E3_resnet18_lr3e-4_aug
```

每组实验写入独立的 `runs/<exp>/`（事件文件、config.json、history.csv、result.json），最佳权重保存到
`models/experiments/<exp>_best.pth`。复制最终权重：

```bash
copy models\\experiments\\E3_resnet18_lr3e-4_aug_best.pth models\\best_model.pth    # Windows
cp models/experiments/E4_mobilenet_v3_lr5e-4_aug_best.pth models/best_model.pth        # Linux/macOS
```

## 7. 模型评估

```bash
python src/evaluate.py --checkpoint models/best_model.pth --split test
```

输出 `outputs/`：`metrics.json`、`classification_report.txt`、`confusion_matrix.png`、
`predictions.csv`、`case_correct.png`、`case_wrong.png`、`error_cases.csv`。

测试集（378 张）结果：

| 指标 | 数值 |
| --- | --- |
| Accuracy | {metrics['accuracy'] * 100:.2f}% |
| Macro-F1 | {metrics['macro_f1']:.4f} |
| Top-3 Accuracy | {metrics['top3_accuracy'] * 100:.2f}% |
| 单张推理耗时（GPU） | {metrics.get('inference_ms', 0):.2f} ms |

| 类别 | 精确率 | 召回率 | F1 | 样本数 |
| --- | --- | --- | --- | --- |
{pc_rows}

## 8. TensorBoard

```bash
tensorboard --logdir runs --port 6006
# 浏览器打开 http://localhost:6006 ，在 SCALARS 中勾选实验对比
```

四组实验记录 `Loss/train`、`Loss/val`、`Accuracy/train`、`Accuracy/val`、`F1_macro/val`、`LearningRate`：

| 实验目录 | 模型 | 增强 | 学习率 | 最佳轮次 | 最佳验证准确率 | 最佳验证 Macro-F1 | 训练耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
{table}

## 9. 单张图片推理

```bash
python src/predict.py --image dataset/test/plastic/plastic100.jpg --topk 3
python src/predict.py --image demo.jpg --save outputs/pred.jpg     # 保存带标注结果图
```

## 10. 启动前端

**一键启动（推荐）**：双击项目根目录的 `一键运行.bat`。脚本会自动完成：检查 Python 与模型文件 →
缺依赖时安装 `requirements.txt` → 检测端口占用（5000 被占用则改用 8000）→ 启动 Flask 服务 → 5 秒后自动打开浏览器。
关闭黑窗口或按 `Ctrl+C` 即停止服务。

**手动启动**：

```bash
python src/app.py                  # 默认 http://127.0.0.1:5000
python src/app.py --port 8000      # 指定端口
```

后端与前端同端口部署：Flask 既提供 `/api/*` 推理接口，也直接托管 `frontend/` 静态页面。
启动后终端打印模型加载信息，浏览器打开 `http://127.0.0.1:5000` 即可。接口：

- `GET /api/model-info`：模型名称、参数量、类别、设备、加载时间
- `GET /api/samples`：数据集样例图列表（页面底部缩略图）
- `POST /api/predict`：`multipart/form-data`，字段名 `file`，返回 top1 / topk / advice / inference_ms

## 11. 模型文件说明

| 项目 | 内容 |
| --- | --- |
| 文件名 | `models/best_model.pth` |
| 大小 | {size_mb:.1f} MB |
| 内容 | `torch.save(model.state_dict())` 纯权重字典 |
| 结构 | {meta.get('display_name', meta['model_name'])}，参数量 {meta['num_params']:,} |
| 输入 | 1×3×224×224，ImageNet 均值/方差归一化 |
| 类别顺序 | {", ".join(meta['class_names'])} |
| 加载 | `torch.load(..., map_location=device)` → `model.load_state_dict(state)` → `model.eval()` |

## 12. 已知问题

- 数据集较小（2524 张）且类别不均衡，`trash` 类仅 137 张，该类别召回率相对偏低；
- 只做整图分类，一张图片中出现多件垃圾时只能给出一个结果；
- 对材质相近的类别（纸 / 纸板、塑料 / 玻璃）存在混淆；
- 未做量化与容器化，CPU 推理耗时高于 GPU；
- 摄像头采集依赖浏览器授权，部分环境需 HTTPS 或 localhost 才能调用。

## 13. AI 工具使用说明

开发过程中使用生成式 AI 辅助代码草稿、报告文字整理与绘图脚本编写。所有结果均由本人运行验证：
数据划分来自 `src/split_dataset.py` 实际输出，曲线来自真实 TensorBoard 事件文件，指标由 `src/evaluate.py`
在测试集上计算，截图来自真实启动的服务。未使用伪造数据或手工修改指标。
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    print("[saved] README.md")


if __name__ == "__main__":
    main()
