"""数据清洗与数据集划分。

流程：
1. 扫描原始目录（Garbage classification/Garbage classification/<class>/*.jpg）
2. 用 PIL 校验图片是否可解码，记录宽高，剔除损坏文件
3. 用 MD5 去重，保证同一张图片不会同时出现在训练/验证/测试集（防数据泄漏）
4. 按类别分层抽样，7:1.5:1.5 划分 train/val/test，复制到 dataset/ 下的 ImageFolder 结构
5. 输出 dataset/dataset_manifest.csv 作为划分清单，并把样例图复制到 dataset/samples

用法：
    python src/split_dataset.py
    python src/split_dataset.py --raw_dir "D:/some/Garbage classification/Garbage classification"
"""
import argparse
import csv
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

import config

RAW_DEFAULT = config.DATASET_DIR / "_raw" / "Garbage classification" / "Garbage classification"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def md5_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def scan(raw_dir: Path):
    records, corrupt = [], []
    for class_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir()):
        cls = class_dir.name
        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in IMG_EXTS:
                continue
            try:
                with Image.open(img_path) as im:
                    im.verify()
                with Image.open(img_path) as im:
                    w, h = im.size
                    mode = im.mode
            except Exception as exc:                      # 损坏文件
                corrupt.append((str(img_path), repr(exc)))
                continue
            records.append({
                "class_name": cls,
                "file_name": img_path.name,
                "source_path": str(img_path),
                "width": w,
                "height": h,
                "mode": mode,
                "md5": md5_of(img_path),
            })
    return records, corrupt


def dedup(records):
    seen, kept, dropped = {}, [], []
    for r in records:
        if r["md5"] in seen:
            dropped.append((r["source_path"], seen[r["md5"]]))
        else:
            seen[r["md5"]] = r["source_path"]
            kept.append(r)
    return kept, dropped


def stratified_split(records, ratios=(0.7, 0.15, 0.15), seed=config.SEED):
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for r in records:
        by_class[r["class_name"]].append(r)

    for cls, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * ratios[2]))
        n_val = max(1, round(n * ratios[1]))
        n_train = n - n_val - n_test
        for i, r in enumerate(items):
            r["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return records


def copy_into_splits(records):
    for split in ("train", "val", "test"):
        for cls in config.CLASS_NAMES:
            (config.DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    for r in records:
        dst = config.DATASET_DIR / r["split"] / r["class_name"] / r["file_name"]
        shutil.copy2(r["source_path"], dst)


def export_samples(records, per_class=2):
    config.SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    picked = defaultdict(int)
    for r in sorted(records, key=lambda x: (x["split"], x["class_name"], x["file_name"])):
        if r["split"] == "test" and picked[r["class_name"]] < per_class:
            dst = config.SAMPLES_DIR / f"{r['class_name']}_{picked[r['class_name']] + 1}_{r['file_name']}"
            shutil.copy2(r["source_path"], dst)
            picked[r["class_name"]] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_dir", default=str(RAW_DEFAULT))
    ap.add_argument("--clean", action="store_true", help="重新划分前清空 dataset/train|val|test")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    assert raw_dir.exists(), f"原始数据目录不存在: {raw_dir}"

    if args.clean:
        for split in ("train", "val", "test"):
            shutil.rmtree(config.DATASET_DIR / split, ignore_errors=True)

    records, corrupt = scan(raw_dir)
    kept, duplicates = dedup(records)
    kept = stratified_split(kept)
    copy_into_splits(kept)
    export_samples(kept)

    fields = ["split", "class_name", "file_name", "width", "height", "mode", "md5", "source_path"]
    with open(config.MANIFEST_PATH, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in sorted(kept, key=lambda x: (x["split"], x["class_name"], x["file_name"])):
            writer.writerow({k: r[k] for k in fields})

    print(f"[扫描] 有效图片 {len(records)} 张，损坏 {len(corrupt)} 张")
    for p, e in corrupt:
        print("   损坏:", p, e)
    print(f"[去重] 重复图片 {len(duplicates)} 张（按 MD5 剔除，防止跨集合泄漏）")
    for dup, first in duplicates[:5]:
        print("   重复:", dup, "<->", first)

    stat = defaultdict(Counter)
    for r in kept:
        stat[r["split"]][r["class_name"]] += 1
    print(f"{'类别':<12}{'train':>8}{'val':>8}{'test':>8}{'合计':>8}")
    for cls in config.CLASS_NAMES:
        tr, va, te = stat['train'][cls], stat['val'][cls], stat['test'][cls]
        print(f"{cls:<12}{tr:>8}{va:>8}{te:>8}{tr + va + te:>8}")
    tot = {k: sum(stat[k].values()) for k in ("train", "val", "test")}
    print(f"{'合计':<12}{tot['train']:>8}{tot['val']:>8}{tot['test']:>8}{sum(tot.values()):>8}")
    print(f"[完成] 划分清单: {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
