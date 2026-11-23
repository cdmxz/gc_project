import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NAME = "25大数据本_垃圾分类图像识别系统"
STAGE = ROOT / "_stage" / NAME
OUT_ZIP = ROOT.parent / "提交内容" / f"{NAME}.zip"

COPY_DIRS = ["src", "frontend", "runs", "tools", "outputs"]
COPY_FILES = ["README.md", "requirements.txt", "source_repository.txt", "一键运行.bat"]
SKIP = {"__pycache__", ".git", "experiments"}


def copy_tree(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name in SKIP or item.name.startswith("_"):
            continue
        if item.is_dir():
            copy_tree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)


def main():
    if (ROOT / "_stage").exists():
        shutil.rmtree(ROOT / "_stage")
    STAGE.mkdir(parents=True)

    for f in COPY_FILES:
        shutil.copy2(ROOT / f, STAGE / f)

    for d in COPY_DIRS:
        src = ROOT / d
        if not src.exists():
            print(f"[warn] 缺少 {d}")
            continue
        copy_tree(src, STAGE / d)

    dst_doc = STAGE / "report" / "项目设计报告.docx"
    dst_doc.parent.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "report" / "项目设计报告.docx", dst_doc)

    (STAGE / "models").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "models" / "best_model.pth", STAGE / "models" / "best_model.pth")
    shutil.copy2(ROOT / "models" / "model_meta.json", STAGE / "models" / "model_meta.json")

    ds = STAGE / "dataset"
    ds.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "dataset" / "dataset_manifest.csv", ds / "dataset_manifest.csv")
    shutil.copytree(ROOT / "dataset" / "samples", ds / "samples", dirs_exist_ok=True)
    (ds / "数据来源说明.txt").write_text(
        "数据来源：飞桨 AI Studio 公开数据集「垃圾分类」（数据集编号 128034）\n"
        "下载地址：https://aistudio.baidu.com/datasetdetail/128034\n"
        "类别：cardboard / glass / metal / paper / plastic / trash 共 6 类\n"
        "规模：原始 2527 张，PIL 校验全部可解码，MD5 去重剔除 3 张跨类别重复图，实得 2524 张\n"
        "划分：按类别分层抽样，训练 1768 / 验证 378 / 测试 378，逐张记录见 dataset_manifest.csv\n"
        "说明：完整原始图片体积较大，未随包提交；解压 垃圾分类.zip 到 dataset/_raw/ 后运行\n"
        "      python src/split_dataset.py 即可按清单重建全部集合，samples/ 为 12 张测试集样例。\n",
        encoding="utf-8")

    vid = STAGE / "video"
    vid.mkdir(exist_ok=True)
    shutil.copy2(ROOT.parent / "提交内容" / f"{NAME}_演示.mp4", vid / f"{NAME}_演示.mp4")
    shutil.copy2(ROOT.parent / "提交内容" / "演示字幕.srt", vid / "演示字幕.srt") if (
        ROOT.parent / "提交内容" / "演示字幕.srt").exists() else None
    (vid / "演示视频链接.txt").write_text(
        f"演示视频：{NAME}_演示.mp4（与压缩包同目录，7 分 45 秒，含中文配音，1920×1080）\n"
        "章节：0:00 项目与场景 / 0:40 数据集 / 1:40 模型与参数 / 2:40 TensorBoard Loss 曲线 /\n"
        "      4:00 前端上传与实时预测 / 6:00 测试结果与错误案例 / 7:00 总结\n"
        "说明：2:40 起为真实屏幕录制（TensorBoard 现场操作、前端现场上传并推理），非静态截图。\n",
        encoding="utf-8")

    tmp_zip = ROOT / "_stage" / f"{NAME}.zip"
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(STAGE.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(ROOT / "_stage"))

    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
    shutil.move(str(tmp_zip), OUT_ZIP)

    size = OUT_ZIP.stat().st_size / 1024 / 1024
    n = sum(1 for _ in STAGE.rglob("*") if _.is_file())
    print(f"[完成] {OUT_ZIP}  {size:.1f} MB  {n} 个文件")


if __name__ == "__main__":
    main()
