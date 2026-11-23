"""合成 5-8 分钟演示视频：真实录屏 + 中文配音 + 章节结构。

分段（与作业要求的时间轴一致）：
  0:00-0:40 项目名称与应用场景        0:40-1:40 数据集来源与样例
  1:40-2:40 模型结构与关键参数        2:40-4:00 TensorBoard Loss 曲线对比（真实录屏）
  4:00-6:00 前端上传与实时预测（真实录屏）  6:00-7:00 测试结果与错误案例
  7:00-7:45 总结与改进方向

用法：python tools/make_demo.py           # 需要先用 record_clips.py 录好两段
输出：video/25大数据本_垃圾分类图像识别系统_演示.mp4、video/演示字幕.srt
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import config  # noqa: E402

OUT = ROOT / "outputs"
VID = ROOT / "video"
BUILD = VID / "_build"
BUILD.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
INK = (20, 32, 27)
INK2 = (74, 90, 83)
GREEN = (31, 122, 77)
GREEN_D = (16, 53, 42)
BG = (246, 248, 247)

VOICE = "zh-CN-YunxiNeural"
RATE = "+8%"
FFMPEG = None


def ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def font(size, bold=False):
    return ImageFont.truetype(FONT_B if bold and Path(FONT_B).exists() else FONT, size)


def wrap(text, fnt, max_width, draw):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=fnt) > max_width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def probe(path):
    out = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


# ------------------------------------------------------------------ 静态画面
def frame(chapter, title, bullets, image=None, caption=None, note=None, big=None):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 8], fill=GREEN)
    d.text((70, 34), "基于 PyTorch 的垃圾图像智能分类系统", font=font(26, True), fill=GREEN_D)
    d.text((W - 380, 38), chapter, font=font(24), fill=(140, 158, 150))

    d.text((70, 100), title, font=font(54, True), fill=INK)
    y = 182
    for b in bullets:
        d.ellipse([74, y + 13, 86, y + 25], fill=GREEN)
        d.text((104, y), b, font=font(25), fill=INK2)
        y += 44

    if big:
        box_top = y + 26
        d.rounded_rectangle([70, box_top, W - 70, box_top + 150], radius=14, fill=(255, 255, 255),
                            outline=(219, 231, 224), width=2)
        d.text((100, box_top + 28), big, font=font(30, True), fill=GREEN_D)
        if note:
            d.text((100, box_top + 86), note, font=font(22), fill=(120, 136, 128))
        y = box_top + 176

    if image and Path(image).exists():
        shot = Image.open(image).convert("RGB")
        avail_h = H - y - 90
        scale = min((W - 140) / shot.width, avail_h / shot.height)
        shot = shot.resize((int(shot.width * scale), int(shot.height * scale)), Image.LANCZOS)
        x = (W - shot.width) // 2
        img.paste(shot, (x, y))
        if caption:
            d.text((x, y + shot.height + 6), caption, font=font(21), fill=(132, 150, 142))

    # 底部章节条
    d.rectangle([0, H - 56, W, H], fill=(255, 255, 255))
    d.line([(0, H - 56), (W, H - 56)], fill=(226, 232, 228), width=2)
    d.text((70, H - 40), chapter, font=font(20), fill=(150, 165, 158))
    d.text((W - 470, H - 40), "演示视频 · 垃圾分类图像识别系统", font=font(20), fill=(150, 165, 158))
    return img


def overlay_band(chapter):
    """录屏片段底部加的章节条（半透明）。"""
    band = Image.new("RGBA", (W, 86), (0, 0, 0, 0))
    d = ImageDraw.Draw(band)
    d.rectangle([0, 0, W, 86], fill=(18, 30, 24, 205))
    d.text((64, 22), chapter, font=font(30, True), fill=(255, 255, 255))
    d.text((W - 620, 28), "真实屏幕录制 · 实际操作", font=font(24), fill=(190, 214, 199))
    return band


# ------------------------------------------------------------------ 配音
def tts(text, path):
    subprocess.run([sys.executable, "-m", "edge_tts", "--text", text, "--voice", VOICE,
                    "--rate", RATE, "--write-media", str(path)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def fit_audio(src, dst, target):
    """把配音时长压/拉到 target 秒（±20% 以内用 atempo，避免变声）。"""
    dur = probe(src)
    if dur <= 0 or not target:
        Path(dst).write_bytes(Path(src).read_bytes())
        return
    tempo = max(0.85, min(1.25, dur / (target - 0.35)))
    af = f"atempo={tempo:.4f}" if abs(tempo - 1) > 0.01 else "anull"
    subprocess.run([ffmpeg(), "-y", "-i", str(src), "-filter:a", af, "-ar", "44100", "-ac", "2",
                    str(dst)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def build_shot(idx, kind, src, duration, narration, chapter, out_dir):
    """生成单段视频（含配音）。kind: still / clip"""
    audio_raw = out_dir / f"voice_{idx}.mp3"
    audio = out_dir / f"audio_{idx}.m4a"
    seg = out_dir / f"seg_{idx}.mp4"

    tts(narration, audio_raw)
    fit_audio(audio_raw, audio, duration)

    if kind == "clip":
        # 录屏压到 1920×994，底部 86px 留给章节条，避免遮住曲线坐标轴
        filt = (f"[0:v]scale=1920:994:flags=lanczos,setsar=1,fps=25,trim=duration={duration:.2f},"
                f"setpts=PTS-STARTPTS[base]")
        subprocess.run([ffmpeg(), "-y", "-i", str(src), "-i", str(BUILD / f"band_{idx}.png"),
                        "-filter_complex",
                        filt + ";[base]pad=1920:1080:0:0:color=white[p];[p][1:v]overlay=0:994:format=auto[v]",
                        "-map", "[v]", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-r", "25", str(seg)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run([ffmpeg(), "-y", "-loop", "1", "-t", f"{duration:.2f}", "-i", str(src),
                        "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "20", "-preset", "medium",
                        "-pix_fmt", "yuv420p", "-r", "25", str(seg)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    seg_av = out_dir / f"segav_{idx}.mp4"
    subprocess.run([ffmpeg(), "-y", "-i", str(seg), "-i", str(audio),
                    "-filter_complex", "[1:a]apad[a]", "-map", "0:v", "-map", "[a]",
                    "-t", f"{duration:.2f}", "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
                    "-ar", "44100", "-ac", "2", str(seg_av)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return seg_av


def main():
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    e = {}
    for name in ["E1_smallcnn_scratch_lr1e-3_noaug", "E2_resnet18_lr1e-3_aug",
                 "E3_resnet18_lr3e-4_aug", "E4_mobilenet_v3_lr5e-4_aug"]:
        p = ROOT / "runs" / name / "result.json"
        e[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    acc, f1, top3, ms = metrics["accuracy"] * 100, metrics["macro_f1"], metrics["top3_accuracy"] * 100, metrics["inference_ms"]
    e1a = e["E1_smallcnn_scratch_lr1e-3_noaug"].get("best_val_acc", 0) * 100
    e2a = e["E2_resnet18_lr1e-3_aug"].get("best_val_acc", 0) * 100
    e3a = e["E3_resnet18_lr3e-4_aug"].get("best_val_acc", 0) * 100
    e4a = e["E4_mobilenet_v3_lr5e-4_aug"].get("best_val_acc", 0) * 100

    tb_clip = BUILD / "clip_tb.mp4"
    web_clip = BUILD / "clip_web.mp4"
    assert tb_clip.exists() and web_clip.exists(), "请先运行 tools/record_clips.py tb / web"

    shots = []

    # 1) 0:00-0:40 项目名称、选题、应用场景
    shots.append(dict(
        kind="still", d=40, chapter="00:00  项目名称与选题",
        frame=dict(chapter="00:00  项目名称与选题", title="基于 PyTorch 的垃圾图像智能分类系统",
                   bullets=["选题：六类生活垃圾图像分类（纸板 / 玻璃 / 金属 / 纸张 / 塑料 / 其他垃圾）",
                            "场景：社区投放点、校园分类箱、环卫初筛的辅助判断",
                            "目标：上传或拍摄一张照片，秒级给出类别、置信度与投放建议"],
                   image=OUT / "frontend_result.png",
                   caption="系统实际运行界面（后端 Flask + 模型 best_model.pth）"),
        narration="大家好，我演示的项目是“基于 PyTorch 的垃圾图像智能分类系统”。"
                  "选题属于计算机视觉中的图像分类任务，对象是日常生活垃圾的六个类别：纸板、玻璃、金属、纸张、塑料和其他垃圾。"
                  "应用场景是社区投放点、校园分类箱和环卫初筛环节，用户拍一张照片或者上传一张图片，"
                  "系统在毫秒级给出类别、置信度和投放建议，降低人工判断的成本。"
                  "整个系统由 PyTorch 训练出的模型、Flask 后端和网页前端组成，下面按数据集、模型、训练曲线、前端演示和结果分析依次汇报。",
    ))

    # 2) 0:40-1:40 数据集
    shots.append(dict(
        kind="still", d=60, chapter="00:40  数据集来源与样例",
        frame=dict(chapter="00:40  数据集来源与样例", title="真实数据集：6 类 2527 张实拍图片",
                   bullets=["来源：飞桨 AI Studio 公开数据集「垃圾分类」（编号 128034）",
                            "清洗：PIL 逐张校验全部可解码，MD5 去重剔除 3 张跨类别重复图",
                            "划分：2524 张按 7 : 1.5 : 1.5 分层切分为 1768 / 378 / 378"],
                   image=OUT / "fig_samples.png",
                   caption="12 张样例图全部来自测试集，未参与训练"),
        narration="数据集来自飞桨 AI Studio 公开数据集“垃圾分类”，编号一二八零三四，压缩包约八十六兆，"
                  "包含纸板、玻璃、金属、纸张、塑料和其他垃圾六个类别，共两千五百二十七张真实拍摄照片。"
                  "数据清洗时我先用 PIL 逐张校验，全部可以正常解码；再用 MD5 做全量比对，发现三张跨类别完全重复的图片并剔除，"
                  "最终得到两千五百二十四张。然后按类别分层抽样，以七比一点五比一点五切分为训练集一千七百六十八张、"
                  "验证集三百七十八张、测试集三百七十八张，每张图片的 MD5 都写进 dataset_manifest 清单，"
                  "保证同一张图不会同时出现在不同集合里。画面上是每个类别各两张样例，全部取自测试集。",
    ))

    # 3) 1:40-2:40 模型与训练
    shots.append(dict(
        kind="still", d=60, chapter="01:40  模型结构与训练参数",
        frame=dict(chapter="01:40  模型结构与训练参数", title="ResNet18 / MobileNetV3 迁移学习",
                   bullets=["结构：ImageNet 预训练主干 + Dropout + 全连接六分类头",
                            "训练：AdamW + 余弦退火 + 标签平滑 0.1 + 混合精度，固定种子 42",
                            "参数：输入 224×224，batch 32，20 轮，最终模型 1,524,006 参数"],
                   image=OUT / "fig_model_arch.png",
                   caption="ResNet18 网络结构（E2/E3 使用）；最终部署为轻量主干 MobileNetV3-Small"),
        narration="模型部分以迁移学习为主。我在 ImageNet 预训练主干上替换最后的全连接层，输出改成六类，"
                  "前面加上 Dropout 抑制过拟合。训练流程是标准的 PyTorch 循环：取一个批次的数据做前向传播，"
                  "用带零点一标签平滑的交叉熵算损失，再反向传播更新参数，每个轮次结束在验证集上评估一次。"
                  "关键参数是：输入二百二十四乘二百二十四，批大小三十二，训练二十轮，优化器 AdamW，学习率余弦退火，"
                  "并开启混合精度，训练前固定随机种子四十二，全部超参数都写进了 runs 目录下的 config 文件。"
                  "为了对比，我一共训练了四组实验：从零训练的自定义 CNN、两组不同学习率的 ResNet18，以及轻量的 MobileNetV3。",
    ))

    # 4) 2:40-4:00 TensorBoard 真实录屏
    tb_len = min(80, probe(tb_clip))
    shots.append(dict(
        kind="clip", d=tb_len, chapter="02:40  TensorBoard Loss 曲线对比",
        src=tb_clip,
        narration="现在打开 TensorBoard，这是命令行 tensorboard 启动后现场展示的界面，左侧是四个实验的运行列表。"
                  "我把 Loss 分组展开，可以看到上排是训练集 Loss、下排是验证集 Loss，四条颜色对应四组实验。"
                  "灰色是从零训练的自定义 CNN，Loss 一直在一点二以上，明显收敛不下去；红色和蓝色是 ResNet18，"
                  "使用预训练权重，前几轮 Loss 就快速下降；橙色是 MobileNetV3，参数量只有一百五十二万，下降同样很快。"
                  "把鼠标移到曲线上可以直接读出每一步的数值，比如最终模型在第十二轮验证 Loss 最低，验证集 Macro F1 达到零点九一九。"
                  "另外从曲线也能看出学习率的影响：E2 用千分之一，下降最快但中途有明显抖动；E3 先用万分之三并冻结主干三轮，曲线更平滑。",
    ))

    # 5) 4:00-6:00 前端真实录屏
    web_len = min(120, probe(web_clip))
    shots.append(dict(
        kind="clip", d=web_len, chapter="04:00  前端上传与实时预测",
        src=web_clip,
        narration="接下来演示系统启动后的真实运行过程。运行 app 点 py 之后，浏览器打开五千端口，页面顶部显示服务正常、模型已加载。"
                  "我先选择一张测试集里的塑料瓶照片，页面立刻显示预览和文件信息，然后点击开始识别。"
                  "请求以表单形式发到后端，由 PyTorch 模型完成预处理和前向计算，返回结果是塑料，置信度百分之九十一点五，"
                  "Top-3 概率条可以看到纸板和其他垃圾只有百分之二左右，右上角显示这次推理耗时。下面还有对应的投放建议。"
                  "接着我点击页面底部的样例缩略图，会自动上传并完成一次识别，这次判断为纸板。"
                  "如果上传的不是图片文件，前端会直接拦截并给出中文提示，后端也会再校验一次类型、大小和可解码性。"
                  "页面下方还能看到当前模型名称、参数量、输入尺寸和运行设备，这些信息都由接口实时返回。",
    ))

    # 6) 6:00-7:00 测试结果与错误案例（两张图，各 30 秒）
    for i, (img_path, cap, narr, ch) in enumerate([
        (OUT / "confusion_matrix.png", "测试集混淆矩阵（378 张，对角线为正确分类）",
         f"最后看测试结果。最终模型在三百七十八张测试集图片上准确率 {acc:.1f}%，Macro F1 是 {f1:.3f}，"
         f"Top-3 准确率 {top3:.1f}%，单张图片推理约 {ms:.0f} 毫秒。混淆矩阵的对角线是正确分类，"
         f"错误主要集中在玻璃被误判成塑料、金属被误判成纸板，以及其他垃圾与纸、塑料之间的混淆。",
         "06:00  测试结果与混淆矩阵"),
        (OUT / "case_gallery.png", "正确预测（绿色边框）与错误预测（红色边框）",
         "画面上排是置信度最高的几个正确案例，绿色边框；下排红框是两个典型错误。"
         "错误原因主要有三点：数据量只有两千多张，模型对瓦楞层、玻璃反光这类材质细节学到的特征有限；"
         "其他垃圾只有一百三十七张，类内差异大；部分图片只拍了物体局部，缺少可判断材质的整体形状。"
         "改进方向是补充少数类样本、尝试更强的骨干网络、加入 Grad-CAM 可解释性，以及导出 ONNX 做部署加速。",
         "06:30  错误案例与改进方向"),
    ]):
        shots.append(dict(kind="still", d=30, chapter=ch,
                          frame=dict(chapter=ch,
                                     title="测试结果与错误案例分析" if i == 0 else "错误案例与改进方向",
                                     bullets=[f"测试集 {metrics['num_samples']} 张：准确率 {acc:.2f}%，Macro-F1 {f1:.4f}，Top-3 {top3:.2f}%"]
                                     if i == 0 else ["主要错误：玻璃↔塑料、金属↔纸板、其他垃圾↔纸/塑料",
                                                     "原因：数据量小、类别不均衡、局部拍摄缺少材质线索"],
                                     image=img_path, caption=cap),
                          narration=narr))

    # 7) 7:00-7:45 总结
    shots.append(dict(
        kind="still", d=45, chapter="07:00  总结",
        frame=dict(chapter="07:00  项目总结", title="总结：一条完整可复现的深度学习链路",
                   bullets=["数据：真实公开数据，清洗、去重、分层划分并留痕",
                            "训练：4 组对照实验，TensorBoard 记录曲线，按验证集 Macro-F1 选模型",
                            "系统：Flask + 原生前端，上传即预测，结果全部来自实时推理"],
                   image=OUT / "fig_curves.png",
                   caption="四组实验 Loss 与验证准确率曲线（由 runs 下的 history.csv 重绘）"),
        narration="总结一下。这个项目完整走通了真实数据、清洗划分、迁移学习训练、TensorBoard 监控、测试评估到网页实时推理的全链路。"
                  "四组对照实验显示，预训练迁移的收益远大于从零训练，最终按验证集 Macro-F1 选出的 MobileNetV3 模型"
                  f"在测试集上准确率 {acc:.2f}%，参数量只有一百五十二万，单张推理六毫秒左右。"
                  "不足是数据规模和类别均衡还有提升空间，模型也只做整图分类。我会继续补充难例、尝试更强主干并做部署优化。"
                  "以上就是我的演示，谢谢观看。",
    ))

    total = sum(s["d"] for s in shots)
    print(f"[计划] {len(shots)} 段，合计 {total / 60:.2f} 分钟")

    seg_files = []
    srt = []
    cursor = 0.0
    for idx, s in enumerate(shots, 1):
        if s["kind"] == "still":
            src = BUILD / f"frame_{idx}.png"
            s["frame_image"] = s.pop("frame")
            Image.save(src) if False else None
            img = frame(**s["frame_image"])
            img.save(src)
        else:
            src = s["src"]
            overlay_band(s["chapter"]).save(BUILD / f"band_{idx}.png")

        seg = build_shot(idx, s["kind"], src, s["d"], s["narration"], s["chapter"], BUILD)
        seg_files.append(seg)

        def ts(t):
            h, rem = divmod(t, 3600)
            m, sec = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",")

        srt.append(f"{idx}\n{ts(cursor)} --> {ts(cursor + s['d'] - 0.2)}\n{s['narration']}\n")
        cursor += s["d"]
        print(f"[{idx:02d}/{len(shots)}] {s['chapter']}  {s['d']:.0f}s")

    listing = BUILD / "concat.txt"
    listing.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files), encoding="utf-8")
    out = VID / "25大数据本_垃圾分类图像识别系统_演示.mp4"
    subprocess.run([ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                    "-c", "copy", str(out)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (VID / "演示字幕.srt").write_text("\n".join(srt), encoding="utf-8")
    print(f"[完成] {out}  {probe(out) / 60:.2f} 分钟  {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
