"""生成带配音的演示视频（MP4）。

流程：为每个场景渲染 1920×1080 讲解画面（真实截图 + 标题 + 字幕）→ edge-tts 生成中文配音 →
按配音时长生成分段视频 → 拼接为完整 MP4，并附带 SRT 字幕。

用法：python tools/make_video.py
输出：video/垃圾分类识别系统_演示.mp4、video/演示字幕.srt
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
TMP = ROOT / "video" / "_build"
VID.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)

W, H = 1920, 1080
FONT = "C:/Windows/Fonts/msyh.ttc"
FONT_B = "C:/Windows/Fonts/msyhbd.ttc"
CONSOLE = "C:/Windows/Fonts/consola.ttf"

INK = (20, 32, 27)
INK2 = (74, 90, 83)
GREEN = (31, 122, 77)
GREEN_D = (16, 53, 42)
BG = (246, 248, 247)
WHITE = (255, 255, 255)

FFMPEG = None


def ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def font(size, bold=False):
    path = FONT_B if bold and Path(FONT_B).exists() else FONT
    return ImageFont.truetype(path, size)


def wrap(text, fnt, max_width, draw):
    lines, cur = [], ""
    for ch in text:
        if ch == "\n":
            lines.append(cur)
            cur = ""
            continue
        if draw.textlength(cur + ch, font=fnt) > max_width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def rounded(img, radius=18):
    from PIL import ImageDraw as ID
    mask = Image.new("L", img.size, 0)
    ID.Draw(mask).rounded_rectangle([0, 0, img.size[0], img.size[1]], radius=radius, fill=255)
    out = Image.new("RGBA", img.size)
    out.paste(img, (0, 0), mask)
    return out


def frame(idx, total, title, bullets, image_path, caption, subtitle_lines):
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    d.rectangle([0, 0, W, 6], fill=GREEN)
    d.text((64, 30), "基于 PyTorch 的垃圾图像智能分类系统", font=font(25, True), fill=GREEN_D)
    d.text((W - 200, 32), f"{idx:02d} / {total:02d}", font=font(23), fill=(155, 170, 163))

    d.text((64, 84), title, font=font(50, True), fill=INK)
    y = 156
    for b in bullets:
        d.ellipse([68, y + 13, 80, y + 25], fill=GREEN)
        d.text((96, y), b, font=font(24), fill=INK2)
        y += 40

    sub_h = 33 * len(subtitle_lines) + 40
    img_top = y + 18
    avail_h = H - img_top - sub_h - 56
    cap_h = 34 if caption else 0

    if image_path and Path(image_path).exists():
        shot = Image.open(image_path).convert("RGB")
        scale = min(1720 / shot.width, (avail_h - cap_h) / shot.height)
        shot = shot.resize((max(1, int(shot.width * scale)), max(1, int(shot.height * scale))), Image.LANCZOS)
        shot = rounded(shot, 12)
        x = (W - shot.width) // 2
        top = img_top + max(0, (avail_h - cap_h - shot.height) // 2)
        shadow = Image.new("RGBA", (shot.width + 20, shot.height + 20), (0, 0, 0, 0))
        ImageDraw.Draw(shadow).rounded_rectangle([10, 10, shot.width + 10, shot.height + 10], 14,
                                                 fill=(20, 40, 30, 24))
        img.paste(shadow.convert("RGB"), (x - 10, top - 10), shadow)
        img.paste(shot, (x, top), shot)
        if caption:
            d = ImageDraw.Draw(img)
            d.text((x, top + shot.height + 6), caption, font=font(20), fill=(132, 150, 142))

    d = ImageDraw.Draw(img)
    base_y = H - 24 - 33 * len(subtitle_lines)
    d.rectangle([0, base_y - 24, W, H], fill=(255, 255, 255))
    d.line([(0, base_y - 24), (W, base_y - 24)], fill=(222, 230, 225), width=2)
    yy = base_y
    for line in subtitle_lines:
        tw = d.textlength(line, font=font(25))
        d.text(((W - tw) / 2, yy), line, font=font(25), fill=(45, 58, 51))
        yy += 33
    return img


def tts(text, path, voice="zh-CN-YunxiNeural", rate="+6%"):
    subprocess.run([sys.executable, "-m", "edge_tts", "--text", text, "--voice", voice,
                    "--rate", rate, "--write-media", str(path)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def duration_of(path):
    out = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


def build_scenes():
    metrics = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
    demo = json.loads((OUT / "inference_demo.json").read_text(encoding="utf-8"))
    exps = {}
    for exp in ["E1_smallcnn_scratch_lr1e-3_noaug", "E2_resnet18_lr1e-3_aug",
                "E3_resnet18_lr3e-4_aug", "E4_mobilenet_v3_lr5e-4_aug"]:
        p = ROOT / "runs" / exp / "result.json"
        exps[exp] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

    e1 = exps["E1_smallcnn_scratch_lr1e-3_noaug"]
    e2 = exps["E2_resnet18_lr1e-3_aug"]
    e3 = exps["E3_resnet18_lr3e-4_aug"]
    code_map = {"E1_smallcnn_scratch_lr1e-3_noaug": "E1", "E2_resnet18_lr1e-3_aug": "E2",
                "E3_resnet18_lr3e-4_aug": "E3", "E4_mobilenet_v3_lr5e-4_aug": "E4"}
    best_exp = max(exps, key=lambda k: exps[k].get("best_val_f1", -1) if exps[k] else -1)
    best_code = code_map.get(best_exp, best_exp)
    acc = metrics["accuracy"] * 100
    f1 = metrics["macro_f1"]
    top3 = metrics["top3_accuracy"] * 100
    best_ep = e3.get("best_epoch", 15)
    inf_ms = demo.get("inference_ms", 10)
    top1_zh = demo["top1"]["class_zh"]
    top1_p = demo["top1"]["probability"] * 100

    scenes = [
        dict(
            title="项目演示：垃圾图像智能分类系统",
            bullets=["真实数据 · 真实训练 · 真实推理", "PyTorch + TensorBoard + Flask"],
            image=OUT / "fig_system_flow.png",
            caption="图 系统总体流程：离线训练链路与在线推理链路",
            narration="大家好，下面演示我的深度学习案例项目，基于 PyTorch 的垃圾图像智能分类系统。"
                      "项目使用公开的真实数据集，完成了数据清洗与划分、ResNet18 迁移学习训练、"
                      "TensorBoard 训练曲线记录，以及 Flask 实时推理前端，一共四组对照实验。"
                      "演示分为数据集、模型与训练、TensorBoard 曲线、前端推理、结果分析五个部分。",
        ),
        dict(
            title="1. 项目背景与系统流程",
            bullets=["目标：单张图片判断六类生活垃圾", "离线链路训练 + 在线链路实时推理"],
            image=OUT / "fig_system_flow.png",
            caption="离线：数据清洗 → 划分 → 训练 → 存权重；在线：上传 → 预处理 → 模型 → 结果",
            narration="生活垃圾的材质判断依赖经验，人工分拣容易受光线和速度影响。"
                      "本项目针对纸板、玻璃、金属、纸张、塑料、其他垃圾六类常见废弃物，做单张图片的整图分类。"
                      "系统分为两条链路：离线链路读取真实数据，清洗去重、分层划分、训练模型，把指标写入 TensorBoard，"
                      "最佳权重保存为 best_model 点 pth；在线链路由 Flask 加载这个权重，前端上传图片后，"
                      "服务端完成预处理与前向计算，把类别、置信度、Top-3 和耗时返回页面。",
        ),
        dict(
            title="2. 数据集来源与真实样例",
            bullets=["AI Studio 公开数据集（编号 128034）", "6 类共 2527 张实拍图片"],
            image=OUT / "fig_samples.png",
            caption="12 张样例全部来自测试集，未参与训练",
            narration="数据来自飞桨 AI Studio 公开数据集，编号一二八零三四，六个类别一共两千五百二十七张真实拍摄照片，"
                      "压缩包约八十六兆。图上展示的是每类各两张样例，全部取自测试集。"
                      "可以看到都是白纸或桌面背景下的单件垃圾，比如纸箱、玻璃瓶、易拉罐、打印纸、饮料瓶和塑料袋，"
                      "没有任何程序生成的合成数据。数据集只用于课程学习，不用于商业用途。",
        ),
        dict(
            title="3. 数据清洗与集合划分",
            bullets=["PIL 校验 + MD5 去重，剔除 3 张重复图", "7 : 1.5 : 1.5 分层划分，共 2524 张"],
            image=OUT / "fig_class_dist.png",
            caption="训练集 1768 / 验证集 378 / 测试集 378，类别存在不均衡",
            narration="清洗时先用 PIL 逐张校验能否解码，两千五百二十七张全部通过；再用 MD5 全局去重，"
                      "发现三张跨类别完全重复的图片，剔除后得到两千五百二十四张。"
                      "然后按类别分层抽样，以七比一点五比一点五划分出训练集一千七百六十八张、验证集三百七十八张、"
                      "测试集三百七十八张，并输出 dataset_manifest 清单，每张图的哈希都记录下来，"
                      "保证同一张图片不会同时出现在不同集合里。右图可以看到类别不均衡，纸张五百九十四张，"
                      "其他垃圾只有一百三十七张，这也是后面用 Macro F1 选模型的原因。",
        ),
        dict(
            title="4. 模型结构与训练策略",
            bullets=["ResNet18 + ImageNet 预训练，替换为六分类头", "全参微调为主，另设冻结主干对照实验"],
            image=OUT / "fig_model_arch.png",
            caption="参数量 1118 万；AdamW + 余弦退火 + 标签平滑 0.1 + 混合精度",
            narration="模型采用 ResNet18，加载 ImageNet 预训练权重，把最后的全连接层换成 Dropout 加五一二到六的分类头，"
                      "总参数量约一千一百一十八万。最终采用的训练方案是加载预训练权重后直接全参微调，学习率十的负三次方；"
                      "另外安排了一组对照实验 E3，先冻结主干三个轮次只训练分类头，再解冻并用更低的学习率微调。"
                      "两组实验共用同一数据划分和优化设置：AdamW、余弦退火、带零点一标签平滑的交叉熵，并使用混合精度训练。"
                      "训练前固定随机种子四十二，所有超参数都写入 runs 目录下的 config 文件。",
        ),
        dict(
            title="5. TensorBoard：四组实验 Loss 对比",
            bullets=["runs/ 下 4 个独立日志目录", "tensorboard --logdir runs --port 6006"],
            image=OUT / "tb_loss_compare.png",
            caption="E1 从零训练无增强；E2/E3 ResNet18；E4 MobileNetV3，均使用增强与预训练",
            narration="训练全程启用 TensorBoard，每个实验一个独立目录，记录训练 Loss、验证 Loss、训练准确率、"
                      "验证准确率、验证 Macro F1 和学习率。这里现场打开的就是 SCALARS 页面，四条曲线分别是四组实验。"
                      f"E1 是从零训练的自定义 CNN，关闭了数据增强，收敛最慢，验证准确率只有 {e1.get('best_val_acc', 0) * 100:.1f}%；"
                      "E2、E3、E4 都用了预训练，前几轮 Loss 就快速下降。E2 用十的负三次方直接全参微调，下降最快，"
                      "但第七轮出现了一次明显波动，验证 Loss 从零点八二反弹到一点一九；"
                      "E3 先冻结三轮再用三乘十的负四次方微调，曲线更平滑，但在有限的轮次里没有追上 E2。",
        ),
        dict(
            title=f"6. 最终模型 {best_code} 的训练曲线",
            bullets=[f"验证集指标最佳轮次：第 {best_ep} 轮", "训练与验证同步收敛，末段轻微过拟合"],
            image=OUT / "tb_single_exp.png",
            caption=f"{best_exp}：Loss/train 与 Loss/val",
            narration=f"这是最终模型 {best_code} 的完整曲线。训练 Loss 从 1.30 一路降到 0.47，"
                      f"验证 Loss 在第 {best_ep} 轮达到最低，之后小幅回升，同时训练准确率继续上升并明显高于验证准确率，"
                      f"说明开始出现轻微过拟合，因此最佳权重取第 {best_ep} 轮而不是最后一轮。"
                      "学习率面板可以看到余弦退火让学习率逐步衰减到初始值的百分之一。"
                      "整个训练过程大部分轮次平稳下降，说明批大小与学习率的选择基本合理。",
        ),
        dict(
            title="7. 实验结果与混淆矩阵",
            bullets=[f"测试集准确率 {acc:.2f}% ，Macro-F1 {f1:.4f} ，Top-3 {top3:.2f}%",
                     "错误集中在材质相近的类别之间"],
            image=OUT / "confusion_matrix.png",
            caption="测试集 378 张图片，未参与训练与调参",
            narration=f"最终模型在三百七十八张测试集图片上，准确率 {acc:.1f}%，Macro F1 是 {f1:.3f}，"
                      f"Top-3 准确率 {top3:.1f}%，单张图片在显卡上推理约 {inf_ms:.0f} 毫秒。"
                      "混淆矩阵的对角线是正确分类，可以看到错误主要集中在纸张和纸板之间、塑料和玻璃之间，"
                      "以及其他垃圾被误判成纸或塑料，这些都是材质外观高度相近的类别，属于合理误差范围。",
        ),
        dict(
            title="8. 前端演示：上传图片与实时推理",
            bullets=["Flask 加载 best_model.pth，页面结果全部来自实时推理", "支持上传、拖拽、粘贴与摄像头"],
            image=OUT / "frontend_result.png",
            caption="上传测试集塑料瓶图片，返回类别、置信度、Top-3 概率与推理耗时",
            narration=f"现在演示前端。运行 app.py 之后打开五千端口，页面顶部显示服务正常、模型已加载，"
                      f"底部能看到模型结构、参数量和运行设备。我选择一张测试集里的塑料瓶图片，预览之后点击开始识别，"
                      f"服务端加载 best_model 点 pth 实时推理，返回塑料，置信度 {top1_p:.1f}%，"
                      "以及 Top-3 概率条和毫秒级推理耗时，下面还有投放建议。"
                      "换一张纸板图片再识别一次，结果同样来自模型实时计算，页面没有写死任何类别。"
                      "除了上传，还支持拖拽、Ctrl 加 V 粘贴和摄像头拍照。",
        ),
        dict(
            title="9. 异常输入处理",
            bullets=["文件类型、大小、可解码性四级校验", "前后端都给出明确中文提示"],
            image=OUT / "frontend_error.png",
            caption="上传非图片文件时前端直接拦截，后端同样做二次校验",
            narration="系统对异常输入也做了处理。前端会校验扩展名、MIME 类型和文件大小，超过八兆或者不是图片，"
                      "会直接提示，不会发起请求；后端还会再检查一次文件类型、文件是否为空、图片能否解析，"
                      "分别返回对应的错误码和中文原因。这样即使绕过前端直接调用接口，也不会把异常数据送进模型。",
        ),
        dict(
            title="10. 不足与改进方向",
            bullets=["数据量偏小、类别不均衡（trash 仅 137 张）", "仅整图分类，未做目标检测与定位"],
            image=OUT / "case_wrong.png",
            caption="典型错误案例：材质相近导致误判",
            narration="项目也有明显的不足。第一，数据量只有两千多张，其他垃圾这一类只有一百三十七张，"
                      "导致这一类召回率偏低，后续可以补充真实拍摄的难例，并对少数类做重采样。"
                      "第二，模型只做整图分类，一张图片里有多件垃圾时只能给出一个结果，可以考虑改成目标检测。"
                      "第三，还可以尝试 EfficientNet 等更强的主干，加入 Grad-CAM 热力图提升可解释性，"
                      "并导出 ONNX 做 CPU 加速部署。以上就是本次演示的全部内容，谢谢观看。",
        ),
    ]

    # 计算字幕行
    dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    for s in scenes:
        s["subtitle"] = wrap(s["narration"], font(25), 1760, dummy)[:3]
    return scenes


def main():
    scenes = build_scenes()
    total = len(scenes)
    seg_files = []
    srt_lines = []
    cursor = 0.0

    for i, s in enumerate(scenes, 1):
        frame_path = TMP / f"frame_{i:02d}.png"
        audio_path = TMP / f"voice_{i:02d}.mp3"
        seg_path = TMP / f"seg_{i:02d}.mp4"

        img = frame(i, total, s["title"], s["bullets"], s["image"], s["caption"], s["subtitle"])
        img.save(frame_path)

        tts(s["narration"], audio_path)
        dur = duration_of(audio_path) + 0.9

        subprocess.run([
            ffmpeg(), "-y", "-loop", "1", "-i", str(frame_path), "-i", str(audio_path),
            "-filter_complex", f"[1:a]apad[a]",
            "-map", "0:v", "-map", "[a]",
            "-t", f"{dur:.2f}", "-r", "25",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "44100", "-ac", "2",
            str(seg_path),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        seg_files.append(seg_path)

        # SRT
        def ts(t):
            h, rem = divmod(t, 3600)
            m, sec = divmod(rem, 60)
            return f"{int(h):02d}:{int(m):02d}:{sec:06.3f}".replace(".", ",")

        srt_lines.append(f"{i}\n{ts(cursor)} --> {ts(cursor + dur - 0.3)}\n{s['narration']}\n")
        cursor += dur
        print(f"[{i:02d}/{total}] {s['title']}  配音 {dur:.1f}s")

    list_file = TMP / "concat.txt"
    list_file.write_text("\n".join(f"file '{p.as_posix()}'" for p in seg_files), encoding="utf-8")

    out_path = VID / "垃圾分类识别系统_演示.mp4"
    subprocess.run([ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(list_file),
                    "-c", "copy", str(out_path)], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    (VID / "演示字幕.srt").write_text("\n".join(srt_lines), encoding="utf-8")

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"[完成] {out_path}  时长 {cursor / 60:.1f} 分钟  大小 {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
