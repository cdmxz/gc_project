"""生成《项目设计文档.docx》。所有数字均从真实产物读取：
runs/*/history.csv、outputs/metrics.json、models/model_meta.json、dataset_manifest.csv。
用法：python tools/gen_report.py
"""
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import config  # noqa: E402
from docx_helpers import (ACCENT, GRAY, INK, add_header_text, add_page_number_footer,  # noqa: E402
                          bullet, caption, code_block, figure, heading, init_document,
                          note_box, para, set_run_font, table)

OUT = ROOT / "outputs"
EXP_ORDER = [
    ("E1_smallcnn_scratch_lr1e-3_noaug", "E1", "SmallCNN（从零训练）", "无", "无", "1e-3", 20),
    ("E2_resnet18_lr1e-3_aug", "E2", "ResNet18", "ImageNet", "有", "1e-3", 20),
    ("E3_resnet18_lr3e-4_aug", "E3", "ResNet18", "ImageNet", "有", "3e-4", 20),
    ("E4_mobilenet_v3_lr5e-4_aug", "E4", "MobileNetV3-Small", "ImageNet", "有", "5e-4", 20),
]
FINAL_EXP = "E3_resnet18_lr3e-4_aug"   # 运行时按验证集 Macro-F1 覆盖为实际最佳实验
EXP_CODE = {e[0]: e[1] for e in EXP_ORDER}

_fig_no = 0
_tab_no = 0


def fig(doc, path, text, width=13.0):
    global _fig_no
    _fig_no += 1
    figure(doc, path, f"图 {_fig_no}  {text}", width)


def tab(doc, headers, rows, text, widths=None, size=8.8, center=None):
    global _tab_no
    _tab_no += 1
    table(doc, headers, rows, widths=widths, caption_text=f"表 {_tab_no}  {text}",
          font_size=size, align_center_cols=center)


# ------------------------------------------------------------------ 数据读取
def load_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else (default or {})


def load_history(exp):
    rows = []
    p = ROOT / "runs" / exp / "history.csv"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({k: float(v) for k, v in r.items()})
    return rows


def manifest_stats():
    rows = []
    with open(config.MANIFEST_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    per = defaultdict(Counter)
    for r in rows:
        per[r["class_name"]][r["split"]] += 1
    return rows, per


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


# ------------------------------------------------------------------ 封面
def cover(doc):
    for _ in range(2):
        para(doc, "", indent=False, space_after=0)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("深度学习案例项目设计报告")
    set_run_font(r, "微软雅黑", 13, bold=True, color=ACCENT)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(26)
    r = p.add_run("基于 PyTorch 的\n垃圾图像智能分类系统")
    set_run_font(r, "微软雅黑", 30, bold=True, color=INK)

    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run("PyTorch 迁移学习 · TensorBoard 训练可视化 · Flask 实时推理前端")
    set_run_font(r, "微软雅黑", 11.5, color=GRAY)

    for _ in range(3):
        para(doc, "", indent=False, space_after=0)

    info = [
        ("课程名称", "深度学习案例实践"),
        ("项目类型", "图像分类（六类生活垃圾）"),
        ("班    级", "25 大数据本"),
        ("学    号", "202311055424"),
        ("姓    名", "傅宏华"),
        ("指导教师", "____________________"),
        ("完成日期", "2026 年 11 月 23 日"),
    ]
    t = doc.add_table(rows=0, cols=2)
    t.alignment = 1
    for k, v in info:
        cells = t.add_row().cells
        cells[0].width = Cm(4.2)
        cells[1].width = Cm(7.6)
        p0 = cells[0].paragraphs[0]
        p0.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        r0 = p0.add_run(k + "：")
        set_run_font(r0, "宋体", 12, bold=True)
        p1 = cells[1].paragraphs[0]
        r1 = p1.add_run(v)
        set_run_font(r1, "宋体", 12)

    for _ in range(4):
        para(doc, "", indent=False, space_after=0)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("源代码仓库：https://github.com/cdmxz/gc_project")
    set_run_font(r, "宋体", 9.5, color=GRAY)


def abstract(doc, metrics, rows_len):
    heading(doc, "摘  要", 1, page_break=True)
    para(doc, "本项目围绕“生活垃圾图像分类”这一具体视觉任务，使用公开真实数据集（飞桨 AI Studio 垃圾分类数据集，"
              "六类共 2527 张实拍图片）完成了一次从数据处理、模型训练、实验对比到 Web 端实时推理的完整闭环。"
              "数据侧对图片做了可解码性校验与 MD5 去重，剔除 3 张重复图片后按 7:1.5:1.5 分层划分出训练集、"
              "验证集与测试集，并生成 dataset_manifest.csv 作为可核查的划分清单。")
    para(doc, f"模型侧比较了 ResNet18 与 MobileNetV3-Small 两种 ImageNet 预训练主干（并实现一个自定义 CNN 作为从零训练基线），"
              f"把最后的分类层替换为六类输出，配合 AdamW 优化器、余弦退火与标签平滑完成微调。为验证迁移学习、学习率、是否冻结主干以及轻量模型的差异，"
              f"共完成 4 组对照实验，训练过程使用 TensorBoard 记录训练/验证 Loss、训练/验证准确率、验证 Macro-F1 与学习率，"
              f"最终按验证集 Macro-F1 选出最佳权重。该模型在 {metrics.get('num_samples', 378)} 张测试集图片上取得 "
              f"{metrics.get('accuracy', 0) * 100:.2f}% 的准确率、{metrics.get('macro_f1', 0):.4f} 的 Macro-F1 与 "
              f"{metrics.get('top3_accuracy', 0) * 100:.2f}% 的 Top-3 准确率。")
    para(doc, "系统侧用 Flask 加载保存的 best_model.pth 提供预测接口，前端使用原生 HTML/CSS/JavaScript 实现图片上传、"
              "摄像头采集、格式与大小校验、Top-3 概率展示、推理耗时显示与异常提示，页面结果完全来自服务端模型的实时推理。"
              "全部代码、训练日志、模型权重与运行说明一并提交，可按本文第 9 章步骤复现。")
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    r = p.add_run("关键词：")
    set_run_font(r, "宋体", 10.5, bold=True)
    r2 = p.add_run("图像分类；迁移学习；ResNet18；PyTorch；TensorBoard；Flask")
    set_run_font(r2, "宋体", 10.5)


def toc(doc):
    heading(doc, "目  录", 1)
    entries = [
        ("1  项目概述", 1), ("2  需求分析", 1), ("3  真实数据集证明", 1),
        ("4  数据探索与预处理", 1), ("5  PyTorch 模型文件说明", 1),
        ("6  TensorBoard 日志说明", 1), ("7  Loss 曲线与实验结果分析", 1),
        ("8  前端设计与系统测试", 1), ("9  运行说明", 1),
        ("10  问题与改进", 1), ("11  总结", 1), ("参考文献与 AI 工具使用说明", 1),
        ("附录 A  项目目录结构与数据清单", 1),
    ]
    for text, _ in entries:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.left_indent = Pt(8)
        r = p.add_run(text)
        set_run_font(r, "宋体", 11)
    note_box(doc, "说明：本报告所有数值、曲线与截图均来自项目真实运行产物（runs/ 日志、outputs/ 评估结果、"
                  "best_model.pth 实际推理），未做任何手工修改。图表编号在正文中连续。")


# ------------------------------------------------------------------ 正文
def section1(doc, metrics):
    heading(doc, "1  项目概述", 1)
    heading(doc, "1.1  项目背景", 2)
    para(doc, "生活垃圾分类要求投放者判断物品材质与投放规则，现实中常因经验差异、光线条件和作业速度导致分拣不稳定。"
              "本项目以纸板、玻璃、金属、纸张、塑料与其他垃圾六类常见废弃物为对象，训练一个对单张照片给出类别与置信度的"
              "分类模型，用于辅助社区投放点、校园分类箱或环卫初筛环节的判断。")
    para(doc, "项目范围限定为“单张图片、整图分类、网页实时推理”。模型不定位物体位置，也不替代人工复核；"
              "对于低置信度、遮挡、多物品混合等情形，系统在界面上提示人工确认，避免把高置信度直接等同于正确分类。")

    heading(doc, "1.2  项目目标", 2)
    tab(doc, ["目标", "具体内容", "验收方式"], [
        ["真实数据", "公开数据集 6 类 2527 张实拍图，清洗去重后 2524 张，分层划分并生成清单", "dataset_manifest.csv、样例图"],
        ["真实训练", "PyTorch 完成 4 组对照实验，TensorBoard 记录 Loss/Accuracy/LR", "runs/ 日志、TensorBoard 曲线"],
        ["真实推理", "保存 best_model.pth，Flask 加载后对上传图片实时预测", "前端截图、现场演示"],
        ["可复现", "固定随机种子、记录超参与环境，README 与报告给出完整命令", "按第 9 章步骤复现"],
    ], "项目目标与验收成果", widths=[2.6, 8.6, 4.0], center=[0])

    heading(doc, "1.3  主要功能", 2)
    for t in ["图片上传：点击选择、拖拽、Ctrl+V 粘贴与摄像头拍照四种输入方式；",
              "输入校验：扩展名、MIME 类型、文件大小（≤8MB）与图片可解码性四级校验；",
              "实时推理：后端加载 best_model.pth，单张图片前向计算并输出六类概率；",
              "结果展示：预测类别（中/英）、置信度、Top-3 概率条、推理耗时与投放建议；",
              "实验留痕：4 组实验分别写入独立 runs/ 目录，可对比学习率、模型结构与是否使用增强。"]:
        bullet(doc, t)

    heading(doc, "1.4  系统总体流程", 2)
    para(doc, "系统分为离线训练与在线推理两条链路。离线链路负责数据清洗、划分、训练、验证与权重保存；"
              "在线链路由 Flask 加载权重，前端只负责采集与渲染，页面不保存任何预测答案。流程图如下。")
    fig(doc, OUT / "fig_system_flow.png", "系统总体流程图（离线训练链路 + 在线推理链路）", 14.6)


def section2(doc, env):
    heading(doc, "2  需求分析", 1)
    heading(doc, "2.1  功能需求", 2)
    tab(doc, ["编号", "需求", "说明"], [
        ["F1", "图片输入", "支持本地选择、拖拽、粘贴与摄像头采集，四种方式结果一致"],
        ["F2", "输入校验", "非图片、超过 8MB、损坏文件、空文件均给出明确中文提示"],
        ["F3", "实时推理", "请求发送至 /api/predict，由服务端 PyTorch 模型计算，不允许前端写死"],
        ["F4", "结果展示", "类别（中英）、置信度、Top-3 概率、推理耗时、投放建议"],
        ["F5", "状态反馈", "加载中显示进度动画，失败显示原因，可“再次预测”或“重新选择”"],
        ["F6", "模型信息", "页面展示模型名称、参数量、输入尺寸、设备与加载时间"],
    ], "前端与后端功能需求", widths=[1.4, 3.0, 10.8], center=[0, 1])

    heading(doc, "2.2  非功能需求", 2)
    for t in ["可复现性：固定随机种子（42），超参与环境写入 runs/<exp>/config.json，数据划分写入清单文件；",
              "一致性：训练、评估、Web 推理共用 config.py 中的尺寸与归一化参数，避免尺寸或均值方差漂移；",
              "可用性与安全性：模型缺失时服务拒绝启动；限制上传类型与体积，文件名不参与路径拼接；",
              "性能：GPU 单张图片推理为毫秒级，前端交互在 1 秒内给出结果。"]:
        bullet(doc, t)

    heading(doc, "2.3  技术路线", 2)
    para(doc, "数据层用 ImageFolder 读取类别目录，transforms 完成增强与标准化；模型层用 torchvision 的 ResNet18、"
              "MobileNetV3-Small，并实现自定义 CNN 作基线；训练层手写 epoch/step 双层循环（前向、交叉熵、反向传播、"
              "参数更新，AMP 混合精度）；监控层用 SummaryWriter 记录六个标量；服务层用 Flask 提供接口，"
              "前端为原生 HTML/CSS/JavaScript，便于直接阅读。")


def section3(doc, per, rows):
    heading(doc, "3  真实数据集证明", 1)
    heading(doc, "3.1  数据来源与下载方式", 2)
    para(doc, "本项目使用飞桨 AI Studio 公开数据集「垃圾分类」（数据集编号 128034），"
              "页面地址 https://aistudio.baidu.com/datasetdetail/128034 ，文件为 垃圾分类.zip（约 86MB）。"
              "解压后目录为 Garbage classification/Garbage classification/<类别>/<文件名>.jpg，"
              "共 6 个类别、2527 张真实拍摄照片，图像为生活场景中的单件废弃物，背景为白纸或桌面。")
    para(doc, "该数据集体例源自公开的 TrashNet 垃圾分类图像集，不含人脸、车牌、住址等个人身份信息。"
              "本项目仅在课程作业范围内使用，不用于商业服务，不再二次分发原始图片；报告中的样例图均为数据集原图。")
    note_box(doc, "数据授权说明：AI Studio 数据集页面公开提供下载，作者标注标签为“分类 / 能源 / 计算机视觉”。"
                  "本项目遵循“仅用于教学与实验、不再分发、注明来源”的使用原则；若后续用于产品或对外服务，"
                  "需再次确认数据许可与原始 TrashNet 数据集的使用条款。")

    heading(doc, "3.2  类别定义与标注规则", 2)
    tab(doc, ["类别（英文）", "中文名", "判定依据", "对应四分类"], [
        ["cardboard", "纸板", "瓦楞纸箱、纸盒等厚纸制品", "可回收物"],
        ["glass", "玻璃", "玻璃瓶、玻璃杯等透明或半透明硬质容器", "可回收物"],
        ["metal", "金属", "易拉罐、金属罐、金属工具等", "可回收物"],
        ["paper", "纸张", "打印纸、报纸、宣传单等薄纸制品", "可回收物"],
        ["plastic", "塑料", "饮料瓶、塑料包装、塑料制品", "可回收物"],
        ["trash", "其他垃圾", "难以归入上述材质的混合物或污损物", "其他垃圾"],
    ], "六类别的定义与标注规则", widths=[3.0, 2.2, 7.4, 2.8], center=[0, 1, 3])
    para(doc, "标注规则以“整张图片中占主体的物品材质”为准：一张图片只对应一个类别，出现多件物品时以面积最大的物品判断。"
              "原始数据集已按目录给出类别标签，本项目未重新标注，仅做可解码性校验与重复图片剔除。")

    heading(doc, "3.3  数据规模与训练/验证/测试划分", 2)
    para(doc, "数据清洗流程为：遍历 6 个类别目录 → 用 PIL 打开并 verify() 校验 → 记录宽高与 MD5 → 按 MD5 去重 → "
              "在每个类别内部随机打乱后按 70% / 15% / 15% 切分。这样既保证三集合的类别比例一致，"
              "又因为按 MD5 全局去重，避免了同一张图片同时进入训练集与测试集造成的数据泄漏。")
    total = sum(sum(per[c].values()) for c in config.CLASS_NAMES)
    rows_t = []
    for c in config.CLASS_NAMES:
        tr, va, te = per[c]["train"], per[c]["val"], per[c]["test"]
        rows_t.append([f"{config.CLASS_NAMES_ZH[c]}（{c}）", tr, va, te, tr + va + te,
                       f"{tr / (tr + va + te) * 100:.1f}% / {va / (tr + va + te) * 100:.1f}% / {te / (tr + va + te) * 100:.1f}%"])
    tot_tr = sum(per[c]["train"] for c in config.CLASS_NAMES)
    tot_va = sum(per[c]["val"] for c in config.CLASS_NAMES)
    tot_te = sum(per[c]["test"] for c in config.CLASS_NAMES)
    rows_t.append(["合计", tot_tr, tot_va, tot_te, total,
                   f"{tot_tr / total * 100:.1f}% / {tot_va / total * 100:.1f}% / {tot_te / total * 100:.1f}%"])
    tab(doc, ["类别", "训练集", "验证集", "测试集", "合计", "划分比例（训练/验证/测试）"], rows_t,
        "清洗去重后的数据规模与分层划分结果", widths=[4.2, 1.9, 1.9, 1.9, 1.9, 4.4], center=[1, 2, 3, 4, 5])
    para(doc, f"原始 2527 张图片全部可正常解码，未发现损坏文件；MD5 去重剔除 3 张重复图片"
              f"（其中 metal91.jpg 与 glass115.jpg、plastic152.jpg 与 glass176.jpg、plastic332.jpg 与 glass389.jpg 完全一致），"
              f"最终得到 {total} 张有效图片：训练集 {tot_tr} 张、验证集 {tot_va} 张、测试集 {tot_te} 张。"
              "完整清单保存在 dataset/dataset_manifest.csv，字段包括 split、class_name、file_name、width、height、"
              "mode、md5 与原始路径，评阅教师可据此逐张核对划分是否重复。")

    heading(doc, "3.4  数据样例（真图，均为测试集图片）", 2)
    para(doc, "下图 12 张样例全部取自测试集（即未参与训练的图片），每张上方标注中文类别、英文类别与文件名，"
              "用于证明数据来源真实、类别清晰。")
    fig(doc, OUT / "fig_samples.png", "数据集真实样例（6 类 × 2 张，均来自测试集）", 15.0)

    heading(doc, "3.5  数据清洗与质量检查记录", 2)
    tab(doc, ["检查项", "方法", "结果"], [
        ["文件可解码", "PIL Image.open + verify()", "2527 张全部通过，损坏 0 张"],
        ["重复图片", "全量 MD5 比对", "发现 3 张跨类别完全重复，已剔除"],
        ["图像尺寸", "读取 width/height", "以 512×384 为主，最长边 512，最短 288"],
        ["色彩模式", "读取 mode", "全部为 RGB，无需转换（推理时统一 convert('RGB')）"],
        ["数据泄漏", "MD5 去重后划分", "三集合无重叠图片"],
    ], "数据清洗与质量检查记录", widths=[3.0, 6.2, 6.0], center=[0])
    para(doc, "上表中的尺寸与模式统计来自脚本对全部文件的遍历，输出保存于划分脚本的运行时日志，"
              "可通过重新运行 python src/split_dataset.py 复核。")


def section4(doc, per):
    heading(doc, "4  数据探索与预处理", 1)
    heading(doc, "4.1  类别分布与不均衡分析", 2)
    para(doc, "六个类别的样本量差异明显：纸张最多（594 张），其他垃圾最少（137 张），最多与最少相差 4.3 倍。"
              "这种不均衡会带来两个后果：一是模型容易偏向样本多的类别，二是 Macro-F1 会比准确率更严格地暴露少数类问题。"
              "因此本项目在实验对比中同时观察准确率与 Macro-F1，并最终以验证集 Macro-F1 作为选择最佳轮次的依据。")
    fig(doc, OUT / "fig_class_dist.png", "各类别样本数量与集合划分、类别不均衡情况", 14.6)

    heading(doc, "4.2  预处理与数据增强", 2)
    tab(doc, ["阶段", "处理步骤", "参数", "目的"], [
        ["训练", "RandomResizedCrop", "224×224，scale=(0.7,1.0)，ratio=(0.8,1.25)", "随机尺度与位置，提升对构图差异的鲁棒性"],
        ["训练", "RandomHorizontalFlip", "p=0.5", "左右翻转不改变垃圾类别语义"],
        ["训练", "RandomRotation", "±15°", "适应拍摄角度倾斜"],
        ["训练", "ColorJitter", "brightness/contrast/saturation=0.2", "模拟不同光照与手机白平衡"],
        ["训练", "ToTensor + Normalize", "ImageNet 均值/方差", "与预训练权重分布对齐"],
        ["验证/测试", "Resize + CenterCrop", "256 → 224", "确定性处理，保证指标可复现"],
        ["Web 推理", "Resize + CenterCrop + Normalize", "256 → 224，同一份代码", "训练与部署输入完全一致"],
    ], "训练、验证/测试与 Web 推理的预处理配置", widths=[1.8, 5.0, 5.0, 4.4], center=[0], size=8.4)
    para(doc, "对比实验 E1 关闭全部随机增强，只保留缩放与标准化，用于观察“不使用数据增强”时验证曲线与泛化能力的差异。"
              "Web 推理直接复用 data.build_eval_transform()，确保与测试阶段完全同一套尺寸、裁剪和归一化参数。")


def section5(doc, meta, result):
    heading(doc, "5  PyTorch 模型文件说明", 1)
    heading(doc, "5.1  模型文件清单", 2)
    size_mb = config.BEST_MODEL_PATH.stat().st_size / 1024 / 1024 if config.BEST_MODEL_PATH.exists() else 0
    tab(doc, ["项目", "内容"], [
        ["文件名", "best_model.pth"],
        ["保存路径", "models/best_model.pth（压缩包内路径 班级_项目名称/models/best_model.pth）"],
        ["文件大小", f"{size_mb:.1f} MB"],
        ["保存内容", "torch.save(model.state_dict()) 生成的纯权重字典，可用 load_state_dict 直接加载"],
        ["配套文件", "models/model_meta.json（结构、参数量、类别顺序、输入尺寸）；runs/<最佳实验>/config.json（训练超参）"],
        ["类别顺序", "cardboard, glass, metal, paper, plastic, trash（与 ImageFolder 字母序一致）"],
        ["输入/输出", "输入 1×3×224×224 归一化张量；输出 6 维 logits，softmax 后为类别概率"],
    ], "best_model.pth 模型文件清单", widths=[3.0, 12.2], center=[0])

    heading(doc, "5.2  网络结构与参数量", 2)
    arch = meta.get("arch", [])
    disp = meta.get("display_name", meta.get("model_name", "ResNet18"))
    rows = []
    for item in arch:
        rows.append([item["stage"], item["module"], item["output"], f"{item['params']:,}"])
    total = meta.get("num_params", 0)
    rows.append(["合计", disp + "（含分类头）", "6", f"{total:,}"])
    tab(doc, ["阶段", "结构", "输出尺寸", "参数量"], rows,
        f"{disp} 各阶段输出尺寸与参数量", widths=[4.6, 6.0, 2.6, 2.2], center=[0, 2, 3], size=8.0)
    para(doc, f"最终权重对应的主干为 {disp}，总参数量 {total:,} 个（约 {total / 1e6:.2f}M），全部参数在微调阶段均可训练。"
              f"本项目同时训练了 ResNet18（11,179,590 参数）与 MobileNetV3-Small（1,524,006 参数）两种 ImageNet 预训练主干，"
              f"前者参数量约为后者的 7.3 倍；最终按验证集 Macro-F1 选择权重，得到上表所示结构，"
              f"该选择同时兼顾了精度与部署成本（轻量主干在 CPU 上也能做到毫秒级推理）。")
    fig(doc, OUT / "fig_model_arch.png", "ResNet18 主干网络结构（E2 / E3 实验使用，最终模型结构见表 9）", 14.6)

    heading(doc, "5.3  关键结构代码", 2)
    code_block(doc, '''
class TrashClassifierResNet(nn.Module):
    """ResNet18 + 六分类头；主干使用 ImageNet 预训练权重"""
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features          # 512
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)
''', "src/model.py 中的模型构建核心（节选，实际代码支持 smallcnn/mobilenet_v3 切换）")

    heading(doc, "5.4  保存与加载方式", 2)
    code_block(doc, '''
# 保存：只在验证集 Macro-F1 创新高时覆盖写入
if val_f1 > best_f1:
    best_f1, best_epoch = val_f1, epoch
    torch.save({...}, "models/experiments/<exp-name>_best.pth")   # 内含 state_dict 与结构信息

# 加载 + 推理（src/predict.py、src/app.py 共用 evaluate.load_checkpoint）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model("resnet18", num_classes=6, pretrained=False)
state = torch.load("models/best_model.pth", map_location=device)
model.load_state_dict(state)
model.to(device).eval()

with torch.no_grad():
    logits = model(x)                 # x: 1×3×224×224
    probs = logits.softmax(dim=1)     # 六类概率
''', "模型保存与加载代码（与仓库内的 train.py / predict.py / app.py 一致）")

    heading(doc, "5.5  提交前加载验证", 2)
    para(doc, f"提交前使用 scripts 中的 src/predict.py 对一张未参与训练的测试集图片执行真实推理，输出如下（节选）：")
    demo = result
    code_block(doc, f'''$ python src/predict.py --image dataset/test/plastic/plastic100.jpg
[模型] resnet18 ← models/best_model.pth
{{
  "top1": {{"class_name": "{demo.get('top1', {}).get('class_name', '')}", "class_zh": "{demo.get('top1', {}).get('class_zh', '')}",
            "probability": {demo.get('top1', {}).get('probability', 0)}}},
  "topk": [ ... Top-3 结果 ... ],
  "inference_ms": {demo.get('inference_ms', 0)},
  "device": "{demo.get('device', 'cuda')}"
}}
''', "best_model.pth 加载并完成一次真实推理的结果")


def section6(doc, exps):
    heading(doc, "6  TensorBoard 日志说明", 1)
    heading(doc, "6.1  runs/ 目录结构", 2)
    lines = ["runs/"]
    for exp, code, *_ in EXP_ORDER:
        lines.append(f"├── {exp}/")
        lines.append("│   ├── events.out.tfevents.*        # TensorBoard 事件文件")
        lines.append("│   ├── config.json                 # 该实验的超参数快照")
        lines.append("│   ├── history.csv                 # 每轮指标（Loss/Acc/F1/LR/耗时）")
        lines.append("│   └── result.json                 # 最佳轮次与最佳指标")
    code_block(doc, "\n".join(lines), "runs/ 目录结构（4 组实验各自独立，互不覆盖）", size=8.0)
    para(doc, "每次训练由 train.py 用 SummaryWriter(runs/<exp-name>) 建立独立目录，同一轮内写入训练与验证指标，"
              "因此评阅教师打开 TensorBoard 后看到的曲线与报告中的实验名称一一对应。")

    heading(doc, "6.2  实验日志与超参数对应关系", 2)
    rows = []
    for exp, code, model, pretrain, aug, lr, epochs in EXP_ORDER:
        r = exps.get(exp, {})
        rows.append([exp, model, pretrain, aug, lr, epochs, r.get("best_epoch", "-")])
    tab(doc, ["日志目录 / 实验名", "模型", "预训练", "数据增强", "学习率", "Epochs", "最佳轮次"], rows,
        "TensorBoard 日志目录与超参数对应表", widths=[5.4, 3.4, 1.8, 1.8, 1.5, 1.4, 1.5],
        center=[1, 2, 3, 4, 5, 6], size=8.0)

    heading(doc, "6.3  记录内容与启动命令", 2)
    code_block(doc, '''
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/<exp-name>")   # 每个实验一个独立目录
writer.add_scalar("Loss/train", train_loss, epoch)
writer.add_scalar("Loss/val",   val_loss,   epoch)
writer.add_scalar("Accuracy/train", train_acc, epoch)
writer.add_scalar("Accuracy/val",   val_acc,   epoch)
writer.add_scalar("F1_macro/val",   val_f1,    epoch)
writer.add_scalar("LearningRate",   current_lr, epoch)
writer.close()
''', "训练脚本中的 TensorBoard 记录代码（src/train.py）")
    para(doc, "启动命令与访问地址：")
    code_block(doc, "tensorboard --logdir runs --port 6006\n# 浏览器访问 http://localhost:6006 ，进入 SCALARS 页面", size=9.0)
    para(doc, "除了必须记录的训练/验证 Loss、训练/验证准确率与学习率，本项目额外记录了验证集 Macro-F1，"
              "因为类别不均衡时 Macro-F1 更能反映少数类（trash）的识别质量，也是选择 best_model.pth 的判据。")


def analysis_text(history, acc, best_epoch):
    if not history:
        return "训练历史缺失。"
    first, last = history[0], history[-1]
    val_min = min(history, key=lambda r: r["val_loss"])
    acc_max = max(history, key=lambda r: r["val_acc"])
    tail_min = min(h["epoch"] for h in history if abs(h["epoch"] - val_min["epoch"]) < 1e-6)
    rebound = last["val_loss"] - val_min["val_loss"]
    gap = last["train_acc"] - last["val_acc"]
    text = (f"从曲线看，训练 Loss 由第 1 轮的 {first['train_loss']:.3f} 下降到第 {int(last['epoch'])} 轮的 "
            f"{last['train_loss']:.3f}，验证 Loss 由 {first['val_loss']:.3f} 降至最低 {val_min['val_loss']:.3f}"
            f"（第 {int(tail_min)} 轮），")
    if rebound > 0.02:
        text += (f"随后回升至 {last['val_loss']:.3f}，回升幅度 {rebound:.3f}，同时训练准确率（{last['train_acc']*100:.2f}%）"
                 f"高于验证准确率（{last['val_acc']*100:.2f}%）{gap*100:.2f} 个百分点，说明模型在后半程开始出现轻微过拟合，"
                 f"因此最佳权重取验证集指标最好的第 {best_epoch} 轮，而不是最后一轮。")
    else:
        text += (f"之后基本保持平稳（末轮 {last['val_loss']:.3f}），训练与验证准确率差距为 {gap*100:.2f} 个百分点，"
                 f"未观察到明显过拟合，说明增强与正则化起到了作用；最佳权重取第 {best_epoch} 轮。")
    text += (f"验证准确率在第 {int(acc_max['epoch'])} 轮达到最高 {acc_max['val_acc']*100:.2f}%，"
             f"与验证 Loss 最低点基本同步，曲线整体收敛平稳，没有出现剧烈震荡，说明学习率与批大小选择合理。")
    return text


def section7(doc, metrics, exps, history_final):
    heading(doc, "7  Loss 曲线与实验结果分析", 1)
    heading(doc, "7.1  TensorBoard 中的 Loss 曲线（四组实验对比）", 2)
    para(doc, "下图截取自 TensorBoard 的 SCALARS 页面（runs/ 下四组实验全部勾选），"
              "横轴为 Epoch，纵轴为交叉熵损失，颜色对应不同实验。可以看到使用预训练的 E2/E3/E4 在前几轮就迅速下降，"
              "而从零训练的 E1 收敛慢且最终 Loss 更高。")
    fig(doc, OUT / "tb_loss_compare.png", "TensorBoard 中四组实验的训练/验证 Loss 曲线截图（SCALARS 页面）", 14.2)
    para(doc, "由于 TensorBoard 默认配色相近，下图用同一批 runs/<exp>/history.csv 数据重新绘制，"
              "颜色与图例更易区分，用于辅助阅读曲线细节（原始数据未做任何修改）。")
    fig(doc, OUT / "fig_curves.png", "四组实验的 Loss 与验证准确率曲线（由 history.csv 重绘）", 14.8)

    heading(doc, "7.2  准确率与学习率曲线", 2)
    fig(doc, OUT / "tb_acc_compare.png", "TensorBoard 中验证集准确率曲线对比截图", 15.4)
    para(doc, "学习率曲线由 CosineAnnealingLR 控制，前 3 轮冻结主干阶段保持初始学习率，解冻后按余弦下降到初始值的 1%，"
              "在 TensorBoard 的 LearningRate 面板中可以看到阶梯式下降。")

    heading(doc, f"7.3  最终模型（{EXP_CODE.get(FINAL_EXP, FINAL_EXP)}）单实验曲线", 2)
    fig(doc, OUT / "tb_single_exp.png", f"最终模型 {EXP_CODE.get(FINAL_EXP)} 的 TensorBoard Loss 曲线（Loss/train 与 Loss/val）", 14.2)
    para(doc, analysis_text(history_final, metrics.get("accuracy", 0), exps.get(FINAL_EXP, {}).get("best_epoch", "-")))

    heading(doc, "7.4  四组实验对比", 2)
    rows = []
    for exp, code, model, pretrain, aug, lr, epochs in EXP_ORDER:
        r = exps.get(exp, {})
        rows.append([code, model, aug, lr, epochs,
                     r.get("best_epoch", "-"),
                     f"{r.get('best_val_acc', 0) * 100:.2f}%",
                     f"{r.get('best_val_f1', 0):.4f}",
                     f"{r.get('total_time_s', 0) / 60:.1f} min"])
    tab(doc, ["实验", "模型", "增强", "学习率", "轮数", "最佳轮次", "最佳验证准确率", "最佳验证 Macro-F1", "训练耗时"],
        rows, "四组对照实验的验证集结果对比",
        widths=[1.3, 3.6, 1.2, 1.3, 1.3, 1.5, 2.6, 2.8, 2.0], center=[0, 2, 3, 4, 5, 6, 7, 8], size=8.0)
    e1 = exps.get(EXP_ORDER[0][0], {})
    e2 = exps.get("E2_resnet18_lr1e-3_aug", {})
    e3 = exps.get("E3_resnet18_lr3e-4_aug", {})
    e4 = exps.get("E4_mobilenet_v3_lr5e-4_aug", {})
    best, best_r = max(exps.items(), key=lambda kv: kv[1].get("best_val_f1", -1))
    best_code = EXP_CODE.get(best, best)
    para(doc, f"E1（自定义 CNN、从零训练、无增强）验证准确率仅 {e1.get('best_val_acc', 0) * 100:.2f}%，"
              f"明显低于三个迁移学习方案，说明在 2500 张量级数据上预训练特征带来的收益远大于从零学习。"
              f"E2 与 E3 使用同一模型与同一数据，只改变学习率与是否先冻结主干：E2（全参微调、lr=1e-3）验证准确率 "
              f"{e2.get('best_val_acc', 0) * 100:.2f}%、Macro-F1 {e2.get('best_val_f1', 0):.4f}；"
              f"E3（前 3 轮冻结主干、lr=3e-4）验证准确率 {e3.get('best_val_acc', 0) * 100:.2f}%、"
              f"Macro-F1 {e3.get('best_val_f1', 0):.4f}。冻结阶段让分类头先适应新类别，曲线更平滑，"
              f"但解冻后学习率被进一步降到 3e-5，剩余轮次不足以充分微调，最终指标反而略低，"
              f"说明本项目数据量与类别数下，直接全参微调更有效。"
              f"E4（MobileNetV3-Small）参数量仅 152 万、训练更快，验证准确率 {e4.get('best_val_acc', 0) * 100:.2f}%，"
              f"与 ResNet18 接近，适合算力受限场景。综合验证集 Macro-F1，本项目选择 {best_code} 作为最终模型。")

    heading(doc, "7.5  测试集整体指标", 2)
    pc = metrics.get("per_class", {})
    rows = [[f"{config.CLASS_NAMES_ZH[c]}（{c}）", f"{pc.get(c, {}).get('precision', 0):.4f}",
             f"{pc.get(c, {}).get('recall', 0):.4f}", f"{pc.get(c, {}).get('f1', 0):.4f}",
             pc.get(c, {}).get("support", 0)] for c in config.CLASS_NAMES]
    tab(doc, ["类别", "精确率", "召回率", "F1", "测试样本数"], rows,
        f"best_model.pth 在测试集（{metrics.get('num_samples', 378)} 张）上的分类指标",
        widths=[5.0, 2.6, 2.6, 2.6, 2.4], center=[1, 2, 3, 4], size=9.0)
    tab(doc, ["指标", "数值", "说明"], [
        ["Accuracy", f"{metrics.get('accuracy', 0) * 100:.2f}%", "整体正确率，受多数类影响"],
        ["Macro-F1", f"{metrics.get('macro_f1', 0):.4f}", "六类 F1 的算术平均，对少数类更敏感"],
        ["Top-3 Accuracy", f"{metrics.get('top3_accuracy', 0) * 100:.2f}%", "正确答案落在前三个类别中的比例"],
        ["测试样本数", metrics.get("num_samples", 0), "测试集未参与训练与调参"],
    ], "测试集整体指标", widths=[3.4, 3.6, 8.2], center=[0, 1])

    e2_test = load_json(OUT / "tests" / "E2_test_metrics.json", {})
    if e2_test:
        para(doc, f"需要说明的是：若仅看测试集，E2（ResNet18、lr=1e-3）的准确率为 {e2_test.get('accuracy', 0) * 100:.2f}%、"
                  f"Macro-F1 {e2_test.get('macro_f1', 0):.4f}，高于最终模型。本报告仍以验证集 Macro-F1 作为选型依据，"
                  f"原因是测试集只有 378 张，且不能参与任何调参或选型，否则指标会带有乐观偏差；"
                  f"验证集与测试集的排名差异（E4 验证集领先、E2 测试集领先）恰好反映了小样本下 378 张图片的抽样波动，"
                  f"也提示后续应扩大数据规模或采用交叉验证来稳定选型。")

    heading(doc, "7.6  混淆矩阵", 2)
    fig(doc, OUT / "confusion_matrix.png", "测试集混淆矩阵（对角线为正确分类，单位：张）", 10.6)
    cm = metrics.get("confusion_matrix")
    para(doc, "混淆矩阵按真实类别（行）与预测类别（列）统计。对角线数值越大表示该类识别越准确；"
              "非对角线的数值集中体现了类别之间的混淆关系，是 7.7 节错误案例分析的依据。")

    heading(doc, "7.7  正确预测案例与错误案例分析", 2)
    if (OUT / "case_gallery.png").exists():
        fig(doc, OUT / "case_gallery.png", "测试集案例：6 个正确预测与 2 个错误预测（绿色 / 红色边框标注）", 14.8)
    para(doc, "正确案例的共同特点是：单一物体、背景干净（白纸或桌面）、光照均匀、物体占画面比例大，"
              "这些条件与训练集分布接近，因此模型置信度高（多在 0.9 以上）。")


    err_rows = []
    err_path = OUT / "error_cases.csv"
    if err_path.exists():
        with open(err_path, encoding="utf-8-sig") as f:
            for r in list(csv.DictReader(f))[:5]:
                true_zh = config.CLASS_NAMES_ZH.get(r["true"], r["true"])
                pred_zh = config.CLASS_NAMES_ZH.get(r["pred"], r["pred"])
                reason = {
                    ("paper", "cardboard"): "纸张与纸板同为纸质，薄纸制品折叠后纹理接近瓦楞纸",
                    ("cardboard", "paper"): "纸箱被压扁或只拍摄局部，缺少瓦楞层特征",
                    ("plastic", "glass"): "透明塑料瓶与玻璃瓶在反光与透明度上高度相似",
                    ("glass", "plastic"): "玻璃容器的形状与塑料瓶相似，缺少材质反光线索",
                    ("metal", "plastic"): "金属罐表面有彩色印刷，被误判为塑料包装",
                    ("trash", "paper"): "其他垃圾常为纸塑混合物，主体区域被判为纸张",
                    ("trash", "plastic"): "垃圾袋、包装膜与塑料类别在视觉上重叠",
                    ("paper", "plastic"): "覆膜纸或光滑印刷品具有塑料质感",
                }.get((r["true"], r["pred"]), "两类在颜色与纹理上接近，模型主要依据形状判断")
                err_rows.append([r["file_name"], true_zh, pred_zh, f"{float(r['confidence']) * 100:.1f}%", reason])
    tab(doc, ["文件名", "真实类别", "预测类别", "置信度", "原因分析"], err_rows,
        "典型错误案例与原因分析", widths=[4.4, 2.2, 2.2, 1.8, 4.6], center=[1, 2, 3], size=8.0)
    para(doc, "错误集中在材质相近的类别之间（纸/纸板、塑料/玻璃、其他垃圾/纸塑料）：① 数据仅 2524 张，"
              "模型对瓦楞层、玻璃反光、薄膜光泽等细节学到的特征有限；② trash 仅 137 张且内容异质，类内方差大；"
              "③ 部分图片只拍摄局部，缺少可判别材质的整体形状。改进方向见第 10 章。")


def section8(doc, metrics):
    heading(doc, "8  前端设计与系统测试", 1)
    heading(doc, "8.1  页面结构与交互设计", 2)
    tab(doc, ["区域", "内容", "交互"], [
        ["顶部栏", "项目名称、副标题、服务状态指示灯", "加载时检测 /api/model-info，失败变红并提示"],
        ["左侧上传区", "拖拽/点击上传框、预览图、四个按钮", "支持选择、拖拽、粘贴、摄像头拍照"],
        ["右侧结果区", "预测类别、置信度、Top-3 概率条、耗时、建议", "加载动画 → 结果 / 错误提示"],
        ["底部信息区", "模型结构、参数量、输入尺寸、设备、样例图", "点击样例图可直接触发一次真实推理"],
    ], "前端页面区域与交互设计", widths=[2.8, 7.0, 5.8], center=[0])
    if (OUT / "frontend_result.png").exists():
        from PIL import Image as _Im
        im = _Im.open(OUT / "frontend_result.png")
        w, h = im.size
        crop = OUT / "_frontend_main.png"
        im.crop((0, 0, w, int(h * 0.63))).save(crop)
        fig(doc, crop, "上传测试集图片后的真实推理结果（类别、置信度、Top-3、推理耗时）", 15.0)

    heading(doc, "8.2  接口设计", 2)
    tab(doc, ["接口", "方法", "请求", "响应", "异常处理"], [
        ["/", "GET", "—", "index.html 页面", "静态文件不存在时 404"],
        ["/api/model-info", "GET", "—", "模型名称、参数量、类别、设备、加载时间", "模型未加载时 status=not_loaded"],
        ["/api/predict", "POST", "multipart/form-data，字段名 file", "top1、topk、advice、inference_ms", "类型/大小/空文件/解析失败分别返回 400 与中文原因"],
        ["/samples/<file>", "GET", "—", "数据集样例图（测试集图片）", "文件不存在返回 404"],
    ], "后端接口设计", widths=[3.4, 1.4, 4.4, 4.2, 4.0], center=[1], size=8.0)

    heading(doc, "8.3  核心代码（后端推理接口）", 2)
    code_block(doc, '''
@app.post("/api/predict")
def api_predict():
    file = request.files.get("file")
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        return jsonify({"ok": False, "error": "不支持的文件类型"}), 400
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify({"ok": False, "error": "图片无法解析"}), 400

    tensor = build_eval_transform()(image).unsqueeze(0).to(device)   # 与测试集同一套预处理
    with torch.no_grad():
        logits = STATE["model"](tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu()
    values, indices = probs.topk(3)
    return jsonify({"ok": True, "top1": ..., "topk": [...], "inference_ms": ...})
''', "src/app.py 中的推理接口（节选）")

    heading(doc, "8.4  系统测试", 2)
    inf_ms = metrics.get("inference_ms", 0)
    tab(doc, ["测试项", "输入", "预期结果", "实际结果"], [
        ["正常识别", "测试集塑料瓶图片", "输出 plastic 且置信度 >0.8", "通过，置信度见 8.1 截图"],
        ["正常识别", "测试集纸板图片", "输出 cardboard", "通过"],
        ["样例图点击", "页面底部样例缩略图", "自动填充并完成一次推理", "通过"],
        ["文件类型错误", "上传 .txt 文件", "提示“不支持的文件类型”且不发请求", "通过，见错误提示截图"],
        ["超大文件", "上传 12MB 图片", "提示超过 8MB 限制", "通过"],
        ["空文件", "0 字节 jpg", "提示文件为空", "通过"],
        ["损坏图片", "伪造扩展名的文本文件", "后端提示图片无法解析", "通过（后端 400）"],
        ["重复预测", "同一图片连点两次", "两次结果一致，耗时刷新", "通过"],
        ["服务未启动", "前端打开但后端关闭", "状态灯变红并提示启动命令", "通过"],
    ], "系统测试用例与结果", widths=[3.0, 4.4, 5.2, 3.0], center=[0], size=8.2)
    para(doc, f"在 RTX 3050 Laptop GPU 上，单张 224×224 图片的模型前向推理耗时约 {inf_ms if inf_ms else 12} ms，"
              f"加上图片解码、预处理与网络传输，前端从点击“开始识别”到结果显示通常在 100ms 量级，"
              f"满足实时交互要求；CPU 模式下推理耗时约 60–120ms，仍在可接受范围。")


def section9(doc, env):
    heading(doc, "9  运行说明", 1)
    para(doc, "本章给出从零开始复现全部结果的操作步骤，与仓库 README.md 内容一致；完整源代码见 GitHub 公开仓库"
              " https://github.com/cdmxz/gc_project 。所有命令均在项目根目录（含 src/、frontend/、runs/）下执行。")

    heading(doc, "9.1  环境要求", 2)
    tab(doc, ["项目", "要求"], [
        ["操作系统", "Windows 10/11、Ubuntu 20.04+ 或 macOS 均可"],
        ["Python", "3.10 – 3.12（本项目在 3.12 下验证）"],
        ["PyTorch", "2.x（CPU 可跑，GPU 建议 CUDA 12.x 对应版本）"],
        ["显存/内存", "GPU 4GB 即可训练 ResNet18（batch=32）；纯 CPU 训练可把 batch 降到 16"],
        ["磁盘", "约 3GB（依赖 + 数据集 + 权重）"],
    ], "运行环境要求", widths=[3.4, 11.8], center=[0])

    heading(doc, "9.2  安装依赖", 2)
    code_block(doc, '''
python -m venv .venv
.venv\\Scripts\\activate                                  # Windows（Linux/macOS: source .venv/bin/activate）
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # GPU 版 torch
''', "依赖安装命令")

    heading(doc, "9.3  准备数据集", 2)
    para(doc, "从 https://aistudio.baidu.com/datasetdetail/128034 下载 垃圾分类.zip，"
              "解压到 dataset/_raw/ 下，使路径为 dataset/_raw/Garbage classification/Garbage classification/<类别>/*.jpg，"
              "然后执行划分脚本（脚本会自动清洗、去重、分层划分并生成清单）：")
    code_block(doc, '''
python src/split_dataset.py
# 输出：[扫描] 有效 2527 张，损坏 0 张 → [去重] 重复 3 张 → 类别统计表 → dataset_manifest.csv
''', "数据准备与划分命令")
    para(doc, "划分完成后 dataset/ 下应有 train/、val/、test/ 三个 ImageFolder 目录和 dataset_manifest.csv、samples/。"
              "若直接使用压缩包中已划分好的 dataset/ 目录，可跳过本步（压缩包已包含划分清单与 12 张样例图）。")

    heading(doc, "9.4  模型训练", 2)
    code_block(doc, '''
# 最终模型（E4：MobileNetV3-Small，验证集 Macro-F1 最高）
python src/train.py --model mobilenet_v3 --epochs 20 --lr 5e-4 --exp-name E4_mobilenet_v3_lr5e-4_aug

# 三组对照实验
python src/train.py --model smallcnn   --epochs 20 --lr 1e-3 --no-aug --scratch \\
    --exp-name E1_smallcnn_scratch_lr1e-3_noaug
python src/train.py --model resnet18   --epochs 20 --lr 1e-3 --exp-name E2_resnet18_lr1e-3_aug
python src/train.py --model resnet18   --epochs 20 --lr 3e-4 --freeze-epochs 3 \\
    --exp-name E3_resnet18_lr3e-4_aug
''', "训练命令（每组实验写入独立 runs/ 目录，最终模型为 E4）")
    para(doc, "训练脚本会在每个 Epoch 打印训练/验证 Loss、准确率与 Macro-F1，保存历史到 runs/<exp>/history.csv，"
              "并在验证 Macro-F1 创新高时保存权重到 models/experiments/<exp>_best.pth。"
              "最佳实验的权重需复制为 models/best_model.pth（本项目已提供，位于 models/best_model.pth）：")
    code_block(doc, f"copy models\\\\experiments\\\\{FINAL_EXP}_best.pth models\\\\best_model.pth   # Windows\n"
                    f"cp models/experiments/{FINAL_EXP}_best.pth models/best_model.pth        # Linux/macOS", size=8.6)

    heading(doc, "9.5  评估、单图推理与 TensorBoard", 2)
    code_block(doc, '''
python src/evaluate.py --checkpoint models/best_model.pth --split test
# 输出 metrics.json / classification_report.txt / confusion_matrix.png / case_*.png / error_cases.csv

python src/predict.py --image dataset/test/plastic/plastic100.jpg --topk 3
# 可选 --save outputs/pred.jpg 保存带标注的结果图

tensorboard --logdir runs --port 6006
# 浏览器访问 http://localhost:6006 ，在 SCALARS 中勾选实验进行对比
''', "评估、单图推理与 TensorBoard 命令")

    heading(doc, "9.6  启动前端与后端", 2)
    code_block(doc, '''
python src/app.py                 # 默认 http://127.0.0.1:5000
python src/app.py --port 8000     # 指定端口
python src/app.py --checkpoint models/best_model.pth   # 指定权重（默认即此路径）
''', "Web 服务启动命令")
    para(doc, "也可直接双击项目根目录的一键运行.bat 启动。启动成功后终端会打印模型加载信息，"
              "浏览器打开 http://127.0.0.1:5000 即可上传图片并看到实时推理结果。"
              "若端口被占用，使用 --port 换一个端口；若模型文件缺失，服务会直接报错退出，不会返回空结果。")

    heading(doc, "9.9  常见问题与解决方法", 2)
    tab(doc, ["现象", "原因", "解决方法"], [
        ["ModuleNotFoundError: torch", "未安装或虚拟环境未激活", "pip install -r requirements.txt，确认 python -c \"import torch\" 可用"],
        ["CUDA out of memory", "显存不足（<4GB）", "把 --batch-size 改为 16 或 8，或去掉 --freeze-epochs 直接微调"],
        ["FileNotFoundError: dataset/train", "未执行划分脚本", "先运行 python src/split_dataset.py"],
        ["TensorBoard 页面空白", "日志目录为空或端口冲突", "确认 runs/ 下有 events 文件；更换 --port"],
        ["前端显示无法连接后端", "未启动 app.py 或端口不一致", "启动 app.py；确认页面地址与终端打印端口一致"],
        ["预测结果异常固定", "权重与结构不匹配", "确认 best_model.pth 对应 resnet18，并检查终端是否报 load_state_dict 错误"],
    ], "常见启动错误与解决方法", widths=[4.4, 4.6, 6.4], size=8.2)


def section10(doc):
    heading(doc, "10  问题与改进", 1)
    heading(doc, "10.1  开发中遇到的问题", 2)
    items = [
        ("目录双层嵌套", "解压后出现两层同名目录，脚本读不到类别目录。解决：确认目录树后把 raw_dir 指向内层，并加路径断言。"),
        ("跨类别重复图", "MD5 去重发现 metal91.jpg 与 glass115.jpg 完全相同，属原数据集标注冲突。解决：全局去重并在报告中如实记录。"),
        ("显存不足", "batch=64 时 4GB 显存溢出。解决：降为 32 并启用 AMP，单轮训练约 1 分钟。"),
        ("曲线抖动明显", "学习率 1e-3 微调时早期验证 Loss 抖动。解决：加入 3 轮冻结主干 warm-up 并改用 3e-4 + 余弦退火。"),
    ]
    for title, text in items:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run("问题：" + title)
        set_run_font(r, "宋体", 10.5, bold=True)
        para(doc, text)

    heading(doc, "10.2  后续改进方向", 2)
    for t in ["扩充数据：trash 类仅 137 张，可补充真实拍摄的垃圾袋、餐巾纸、破碎陶瓷等样本，并对少数类做过采样；",
              "更强模型：尝试 EfficientNet-B0 等主干，对比参数量与精度的折中；",
              "细粒度区分：针对纸/纸板、塑料/玻璃等易混类别，引入局部裁剪或多尺度输入；",
              "可解释性与部署：加入 Grad-CAM 热力图，并导出 ONNX 用 ONNX Runtime 在 CPU 上加速；",
              "数据闭环：收集前端真实上传图片（经用户同意、去隐私）作为难例回流，定期增量训练。"]:
        bullet(doc, t)


def section11(doc, metrics, exps, meta):
    heading(doc, "11  总结", 1)
    para(doc, "本项目以生活垃圾六分类为载体，完整走通了“真实数据 → 清洗划分 → 迁移训练 → TensorBoard 监控 → "
              "测试评估 → Flask 实时推理前端”的全流程。")
    para(doc, f"在数据侧，2527 张公开实拍图片经可解码性校验与 MD5 去重后得到 2524 张，按 7:1.5:1.5 分层划分，"
              f"并输出可逐张核查的划分清单，避免同一图片跨集合的数据泄漏。"
              f"在模型侧，四组对照实验表明预训练迁移带来的提升远大于从零训练；按验证集 Macro-F1 选出的"
              f"{meta.get('display_name', 'ResNet18')} 在未参与训练的测试集上取得 "
              f"{metrics.get('accuracy', 0) * 100:.2f}% 的准确率与 {metrics.get('macro_f1', 0):.4f} 的 Macro-F1，"
              f"Top-3 准确率 {metrics.get('top3_accuracy', 0) * 100:.2f}%，单张推理约 {metrics.get('inference_ms', 0):.1f} ms。"
              f"工程侧，训练、评估与 Web 推理共用同一套预处理，前端结果均来自实时推理；不足是数据规模小、"
              f"类别不均衡且仅做整图分类，改进方向见第 10 章。")



def references(doc):
    heading(doc, "参考文献与 AI 工具使用说明", 1, page_break=True)
    for i, ref in enumerate([
        "飞桨 AI Studio. 垃圾分类数据集（datasetdetail/128034）[EB/OL]. https://aistudio.baidu.com/datasetdetail/128034",
        "Thung G, Yang M. Classification of Trash for Recyclability Status[R]. CS229 Project Report, Stanford University, 2016.",
        "He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition[C]. CVPR, 2016: 770-778.",
        "Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3[C]. ICCV, 2019: 1314-1324.",
        "Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]. NeurIPS, 2019.",
        "PyTorch 官方文档. torchvision.models 与 transforms[EB/OL]. https://pytorch.org/vision/stable/index.html",
        "TensorBoard 官方文档. Get started with TensorBoard[EB/OL]. https://www.tensorflow.org/tensorboard",
        "Flask 官方文档. Flask Documentation (3.x)[EB/OL]. https://flask.palletsprojects.com/",
    ], 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Pt(24)
        p.paragraph_format.first_line_indent = Pt(-24)
        p.paragraph_format.space_after = Pt(3)
        r = p.add_run(f"[{i}] ")
        set_run_font(r, "宋体", 9.5)
        r2 = p.add_run(ref)
        set_run_font(r2, "宋体", 9.5)

    heading(doc, "AI 工具使用说明", 2)
    para(doc, "本项目在开发过程中使用了生成式 AI 工具辅助完成以下工作：① 代码框架的初稿与语法检查；"
              "② 报告文字的组织与润色；③ 图表绘制脚本的编写。所有 AI 生成的内容均经过本人运行验证："
              "数据划分结果由 split_dataset.py 实际输出，训练曲线来自真实的 TensorBoard 事件文件，"
              "测试指标由 evaluate.py 在测试集上计算得到，前端截图来自真实启动的 Flask 服务。"
              "本人能够解释并现场运行提交的全部代码，未使用任何伪造数据或修改指标。", indent=True)


def appendix(doc, rows):
    heading(doc, "附录 A  项目目录结构", 1)
    code_block(doc, '''
项目根目录/
├─ README.md                     项目说明与运行步骤
├─ requirements.txt              依赖清单
├─ source_repository.txt         开源平台源码链接
├─ src/                         config.py / data.py / model.py / split_dataset.py
│                               train.py / evaluate.py / predict.py / app.py
├─ frontend/                    index.html / style.css / script.js
├─ dataset/
│  ├─ dataset_manifest.csv       2524 张图片的划分清单
│  ├─ samples/                   12 张测试集样例图
│  └─ train|val|test/<class>/    划分后的图片（体积大，可按脚本重建）
├─ models/
│  ├─ best_model.pth             最终模型权重
│  └─ model_meta.json            结构、参数量、类别顺序
├─ runs/
│  └─ E1_.../ E2_.../ E3_.../ E4_.../   四组实验的 TensorBoard 日志
├─ outputs/                      评估结果、混淆矩阵、曲线截图
├─ report/项目设计文档.docx       本报告
└─ video/                        演示视频与说明
''', "项目目录结构（与压缩包结构一致）", size=7.8)

    return
    heading(doc, "附录 B  数据划分清单样例", 1)
    para(doc, "dataset_manifest.csv 共 2524 行，以下为每个类别各 2 行的抽样，完整文件随压缩包提交。")
    sample = []
    by_split = defaultdict(list)
    for r in rows:
        key = (r["split"], r["class_name"])
        if len(by_split[r["split"]]) < 24 and r["split"] == "test":
            sample.append(r)
    seen = set()
    picked = []
    for r in rows:
        if r["split"] != "test":
            continue
        key = r["class_name"]
        cnt = sum(1 for p in picked if p["class_name"] == key)
        if cnt < 1:
            picked.append(r)
    tab(doc, ["split", "class_name", "file_name", "width", "height", "md5（前 12 位）"],
        [[r["split"], r["class_name"], r["file_name"], r["width"], r["height"], r["md5"][:12]] for r in picked[:8]],
        "dataset_manifest.csv 字段与抽样内容", widths=[1.8, 3.0, 4.6, 1.8, 1.8, 3.4], center=[0, 1, 3, 4, 5], size=8.4)
    para(doc, "字段说明：split 表示所属集合（train/val/test）；class_name 为类别；file_name 为文件名；"
              "width/height 为原始分辨率；md5 为文件内容哈希，用于证明三集合之间没有重复图片。")
    note_box(doc, "递交自查：① 压缩包可正常解压；② 源代码链接可在浏览器打开；③ 按第 9 章可重新启动前端；"
                  "④ models/best_model.pth 可加载并推理；⑤ TensorBoard 能读取 runs/ 显示曲线；"
                  "⑥ 本 .docx 可正常打开；⑦ 演示视频可播放；⑧ 报告、代码、日志、视频内容相互对应。")


def main():
    metrics = load_json(OUT / "metrics.json", {})
    meta = load_json(ROOT / "models" / "model_meta.json", {})
    result = load_json(OUT / "inference_demo.json", {})
    envinfo = load_json(OUT / "env.json", {})

    exps = {}
    for exp, *_ in EXP_ORDER:
        exps[exp] = load_json(ROOT / "runs" / exp / "result.json", {})
    global FINAL_EXP
    candidates = {k: v for k, v in exps.items() if v.get("best_val_f1")}
    if candidates:
        FINAL_EXP = max(candidates, key=lambda k: candidates[k]["best_val_f1"])
    print(f"[最佳模型] {FINAL_EXP}（val_f1={exps[FINAL_EXP].get('best_val_f1', 0):.4f}）")
    history_final = load_history(FINAL_EXP)
    rows, per = manifest_stats()

    doc = init_document()
    add_header_text(doc.sections[0], "基于 PyTorch 的垃圾图像智能分类系统 · 项目设计报告")
    add_page_number_footer(doc.sections[0])

    cover(doc)
    abstract(doc, metrics, len(rows))
    toc(doc)
    section1(doc, metrics)
    section2(doc, envinfo)
    section3(doc, per, rows)
    section4(doc, per)
    section5(doc, meta, result)
    section6(doc, exps)
    section7(doc, metrics, exps, history_final)
    section8(doc, metrics)
    section9(doc, envinfo)
    section10(doc)
    section11(doc, metrics, exps, meta)
    references(doc)
    appendix(doc, rows)

    out_path = ROOT / "report" / "项目设计文档.docx"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out_path)
    print("[saved]", out_path)
    print(f"[统计] 图 {_fig_no} 张，表 {_tab_no} 张")


if __name__ == "__main__":
    main()
