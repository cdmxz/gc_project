"""项目全局配置：路径、类别、预处理参数与默认超参数。

所有脚本（训练 / 评估 / 推理 / Web 服务）都从这里读取同一份预处理配置，
避免训练使用 224 而部署使用 128 这类尺寸不一致问题。
"""
from pathlib import Path

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"
MANIFEST_PATH = DATASET_DIR / "dataset_manifest.csv"
SAMPLES_DIR = DATASET_DIR / "samples"

MODELS_DIR = PROJECT_ROOT / "models"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pth"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"          # 评估结果、混淆矩阵、指标
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# ---------------------------------------------------------------- 类别
# 类别顺序与 ImageFolder 的字母序一致，训练/评估/部署必须完全相同
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_NAMES_ZH = {
    "cardboard": "纸板",
    "glass": "玻璃",
    "metal": "金属",
    "paper": "纸张",
    "plastic": "塑料",
    "trash": "其他垃圾",
}
# 四分类投放建议（TrashNet 六类映射到国内四分类习惯）
DISPOSAL_ADVICE = {
    "cardboard": "可回收物：压平后投入可回收物桶",
    "glass": "可回收物：轻放避免破碎，破碎玻璃请包裹后投放",
    "metal": "可回收物：清空内容物后投入可回收物桶",
    "paper": "可回收物：保持干燥洁净，污染纸张请投入其他垃圾",
    "plastic": "可回收物：清空并压扁后投放",
    "trash": "其他垃圾：投入其他垃圾桶，注意不要混入可回收物",
}
NUM_CLASSES = len(CLASS_NAMES)

# ---------------------------------------------------------------- 预处理
IMG_SIZE = 224
RESIZE_SIZE = 256          # 验证/测试先缩放再中心裁剪
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------- 训练默认值
SEED = 42
BATCH_SIZE = 32
NUM_WORKERS = 4
DEFAULT_EPOCHS = 20
DEFAULT_LR = 3e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1

# 推理服务限制
MAX_UPLOAD_MB = 8
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
