"""
集中管理所有配置项，从环境变量读取并设置默认值。
其他模块统一从此处导入配置，避免散落在各处的 os.getenv 调用。
"""

import os

# ── 项目路径 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, "log")

# ── 模型配置 ──
MODEL_ID = os.getenv("QWEN_MODEL_ID", r"D:\models\Qwen2.5-VL-3B-Instruct")
HOST = os.getenv("QWEN_API_HOST", "0.0.0.0")
PORT = int(os.getenv("QWEN_API_PORT", "8000"))

# ── 生成参数 ──
MAX_NEW_TOKENS = int(os.getenv("QWEN_MAX_NEW_TOKENS", "256"))
SERVER_MAX_TOKENS = int(os.getenv("QWEN_MAX_SERVER_TOKENS", "512"))
GEN_TIMEOUT = int(os.getenv("QWEN_GEN_TIMEOUT", "120"))

# ── 图片处理（防 OOM） ──
MAX_IMAGE_DIMENSION = int(os.getenv("QWEN_MAX_IMAGE_DIMENSION", "1080"))
MAX_PIXELS = MAX_IMAGE_DIMENSION * MAX_IMAGE_DIMENSION
MIN_PIXELS = 262144  # 512*512，保证 OCR 级别可读性

# ── 运行模式开关 ──
ENABLE_TOOLS = os.getenv("QWEN_ENABLE_TOOLS", "1").lower() in ("1", "true", "yes")
ENABLE_RAG = os.getenv("QWEN_ENABLE_RAG", "0").lower() in ("1", "true", "yes")
SKIP_GPU_CHECK = os.getenv("QWEN_SKIP_GPU_CHECK", "0").lower() in ("1", "true", "yes")
FORCE_CPU = os.getenv("QWEN_FORCE_CPU", "0").lower() in ("1", "true", "yes")

# ── 邮件与定时任务配置 ──
MAIL_SMTP_HOST = os.getenv("MAIL_SMTP_HOST", "smtp.qq.com")
MAIL_SMTP_PORT = int(os.getenv("MAIL_SMTP_PORT", "465"))
MAIL_USERNAME = os.getenv("MAIL_USERNAME", "416650488@qq.com")
MAIL_PASSWORD = os.getenv("MAIL_PASSWORD", "")
MAIL_RECEIVER = os.getenv("MAIL_RECEIVER", "416650488@qq.com")

# ── RAG 知识库配置 ──
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
RAG_DATA_DIR = os.getenv("RAG_DATA_DIR", os.path.join(DATA_ROOT, "knowledge"))
RAG_TEMP_DIR = os.getenv("RAG_TEMP_DIR", os.path.join(DATA_ROOT, "temp"))
RAG_CACHE_DIR = os.getenv("RAG_CACHE_DIR", os.path.join(DATA_ROOT, "cachedb"))
RAG_CHUNK_STRATEGY = os.getenv("RAG_CHUNK_STRATEGY", "semantic")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
