# 本地搭建千问多模态模型（Qwen2.5-VL）

这是一个最小可运行的本地 Qwen 多模态推理示例，适合 Windows + NVIDIA GPU 的开发环境。

## 1. 先决条件

- Python 3.10+
- NVIDIA GPU（推荐 8GB+ 显存，7B 模型更稳妥）
- 至少 16GB 内存
- 可访问 Hugging Face

如果你没有 GPU，可以运行 3B 级模型，但生成速度会明显变慢。

## 2. 创建虚拟环境

你现在的 Python 在 `D:\tools\Python312`，不要依赖 PATH。请在项目根目录执行：

```powershell
D:\tools\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

然后激活虚拟环境：

```powershell
.\.venv\Scripts\activate
```

## 3. 安装 PyTorch

根据你的显卡 CUDA 版本选择对应安装命令。下面给出 CUDA 12.1 的常用版本：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

如果你是 CPU-only 环境，则执行：

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio
```

## 4. 安装 Python 依赖

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 5. 准备一张测试图片

把图片放到项目目录，例如：

```text
./demo.jpg
```

## 6. 运行多模态推理

默认使用轻量版 Qwen2.5-VL-3B-Instruct：

```powershell
.\.venv\Scripts\python.exe run_qwen_vl.py --model Qwen/Qwen2.5-VL-3B-Instruct --image .\demo.jpg --prompt "请详细描述这张图片中的内容。"
```

如果想切换更大的模型：

```powershell
.\.venv\Scripts\python.exe run_qwen_vl.py --model Qwen/Qwen2.5-VL-7B-Instruct --image .\demo.jpg --prompt "请详细描述这张图片中的内容。"
```

## 7. 常见问题

### 1）下载模型很慢

可试试 Hugging Face 镜像：

```powershell
set HF_ENDPOINT=https://hf-mirror.com
```

或者在命令前加：

```powershell
set HF_HUB_ENABLE_HF_TRANSFER=1
```

这个可以用
set HF_ENDPOINT=https://hf-mirror.com
set HF_XET_HIGH_PERFORMANCE=0
set HF_HUB_ENABLE_HF_TRANSFER=0
set HF_HUB_DISABLE_XET=1

D:\Code\localcvmodel\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='Qwen/Qwen2.5-VL-3B-Instruct', local_dir=r'D:\models\Qwen2.5-VL-3B-Instruct'))"

### 2）显存不足

- 改用更小的模型，例如 3B
- 调小 `--max-new-tokens`
- 减少图片分辨率

### 3）找不到 `transformers` 或 `accelerate`

重新在虚拟环境里安装：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 8. 推荐模型

- Qwen/Qwen2.5-VL-3B-Instruct：适合本地入门，资源占用较低
- Qwen/Qwen2.5-VL-7B-Instruct：适合更高质量输出
- Qwen/Qwen2.5-VL-32B-Instruct：适合高端显卡和更强算力

## 9. 说明

这是一个本地开发 demo，适合你在本机快速验证千问多模态模型的能力。后续如果你想接入 Web UI、API 服务、或者聊天机器人，可以继续扩展。

如果你愿意，我也可以继续帮你把它升级成：

- Web 交互页面（Gradio / Streamlit）
- FastAPI 本地服务
- 批量图片识别脚本
- 可视化聊天界面


## 10. 手动附加

# 启动服务（与之前相同）
1start_qwen_local_api.bat

# 或手动启动
cd D:\Code\localcvmodel
.\.venv\Scripts\activate
python -m code.app

# 测试天气工具（不需要模型）
python -m code.scripts.test_weather_tool

# 端到端测试（需要先启动服务）
python -m code.scripts.test_weather_agent



重构后目录结构
D:\Code\localcvmodel\
├── code/                          ← 代码根目录（与 .venv 同级）
│   ├── __init__.py
│   ├── app.py                     ← 主入口（python -m code.app）
│   ├── config/
│   │   └── settings.py            ← 配置集中管理（所有环境变量）
│   ├── utils/
│   │   ├── logger.py              ← 日志系统（控制台+文件，ReAct专用函数）
│   │   └── image.py               ← 图片加载预处理
│   ├── tools/                     ← 工具模块（可扩展）
│   │   ├── base.py                ← 工具基类 + 注册表
│   │   └── weather.py             ← 天气查询工具
│   ├── models/
│   │   └── qwen_adapter.py        ← 模型加载 + 单轮/多轮推理
│   ├── agent/
│   │   ├── prompts.py             ← ReAct 系统提示词（动态生成）
│   │   └── react.py               ← ReAct 智能体循环
│   ├── api/
│   │   ├── schemas.py             ← Pydantic 数据模型
│   │   └── routes.py              ← FastAPI 路由
│   └── scripts/
│       ├── run_qwen_vl.py         ← 多模态推理脚本
│       ├── test_weather_agent.py  ← 端到端测试
│       └── test_weather_tool.py   ← 工具独立测试
├── log/                           ← 日志目录（与 code 同级）
│   └── agent_YYYYMMDD.log         ← 按日期轮转的日志文件
├── .venv/
├── 1start_qwen_local_api.bat      ← 启动脚本（已更新为 python -m code.app）
├── requirements.txt
├── README.md
└── ...

架构改进要点
改进项	        说明
模块分层	    config → utils → tools → models → agent → api，单向依赖，每层职责清晰
配置集中	    settings.py 统一管理所有 os.getenv，其他模块只读导入
工具可扩展	    base.py 提供 BaseTool 抽象基类 + 注册表，新增工具只需继承并 register_tool()
提示词动态生成  prompts.py 从工具注册表自动拼接工具说明，增删工具无需改提示词
日志系统	    logger.py 同时输出控制台(INFO+)和文件(DEBUG+)，ReAct 各阶段有专用日志函数
方法注释	    所有公开方法均有 docstring，说明参数、返回值和职责