"""
多模态推理脚本：对单张图片进行推理。
独立于 API 服务运行，适合快速测试模型能力。

用法:
    .venv\\Scripts\\python.exe -m code.scripts.run_qwen_vl --image demo.jpg --prompt "请描述这张图片"
"""

import argparse
import os

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from code.utils.logger import get_logger

log = get_logger("cli")


def load_model(model_id: str, low_vram: bool = True):
    """加载模型和 processor，返回 (model, processor, device) 元组。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if device == "cuda" else torch.float32

    model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True, "trust_remote_code": True}
    if device == "cuda":
        model_kwargs["device_map"] = {"": 0} if low_vram else "auto"

    log.info(f"加载模型: {model_id} | 设备: {device}")
    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if device == "cpu":
        model.to(device)
    return model, processor, device


def build_messages(prompt: str, image_path: str):
    """构造 Qwen chat template 格式的消息。"""
    return [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt},
    ]}]


def main():
    """命令行入口：解析参数 → 加载模型 → 推理 → 输出结果。"""
    parser = argparse.ArgumentParser(description="本地 Qwen 多模态推理")
    parser.add_argument("--model", default=r"D:\models\Qwen2.5-VL-3B-Instruct", help="模型路径")
    parser.add_argument("--image", required=True, help="图片路径")
    parser.add_argument("--prompt", default="请详细描述这张图片。", help="提示词")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="最大生成 token 数")
    parser.add_argument("--max-pixels", type=int, default=786432, help="图片最大像素数")
    parser.add_argument("--no-low-vram", action="store_true", help="禁用低显存模式")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = Image.open(args.image).convert("RGB")
    model, processor, device = load_model(args.model, low_vram=not args.no_low_vram)

    messages = build_messages(args.prompt, args.image)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], max_pixels=args.max_pixels, min_pixels=262144, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=min(args.max_new_tokens, 256), do_sample=False, use_cache=False)

    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]
    print(f"\n=== Model Response ===\n{response}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
