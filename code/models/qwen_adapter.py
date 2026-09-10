"""
Qwen 多模态模型适配器：负责模型加载、单轮/多轮推理。
支持图片+文本输入，自动 GPU 内存管理。
"""

import os
import threading
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from code.config.settings import (
    FORCE_CPU, MAX_PIXELS, MIN_PIXELS, SKIP_GPU_CHECK,
)
from code.utils.image import load_image
from code.utils.logger import get_logger, log_model_generate, log_model_response

log = get_logger("model")


class QwenAdapter:
    """
    Qwen 多模态模型适配器。

    职责：
      - 加载模型和 processor 到 GPU/CPU
      - 单轮推理 generate()：从 messages 中提取最后一条 user 消息
      - 多轮推理 generate_multiturn()：支持 system+多轮对话上下文
      - GPU 显存检查与回收
    """

    def __init__(self, model_id: str):
        """
        加载模型和 processor。

        Args:
            model_id: 模型路径或 HuggingFace ID
        """
        self.model_id = model_id
        self.device = "cpu" if FORCE_CPU or not torch.cuda.is_available() else "cuda"

        # 根据设备和硬件能力选择数据类型
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        elif self.device == "cuda":
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

        device_map = {"": 0} if self.device == "cuda" else None

        load_kwargs = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "local_files_only": True,
            "device_map": device_map,
        }

        log.info(f"开始加载模型: {model_id}")
        log.info(f"设备: {self.device} | 数据类型: {self.dtype}")

        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, local_files_only=True
        )

        if self.device == "cpu":
            self.model.to("cpu")

        self._infer_lock = threading.Lock()
        log.info("模型加载完成")

    def _check_gpu_memory(self) -> None:
        """检查 GPU 可用显存，不足时尝试回收缓存。"""
        if self.device != "cuda":
            return
        free, _ = torch.cuda.mem_get_info()
        free_mb = free / 1024 / 1024
        if free_mb < 512:
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info()
            free_mb = free / 1024 / 1024
        if free_mb < 256:
            raise RuntimeError(f"GPU 显存不足，剩余 {free_mb:.0f}MB")
        log.debug(f"GPU 可用显存: {free_mb:.0f}MB")

    def _extract_generated_text(self, outputs, inputs) -> str:
        """
        从模型输出中提取生成的文本部分（去掉输入 prompt）。

        优先通过 token 长度截取；失败时退回到按 'assistant' 关键词分割。
        """
        full_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        try:
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            generated = self.processor.batch_decode(
                generated_tokens.unsqueeze(0), skip_special_tokens=True
            )[0]
            if generated.strip():
                return generated.strip()
        except Exception:
            pass
        if "assistant" in full_text:
            return full_text.rsplit("assistant", 1)[-1].strip()
        return full_text.strip()

    def _extract_prompt_and_images(self, messages: List[Dict[str, Any]]):
        """从 messages 中提取最后一条 user 消息的文本和图片。"""
        last_user = None
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m
                break
        if last_user is None:
            raise ValueError("No user message found")

        content = last_user.get("content", "")
        text_parts: List[str] = []
        image_parts: List[Image.Image] = []

        if isinstance(content, str):
            prompt = content
        else:
            prompt = ""
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif item.get("type") in {"image", "image_url"}:
                        value = item.get("image") or item.get("image_url")
                        if isinstance(value, dict):
                            value = value.get("url")
                        image_parts.append(load_image(value))
                elif isinstance(item, str):
                    text_parts.append(item)
            prompt = "\n".join(text_parts).strip()

        return prompt, image_parts

    def generate(self, messages: List[Dict[str, Any]], max_tokens: int, temperature: float) -> str:
        """
        单轮推理：提取最后一条 user 消息进行生成。

        适用于简单的图片描述、单次问答场景。
        """
        prompt, images = self._extract_prompt_and_images(messages)
        prompt = prompt or "请描述这张图片。"

        # 提取 system 消息
        system_content = self._extract_system_content(messages)

        qwen_messages = []
        if system_content:
            qwen_messages.append({"role": "system", "content": system_content})
        qwen_messages.append({
            "role": "user",
            "content": [
                *[{"type": "image", "image": img} for img in images],
                {"type": "text", "text": prompt},
            ],
        })

        return self._run_inference("generate", qwen_messages, images, max_tokens, temperature)

    def generate_multiturn(
        self, messages: List[Dict[str, Any]], max_tokens: int, temperature: float
    ) -> str:
        """
        多轮推理：保留完整的对话历史（system + 多轮 user/assistant）。

        适用于 ReAct 智能体循环等需要上下文的场景。
        """
        system_content, chat_msgs, images = self._parse_multiturn_messages(messages)

        qwen_messages: List[Dict[str, Any]] = []
        if system_content:
            qwen_messages.append({"role": "system", "content": system_content})

        last_user_idx = None
        for msg in chat_msgs:
            if msg["role"] == "user":
                last_user_idx = len(qwen_messages)
            qwen_messages.append(msg)

        # 如果有图片，附加到最后一条 user 消息
        if last_user_idx is not None and images:
            qwen_messages[last_user_idx] = {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": qwen_messages[last_user_idx]["content"]},
                ],
            }

        return self._run_inference("generate_multiturn", qwen_messages, images, max_tokens, temperature)

    def _extract_system_content(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """从 messages 中提取 system 消息内容。"""
        for m in messages:
            if m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, list):
                    return "\n".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                elif isinstance(content, str):
                    return content
                else:
                    return str(content) if content else None
        return None

    def _parse_multiturn_messages(self, messages: List[Dict[str, Any]]):
        """将 OpenAI 格式的 messages 解析为 Qwen chat template 格式。"""
        system_content = None
        chat_msgs: List[Dict[str, str]] = []
        images: List[Image.Image] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content")

            if role == "system":
                if isinstance(content, list):
                    system_content = "\n".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                elif isinstance(content, str):
                    system_content = content
                else:
                    system_content = str(content) if content else None
                continue

            if isinstance(content, str):
                chat_msgs.append({"role": role, "content": content})
            elif isinstance(content, list):
                text_parts: List[str] = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            text_parts.append(item.get("text", ""))
                        elif item.get("type") in ("image", "image_url"):
                            value = item.get("image") or item.get("image_url")
                            if isinstance(value, dict):
                                value = value.get("url")
                            if value and role == "user":
                                images.append(load_image(value))
                chat_msgs.append({"role": role, "content": "\n".join(text_parts)})
            else:
                chat_msgs.append({"role": role, "content": str(content) if content else ""})

        return system_content, chat_msgs, images

    def _run_inference(
        self,
        method: str,
        qwen_messages: List[Dict[str, Any]],
        images: List[Image.Image],
        max_tokens: int,
        temperature: float,
    ) -> str:
        """执行模型推理的内部方法，加锁防止并发。"""
        log_model_generate(method, len(qwen_messages), max_tokens)

        if not SKIP_GPU_CHECK:
            self._check_gpu_memory()

        with self._infer_lock:
            text = self.processor.apply_chat_template(
                qwen_messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text],
                images=images if images else None,
                max_pixels=MAX_PIXELS,
                min_pixels=MIN_PIXELS,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, 2048),
                    do_sample=temperature > 0,
                    temperature=temperature,
                    use_cache=False,
                )

            response = self._extract_generated_text(outputs, inputs)
            log_model_response(method, response)

            del inputs, outputs
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return response
