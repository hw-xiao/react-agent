"""
端到端测试脚本：验证天气查询智能体功能。

用法:
    1. 先启动模型服务: 1start_qwen_local_api.bat
    2. 运行测试: .venv\\Scripts\\python.exe -m code.scripts.test_weather_agent
"""

import json
import sys
import urllib.request

API_URL = "http://127.0.0.1:8000/v1/chat/completions"

TEST_CASES = [
    {"name": "查询北京天气", "messages": [
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "查询北京的当前天气和温度"},
    ]},
    {"name": "查询上海天气", "messages": [
        {"role": "user", "content": "上海今天天气怎么样？温度多少？"},
    ]},
    {"name": "未指定城市", "messages": [
        {"role": "user", "content": "查询当前天气、温度"},
    ]},
    {"name": "非天气问题", "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己"},
    ]},
]


def send_request(messages, max_tokens=512, temperature=0.2):
    """向本地 API 发送 chat completion 请求。"""
    payload = json.dumps({
        "model": "local-qwen",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }).encode("utf-8")

    req = urllib.request.Request(API_URL, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result["choices"][0]["message"]["content"]
    except urllib.error.URLError as e:
        return f"[连接失败] 请确认模型服务已启动\n错误: {e}"
    except Exception as e:
        return f"[请求异常] {e}"


def main():
    """运行所有测试用例。"""
    print("=" * 60)
    print("天气查询智能体测试")
    print("=" * 60)

    # 检查服务是否在线
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as resp:
            health = json.loads(resp.read().decode("utf-8"))
            print(f"服务状态: {health['status']} | 模型: {health['model']}")
    except Exception as e:
        print(f"[错误] 模型服务未启动，请先运行 1start_qwen_local_api.bat\n详情: {e}")
        sys.exit(1)

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}: {case['name']}")
        print(f"{'='*60}")
        print(f"用户: {case['messages'][-1]['content']}")
        print("-" * 40)
        answer = send_request(case["messages"])
        print(f"助手: {answer}\n")

    print("=" * 60)
    print("测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
