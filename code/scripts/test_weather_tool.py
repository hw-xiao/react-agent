"""
天气工具独立测试：不需要启动模型服务，直接测试 API 调用。

用法:
    .venv\\Scripts\\python.exe -m code.scripts.test_weather_tool
"""

from code.tools.weather import WeatherTool
from code.tools.base import execute_tool, list_tools

print("=" * 50)
print("天气工具模块独立测试")
print("=" * 50)

# 显示已注册工具
tools = list_tools()
print(f"已注册工具: {list(tools.keys())}")

# 测试多个城市
cities = ["北京", "上海", "深圳", "杭州", "广州"]
tool = WeatherTool()

for city in cities:
    print(f"\n--- {city} ---")
    result = tool.execute(city=city)
    print(result)

# 测试统一入口
print("\n" + "=" * 50)
print("测试 execute_tool 统一入口")
print("=" * 50)
result = execute_tool("get_weather", {"city": "成都"})
print(f"execute_tool('get_weather', {{'city': '成都'}}):")
print(result)

print("\n测试完成。")
