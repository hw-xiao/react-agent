"""
天气查询工具：使用 Open-Meteo 免费 API（无需 API Key）。
支持中英文城市名称，返回实时天气、温度、湿度、风速等信息。

API 文档: https://open-meteo.com/
"""

import json
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from code.tools.base import BaseTool, register_tool
from code.utils.logger import get_logger

log = get_logger("weather")

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO 天气代码映射
WMO_WEATHER_CODES = {
    0: "晴天", 1: "晴间多云", 2: "多云", 3: "阴天",
    45: "有雾", 48: "有雾凇",
    51: "小毛毛雨", 53: "中毛毛雨", 55: "大毛毛雨",
    56: "冻毛毛雨(轻)", 57: "冻毛毛雨(密)",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨(轻)", 67: "冻雨(强)",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "阵雪",
    80: "小阵雨", 81: "中阵雨", 82: "大阵雨",
    85: "小阵雪", 86: "大阵雪",
    95: "雷暴", 96: "雷暴伴小冰雹", 99: "雷暴伴大冰雹",
}


def _http_get(url: str, timeout: int = 10) -> Dict[str, Any]:
    """发送 HTTP GET 请求并解析 JSON 响应。"""
    req = urllib.request.Request(url, headers={"User-Agent": "local-qwen-agent/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def geocode_city(city: str) -> Optional[Dict[str, Any]]:
    """
    将城市名称转换为经纬度坐标。

    Args:
        city: 城市名称（中英文均可）

    Returns:
        包含 latitude/longitude/name 等字段的字典，找不到则返回 None
    """
    params = urllib.parse.urlencode({
        "name": city, "count": 1, "language": "zh", "format": "json",
    })
    url = f"{GEOCODING_URL}?{params}"
    data = _http_get(url)
    results = data.get("results")
    if not results:
        return None
    return results[0]


class WeatherTool(BaseTool):
    """天气查询工具，使用 Open-Meteo 免费 API。"""

    @property
    def name(self) -> str:
        return "get_weather"

    @property
    def description(self) -> str:
        return "查询指定城市的当前实时天气信息，包括天气状况、温度、湿度、风速等。当用户询问天气、温度、下雨等情况时使用此工具。"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "要查询天气的城市名称，支持中英文，如 北京、上海、Shenzhen",
                }
            },
            "required": ["city"],
        }

    def execute(self, city: str = "") -> str:
        """
        查询指定城市的当前实时天气。

        Args:
            city: 城市名称（中英文均可）

        Returns:
            格式化的天气信息字符串
        """
        if not city:
            return "错误：缺少城市名称参数"

        log.info(f"开始查询城市天气: {city}")

        # 步骤1：地理编码 — 城市名 → 经纬度
        geo = geocode_city(city)
        if not geo:
            log.warning(f"未找到城市: {city}")
            return f"未找到城市「{city}」，请确认城市名称是否正确。"

        lat = geo["latitude"]
        lon = geo["longitude"]
        name = geo.get("name", city)
        country = geo.get("country", "")
        admin1 = geo.get("admin1", "")
        timezone = geo.get("timezone", "Asia/Shanghai")
        log.info(f"地理编码成功: {name} ({lat}, {lon})")

        # 步骤2：查询天气预报
        params = urllib.parse.urlencode({
            "latitude": lat, "longitude": lon,
            "current": ",".join([
                "temperature_2m", "relative_humidity_2m", "apparent_temperature",
                "weather_code", "wind_speed_10m", "wind_direction_10m", "pressure_msl",
            ]),
            "timezone": timezone, "forecast_days": 1,
        })
        url = f"{FORECAST_URL}?{params}"
        data = _http_get(url)
        current = data.get("current", {})

        # 步骤3：解析天气数据
        temp = current.get("temperature_2m", "N/A")
        humidity = current.get("relative_humidity_2m", "N/A")
        apparent = current.get("apparent_temperature", "N/A")
        wmo_code = current.get("weather_code", 0)
        wind_speed = current.get("wind_speed_10m", "N/A")
        wind_dir = current.get("wind_direction_10m", "N/A")
        pressure = current.get("pressure_msl", "N/A")
        weather_desc = WMO_WEATHER_CODES.get(wmo_code, "未知")

        # 步骤4：格式化输出
        location = name
        if admin1 and admin1 != name:
            location += f"，{admin1}"
        if country:
            location += f"，{country}"

        result = (
            f"城市：{location}\n"
            f"天气：{weather_desc}\n"
            f"温度：{temp}°C（体感温度 {apparent}°C）\n"
            f"湿度：{humidity}%\n"
            f"风速：{wind_speed} km/h（风向 {wind_dir}°）\n"
            f"气压：{pressure} hPa"
        )
        log.info(f"天气查询完成: {name} {weather_desc} {temp}°C")
        return result


# 模块加载时自动注册
register_tool(WeatherTool())
