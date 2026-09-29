"""财经热点检索、摘要与邮件推送模块。

功能：
- 启动时立即执行一次检索和总结
- 每日 08:00 定时触发一次
- 抓取 Google News RSS 中的财经资讯
- 汇总热点方向并尝试通过 SMTP 发送到指定邮箱
"""

from __future__ import annotations

import datetime as dt
import re
import smtplib
import threading
import time
from email.header import Header
from email.mime.text import MIMEText
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET

from code.config.settings import (
    MAIL_PASSWORD,
    MAIL_RECEIVER,
    MAIL_SMTP_HOST,
    MAIL_SMTP_PORT,
    MAIL_USERNAME,
)
from code.utils.logger import get_logger

log = get_logger("market")

SEARCH_QUERIES = [
    "财经热点",
    "宏观经济",
    "A股",
    "美联储",
    "大宗商品",
    "数字货币",
    "房地产政策",
]

NEWS_RSS_SOURCES = [
    "https://rss.sina.com.cn/roll/mobile/hot_daily_100.xml",
    "https://finance.sina.com.cn/stock/",
    "https://www.cnbeta.com/backend.php?feed=rss2",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s={query}&region=US&lang=en-US",
    "https://news.google.com/rss/search?q={query}&hl=zh-CN&gl=CN&ceid=CN:zh-Hans",
    "https://www.bing.com/news/search?q={query}&format=rss",
]

CATEGORY_KEYWORDS = {
    "宏观政策": [
        "美联储",
        "降息",
        "利率",
        "央行",
        "财政",
        "政策",
        "经济",
        "房地产",
        "消费",
        "货币",
        "通胀",
    ],
    "A股与行业": [
        "A股",
        "上证",
        "沪指",
        "深证",
        "创业板",
        "半导体",
        "芯片",
        "电池",
        "汽车",
        "医药",
        "AI",
        "科技",
        "金融",
        "消费电子",
    ],
    "大宗商品": [
        "油价",
        "原油",
        "黄金",
        "铜",
        "铁矿石",
        "铝",
        "大宗商品",
        "能源",
        "农产品",
    ],
    "数字资产": [
        "比特币",
        "以太坊",
        "数字货币",
        "加密",
        "数字资产",
        "币圈",
        "稳定币",
    ],
    "全球市场": [
        "美股",
        "欧股",
        "港股",
        "外汇",
        "美元",
        "欧元",
        "日本",
        "全球市场",
        "贸易",
        "地缘",
    ],
}

_monitor_thread: Optional[threading.Thread] = None
_monitor_lock = threading.Lock()


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "")).strip()


def _fetch_url_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_finance_news(limit: int = 12) -> List[Dict[str, str]]:
    """抓取财经资讯标题与链接；优先国内 RSS，失败时尝试备选源，最终走离线兜底。"""
    seen = set()
    items: List[Dict[str, str]] = []

    def parse_news_xml(xml_text: str):
        root = ET.fromstring(xml_text)
        for item in root.findall("./channel/item"):
            title = _normalize_title(item.findtext("title") or "")
            link = (item.findtext("link") or "").strip()
            pub_date = (item.findtext("pubDate") or "").strip()
            if not title or not link:
                continue
            dedupe_key = title.lower()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            items.append({
                "title": title,
                "link": link,
                "published_at": pub_date,
            })
            if len(items) >= limit:
                return True
        return False

    for query in SEARCH_QUERIES:
        encoded = quote(query)
        rss_candidates = [
            source.format(query=encoded) if "{query}" in source else source
            for source in NEWS_RSS_SOURCES
        ]

        for rss_url in rss_candidates:
            try:
                xml_text = _fetch_url_text(rss_url)
                if parse_news_xml(xml_text):
                    return items[:limit]
            except Exception as exc:  # pragma: no cover - 仅记录网络异常
                log.warning(f"抓取财经资讯失败（{query}，源={rss_url}）: {exc}")

    if not items:
        log.warning("当前环境无法访问外部新闻 RSS，已启用离线兜底摘要。")

    return items[:limit]


def build_market_briefing(headlines: Iterable[str]) -> str:
    """构建中文财经热点摘要。"""
    headline_list = [_normalize_title(item) for item in headlines if _normalize_title(item)]
    if not headline_list:
        return (
            "财经热点摘要：\n"
            "截至当前时间，未抓取到新的财经资讯，建议重点关注流动性变化、政策落地和行业资金流向。\n\n"
            "关注方向：\n"
            "- 宏观政策：观察央行政策表述、降息预期和财政刺激落地情况。\n"
            "- A股与行业：跟踪大盘指数、行业估值和资金流向变化。\n"
            "- 大宗商品：关注油价、铜价和黄金等跨周期指标。\n"
            "- 数字资产：监控比特币和稳定币市场的风险偏好变化。\n"
            "- 全球市场：关注美联储政策、地缘风险与外汇走势。"
        )

    matched: Dict[str, List[str]] = {category: [] for category in CATEGORY_KEYWORDS}
    for headline in headline_list:
        lowered = headline.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(keyword.lower() in lowered for keyword in keywords):
                matched[category].append(headline)
                break

    lines = [
        "财经热点摘要：",
        f"截至 {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}，已检索到 {len(headline_list)} 条财经资讯，当前热点以政策、资产价格和跨市场资金流向为主。",
        "",
        "关注方向：",
    ]

    ordered_categories = [
        "宏观政策",
        "A股与行业",
        "大宗商品",
        "数字资产",
        "全球市场",
    ]

    for category in ordered_categories:
        items = matched.get(category, [])
        if not items:
            continue
        sample = "；".join(items[:2])
        lines.append(f"- {category}：{sample}")

    if not any(matched.values()):
        lines.append("- 宏观政策：政策利好、流动性和市场情绪变化是当前最关键的观察变量。")
        lines.append("- A股与行业：关注大盘指数、行业资金流向和估值修复节奏。")

    return "\n".join(lines)


def build_email_body(report: Dict[str, Any]) -> str:
    summary = report.get("summary", "")
    headlines = report.get("headlines", [])
    lines = [
        "财经热点观察日报",
        f"更新时间：{report.get('generated_at', dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
        "",
        summary,
        "",
        "精选资讯：",
    ]

    for idx, item in enumerate(headlines[:8], start=1):
        title = item.get("title", "")
        link = item.get("link", "")
        lines.append(f"{idx}. {title}")
        if link:
            lines.append(f"   链接：{link}")

    return "\n".join(lines)


def send_market_report_email(report: Dict[str, Any]) -> bool:
    """通过 SMTP 发送财经热点总结邮件。未配置密码时返回 False。"""
    if not MAIL_PASSWORD:
        log.warning("未配置 MAIL_PASSWORD，跳过发邮件。使用 MAIL_USERNAME=%s, MAIL_RECEIVER=%s", MAIL_USERNAME, MAIL_RECEIVER)
        return False

    try:
        msg = MIMEText(build_email_body(report), "plain", "utf-8")
        msg["Subject"] = Header(f"财经热点观察日报 {dt.datetime.now().strftime('%Y-%m-%d')}", "utf-8")
        msg["From"] = MAIL_USERNAME
        msg["To"] = MAIL_RECEIVER

        with smtplib.SMTP_SSL(MAIL_SMTP_HOST, MAIL_SMTP_PORT) as server:
            server.login(MAIL_USERNAME, MAIL_PASSWORD)
            server.sendmail(MAIL_USERNAME, [MAIL_RECEIVER], msg.as_string())

        log.info("财经热点邮件已发送至 %s", MAIL_RECEIVER)
        return True
    except Exception as exc:  # pragma: no cover - 依赖外部邮件环境
        log.error(f"财经热点邮件发送失败: {exc}", exc_info=True)
        return False


def run_market_report_cycle() -> Dict[str, Any]:
    """执行一轮检索：抓取资讯、归纳热点、发送邮件。"""
    news_items = fetch_finance_news(limit=12)
    titles = [item["title"] for item in news_items]
    summary = build_market_briefing(titles)
    report = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": summary,
        "headlines": news_items,
    }

    log.info("财经热点汇总生成完成，涵盖 %s 条资讯", len(news_items))
    log.info(summary)

    sent = send_market_report_email(report)
    report["email_sent"] = sent
    report["email_receiver"] = MAIL_RECEIVER
    return report


def start_market_monitor() -> Optional[threading.Thread]:
    """启动后台定时器：应用启动时立即执行一次，并且每天 08:00 执行一次。"""
    global _monitor_thread

    with _monitor_lock:
        if _monitor_thread and _monitor_thread.is_alive():
            return _monitor_thread

        def worker() -> None:
            log.info("启动财经热点监控任务，先执行一次启动测试")
            run_market_report_cycle()
            while True:
                now = dt.datetime.now()
                next_run = dt.datetime.combine(now.date(), dt.time(8, 0))
                if now >= next_run:
                    next_run += dt.timedelta(days=1)
                sleep_seconds = (next_run - now).total_seconds()
                log.info("财经热点监控下一次执行时间：%s", next_run.strftime("%Y-%m-%d %H:%M:%S"))
                time.sleep(sleep_seconds)
                log.info("执行每日 08:00 财经热点检索任务")
                run_market_report_cycle()

        _monitor_thread = threading.Thread(target=worker, name="market-monitor", daemon=True)
        _monitor_thread.start()
        return _monitor_thread
