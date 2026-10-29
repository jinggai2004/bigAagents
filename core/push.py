"""通知推送（钉钉 / 飞书）

Phase 1：从 core/review_agent 抽出，供「每日复盘」与「盘中异动预警」共用。
原先只有复盘会推送，异动预警若再复制一份 webhook 逻辑就会出现两套实现。

配置来源：config.settings.config（.env 由配置页写入并 reload，见 reload_config）。
未配置 webhook 时返回提示而非报错——本地开发不应因为没配机器人就失败。

2026-10-23 改造（针对"飞书通知要有针对性"）：
  · 飞书改用 **post 富文本**（此前用 text，Markdown 全是字面量，`**加粗**`、
    `##` 标题、`|` 表格在手机上显示成一坨源码）；
  · 加 **推送留痕 + 幂等**（`push_log` 表）：同日同一语义内容只推一次
    （10-23 实测：手动重放 + 定时任务把同一份复盘推了两遍）；
  · 正文超限时**留截断标记**（此前静默截断，读者不知道内容被切了）。
"""

import hashlib
import logging
import re
from datetime import datetime

import requests

from config.settings import config

_logger = logging.getLogger(__name__)

DINGTALK_MAX_CHARS = 26000
FEISHU_MAX_CHARS = 9000
# 飞书 post 单条消息的行数上限（官方 200 行，留足余量）
FEISHU_MAX_LINES = 120


def _push_dingtalk(webhook: str, title: str, markdown_text: str):
    payload = {"msgtype": "markdown",
               "markdown": {"title": title, "text": markdown_text[:DINGTALK_MAX_CHARS]}}
    resp = requests.post(webhook, json=payload, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("errcode") != 0:
        raise RuntimeError(f"钉钉推送失败: {body}")


# ── 飞书 post 富文本：Markdown → post 元素 ─────────────────────────────
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_HEAD = re.compile(r"^\s{0,3}#{1,6}\s*")
_MD_BULLET = re.compile(r"^\s*[-*]\s+")
_MD_QUOTE = re.compile(r"^\s*>\s?")


def _inline(text: str) -> list[dict]:
    """一行文本 → post 元素数组（`**xx**` 拆成加粗片段）。"""
    out: list[dict] = []
    pos = 0
    for m in _MD_BOLD.finditer(text):
        if m.start() > pos:
            out.append({"tag": "text", "text": text[pos:m.start()]})
        out.append({"tag": "text", "text": m.group(1), "style": ["bold"]})
        pos = m.end()
    if pos < len(text):
        out.append({"tag": "text", "text": text[pos:]})
    return out or [{"tag": "text", "text": ""}]


def markdown_to_post_lines(markdown_text: str, max_lines: int = FEISHU_MAX_LINES) -> list:
    """Markdown → 飞书 post 的 `content`（行数组）。

    为什么自己解析而不靠飞书的 `md` 标签：`md` 标签在不同版本/客户端上
    渲染不一致（表格尤其不稳）。自己解析成 text + bold 样式，结果确定。
    ⚠️ 表格行（`| a | b |`）保留为纯文本行 —— post 不支持表格，
       宁可让用户看到"列 | 列"，也不要假装排好了版。
    """
    lines: list = []
    for raw in (markdown_text or "").splitlines():
        if not raw.strip():
            lines.append([{"tag": "text", "text": " "}])
            continue
        if len(lines) >= max_lines:
            break
        text = raw
        head = _MD_HEAD.match(text)
        if head:
            body = _MD_HEAD.sub("", text).strip()
            lines.append([{"tag": "text", "text": body, "style": ["bold"]}])
            continue
        if _MD_QUOTE.match(text):
            body = _MD_QUOTE.sub("", text).strip()
            lines.append([{"tag": "text", "text": "｜" + body}])
            continue
        if _MD_BULLET.match(text):
            body = _MD_BULLET.sub("", text)
            lines.append(_inline("• " + body))
            continue
        lines.append(_inline(text))
    if len(lines) >= max_lines:
        lines.append([{"tag": "text", "text": "…（内容超长，已截断，完整版见本地报告）",
                       "style": ["bold"]}])
    return lines


def _push_feishu(webhook: str, title: str, markdown_text: str):
    """飞书自定义机器人 post 富文本推送。

    ⚠️ post 不被支持时**降级**回 text（宁可排版差一点，也不能不送达）。
    """
    payload = {"msg_type": "post",
               "content": {"post": {"zh_cn": {
                   "title": title or "bigAagents",
                   "content": markdown_to_post_lines(markdown_text)}}}}
    resp = requests.post(webhook, json=payload, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") not in (0, None) or body.get("StatusCode") not in (0, None):
        # 降级：post 不可用 ⇒ 用纯文本再试一次（去掉 Markdown 记号）
        plain = re.sub(r"[*#>]", "", markdown_text)
        _push_feishu_text(webhook, plain)
        return
    return body


def _push_feishu_text(webhook: str, plain_text: str):
    payload = {"msg_type": "text", "content": {"text": plain_text[:FEISHU_MAX_CHARS]}}
    resp = requests.post(webhook, json=payload, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    if body.get("code") not in (0, None) or body.get("StatusCode") not in (0, None):
        raise RuntimeError(f"飞书推送失败: {body}")


def webhook_status() -> dict:
    """当前 webhook 配置情况（供 UI / 诊断显示）"""
    ding = (config.DINGTALK_WEBHOOK or "").strip()
    feishu = (config.FEISHU_WEBHOOK or "").strip()
    return {
        "dingtalk": bool(ding and "YOUR_" not in ding),
        "feishu": bool(feishu and "YOUR_" not in feishu),
    }


def _content_hash(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def push_markdown(markdown_text: str, title: str = None, dedupe_key: str = None) -> dict:
    """推送 Markdown 文本到已配置的钉钉/飞书（都未配置则跳过）。

    Args:
        markdown_text: 正文（Markdown）。
        title: 消息标题（**飞书 post 现在会真正用它**，此前只有钉钉用）。
        dedupe_key: **语义幂等键**（如 `review:2026-10-23`）。给了它，
            同一天同一渠道同一键**只推一次** —— 修掉"手动重放 + 定时任务"
            把同一份内容推两遍的问题。不给则按正文哈希去重（同内容不重复）。

    Returns: {"钉钉": "已推送"|"失败: ...", "飞书": ..., "提示": ...}
    """
    results = {}
    title = title or f"bigAagents {datetime.now():%Y-%m-%d %H:%M}"
    day = datetime.now().strftime("%Y-%m-%d")
    text = markdown_text or ""
    # 幂等键优先用语义键；否则用正文哈希
    base_hash = _content_hash(dedupe_key) if dedupe_key else _content_hash(text)

    def _already_sent(channel: str) -> bool:
        try:
            from db import store
            return store.push_already_sent(day, channel, base_hash)
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[推送] 幂等检查失败（按未推过处理）: %s", e)
            return False

    def _record(channel: str, status: str, detail: str = "") -> None:
        try:
            from db import store
            store.log_push(day, channel, title, base_hash, len(text), status, detail)
        except Exception as e:                              # noqa: BLE001
            _logger.warning("[推送] 写留痕失败（不影响推送）: %s", e)

    ding = (config.DINGTALK_WEBHOOK or "").strip()
    if ding and "YOUR_" not in ding:
        if _already_sent("钉钉"):
            results["钉钉"] = "已推送过（内容未变，跳过）"
        else:
            try:
                _push_dingtalk(ding, title, text)
                results["钉钉"] = "已推送"
                _record("钉钉", "ok")
            except Exception as e:
                _logger.warning("[推送] 钉钉失败: %s", e)
                results["钉钉"] = f"失败: {e}"
                _record("钉钉", "failed", str(e))
    feishu = (config.FEISHU_WEBHOOK or "").strip()
    if feishu and "YOUR_" not in feishu:
        if _already_sent("飞书"):
            results["飞书"] = "已推送过（内容未变，跳过）"
        else:
            try:
                _push_feishu(feishu, title, text)
                results["飞书"] = "已推送"
                _record("飞书", "ok")
            except Exception as e:
                _logger.warning("[推送] 飞书失败: %s", e)
                results["飞书"] = f"失败: {e}"
                _record("飞书", "failed", str(e))
    if not results:
        results["提示"] = "未配置 webhook，内容仅保存本地"
    return results
