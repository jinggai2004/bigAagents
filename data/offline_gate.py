"""离线闸门：`BIGA_OFFLINE=1` 时**任何外呼直接抛 `OfflineViolation`**。

为什么需要它（D4，2026-10-18 审查暴露）：

    `bars.offline_mode()` 原来只是「消费方自觉遵守的约定」——
    `if bars.offline_mode(): 缺失别联网`。但全仓只有 4 处 K 线兜底判断了它，
    实测设 `BIGA_OFFLINE=1` 后打分仍会打网络（腾讯 K 线、tushare moneyflow），
    单只 3.7s。**约定无法强制** ⇒ 改在 request 层拦截。

设计取舍：

  * 拦 `requests.sessions.Session.request`（覆盖 `requests.get/post` 与所有
    `Session` 实例；项目里 14 个模块都走 requests）与 `urllib.request.urlopen`。
  * 违规**立即抛异常**（不返回空、不静默），因为"假装拿到了空数据"正是本项目
    反复踩的坑；调用方的 `except` 会把该维度标为不可评，这已是既有语义。
  * 记录**首次尝试的调用点**（模块:函数），让"谁在偷偷联网"可查——这是让它
    从"愿望"变成"可断言能力"的关键：测试断言 `blocked() == 0`。
  * 只在 `BIGA_OFFLINE` 为真时安装；正常运行时零开销（不 patch、不 import requests）。

用法：

    import data.offline_gate as gate
    gate.blocked()      # 被拦下的外呼次数（离线自检/测试断言用它）
    gate.stats()        # {"offline": True, "installed": True, "blocked": n, "samples": [...]}
"""

from __future__ import annotations

import os
import sys
import logging

_logger = logging.getLogger(__name__)

# 采样上限：只留最近 N 条（离线自检要报告"谁在联网"，但不需要无限增长）
_MAX_SAMPLES = 50


class OfflineViolation(RuntimeError):
    """离线模式下试图发起网络请求。

    故意继承 RuntimeError 而不是 BaseException：调用方既有的
    `except Exception` 降级路径仍然生效（把该维度标不可评），
    但 `except Exception: pass` 那种"静默吞掉"会因为计数不为 0 而被测试抓到。
    """


_installed: bool = False
_blocked: int = 0
_samples: list[dict] = []
_originals: dict = {}
_socket_hits: int = 0
_socket_samples: list[dict] = []


def _count_socket(sock, address) -> None:
    """裸 socket 连接**只记账**（AF_INET/AF_INET6）。"""
    global _socket_hits
    try:
        import socket as _s
        if getattr(sock, "family", None) not in (_s.AF_INET, _s.AF_INET6):
            return
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[offline] socket 族判定失败（不计入）: %s", e)
        return
    _socket_hits += 1
    if len(_socket_samples) < _MAX_SAMPLES:
        _socket_samples.append({"kind": "socket.connect", "url": str(address),
                                "at": _caller()})
    _logger.warning("[offline] 裸 socket 外呼（未走 requests，未被拦截）：%s ← %s",
                    address, _caller())


def is_offline() -> bool:
    """`BIGA_OFFLINE` ∈ {1,true,yes,on} ⇒ 离线（只读本地）。"""
    return os.environ.get("BIGA_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")


def _caller(max_depth: int = 3) -> str:
    """最近的**业务栈帧链**（`module:function:line`，最多 max_depth 层）。

    跳过本模块与 HTTP 库自身（`requests.*` / `urllib*` / `http.*` / `ssl`）——
    否则报告的永远是 `requests.api:request:71`，等于没说"谁在联网"。
    保留**链**而不是单帧：本仓里有 shim（如 `data.akshare_helper._patched_get`
    给 requests 打补丁），只看最内层会停在 shim 上，看不到真正的调用方。
    """
    here = os.path.basename(__file__)
    skip_roots = ("requests", "urllib", "http", "urllib3", "socket", "ssl")
    labels: list[str] = []
    frame = sys._getframe(1)
    while frame is not None and len(labels) < max_depth:
        fname = os.path.basename(frame.f_code.co_filename)
        mod = str(frame.f_globals.get("__name__", "?"))
        if fname != here and mod.split(".")[0] not in skip_roots:
            labels.append(f"{mod}:{frame.f_code.co_name}:{frame.f_lineno}")
        frame = frame.f_back
    return " ← ".join(labels) or "?"


def _record(url: str, kind: str) -> None:
    global _blocked
    _blocked += 1
    if len(_samples) < _MAX_SAMPLES:
        _samples.append({"kind": kind, "url": str(url)[:200], "at": _caller()})
    _logger.warning("[offline] 已拦截外呼（%s）：%s ← %s", kind, str(url)[:120],
                    _samples[-1]["at"] if _samples else "?")
    raise OfflineViolation(
        f"BIGA_OFFLINE=1 禁止外呼：{kind} {str(url)[:200]}"
    )


def install() -> bool:
    """安装拦截器（幂等）。返回是否**本次**完成安装（至少一处成功才算）。"""
    global _installed
    if _installed:
        return False
    ok = 0
    try:
        import requests.sessions as _rs
        _originals["requests"] = _rs.Session.request

        def _patched_request(self, method, url, *args, **kwargs):   # noqa: ANN001
            _record(url, f"requests.{str(method).upper()}")

        _rs.Session.request = _patched_request
        ok += 1
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[offline] requests 拦截器安装失败（仍会拦 urllib）: %s", e)

    try:
        import urllib.request as _ur
        _originals["urllib"] = _ur.urlopen

        def _patched_urlopen(url, *args, **kwargs):                 # noqa: ANN001
            target = url if isinstance(url, str) else getattr(url, "full_url", url)
            _record(target, "urllib.urlopen")

        _ur.urlopen = _patched_urlopen
        ok += 1
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[offline] urllib 拦截器安装失败: %s", e)

    # 裸 socket **只计数不阻断**（R4）：`baostock` 这类库直接走 TCP，requests 层拦不到。
    # 不阻断是因为本项目在 Windows 上还有本地 IPC（multiprocessing/streamlit）走 AF_INET，
    # 一刀切会把它们弄挂；但"有没有绕过闸门"必须可被测试与审计看见。
    try:
        import socket as _sock
        _originals["socket"] = _sock.socket.connect

        def _counting_connect(self, address, *args, **kwargs):      # noqa: ANN001
            _count_socket(self, address)
            return _originals["socket"](self, address, *args, **kwargs)

        _sock.socket.connect = _counting_connect
        ok += 1
    except Exception as e:                                  # noqa: BLE001
        _logger.warning("[offline] socket 计数器安装失败: %s", e)

    _installed = ok > 0          # ⚠️ R8：全都失败就不能说"已启用"
    if _installed:
        _logger.info("[offline] 离线闸门已启用（%d 个拦截点）：requests/urllib 会抛 "
                     "OfflineViolation；裸 socket 只计数", ok)
    else:
        _logger.error("[offline] 离线闸门**未能启用**——没有装上任何拦截点")
    return _installed


def uninstall() -> None:
    """还原（测试用；生产不会调用）。"""
    global _installed
    if not _installed:
        return
    try:
        import requests.sessions as _rs
        if "requests" in _originals:
            _rs.Session.request = _originals["requests"]
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[offline] requests 还原失败: %s", e)
    try:
        import urllib.request as _ur
        if "urllib" in _originals:
            _ur.urlopen = _originals["urllib"]
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[offline] urllib 还原失败: %s", e)
    try:
        import socket as _sock
        if "socket" in _originals:
            _sock.socket.connect = _originals["socket"]
    except Exception as e:                                  # noqa: BLE001
        _logger.debug("[offline] socket 计数器还原失败: %s", e)
    _originals.clear()
    _installed = False


def ensure() -> bool:
    """若为离线模式则确保已安装（幂等；在线时是 no-op）。"""
    if not is_offline():
        return False
    install()
    return True


def blocked() -> int:
    """被拦下的外呼次数（**离线自检的唯一判据**）。"""
    return _blocked


def socket_attempts() -> int:
    """**绕过 requests 层**的裸 TCP 连接次数（R4：如 baostock）。

    ⚠️ 这部分**没有拦截**（Windows 上本地 IPC 也走 AF_INET，一刀切会弄挂
    multiprocessing/streamlit）——它只是让"闸门有没有被绕过"可被审计。
    离线自检的判据应是 `blocked()==0 and socket_attempts()==0`。
    """
    return _socket_hits


def reset() -> None:
    """清零计数与采样（测试用）。不改变安装状态。"""
    global _blocked, _socket_hits
    _blocked = 0
    _socket_hits = 0
    _samples.clear()
    _socket_samples.clear()


def stats() -> dict:
    return {"offline": is_offline(), "installed": _installed,
            "blocked": _blocked, "socket_attempts": _socket_hits,
            "samples": list(_samples), "socket_samples": list(_socket_samples)}


# 导入即武装：`data/__init__.py` 引入本模块 ⇒ 任何 `data.*` 使用都已被拦截。
# （在线时 is_offline() 为假，这里什么都不做。）
if is_offline():
    install()
