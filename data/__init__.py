"""数据源包。

⚠️ 唯一副作用：导入本包即**武装离线闸门**（`BIGA_OFFLINE=1` 时任何外呼抛
`OfflineViolation`）。放在包初始化里是刻意的——约定式离线（每个消费方自己
写 `if offline_mode()`）已被实测证明不可靠（2026-10-18 审查 D4）。
在线时该模块什么都不做（不 patch、不 import requests）。
"""

from . import offline_gate  # noqa: F401  （导入即按 BIGA_OFFLINE 决定是否安装拦截器）
