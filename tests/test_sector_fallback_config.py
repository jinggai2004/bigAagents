# -*- coding: utf-8 -*-
"""选股兜底池**配置外置**的守护（2026-10-17，P2-3）。

## 为什么外置

`SECTOR_HARDCODED` 是"本地快照也不可用时"的最后兜底池 —— 它是**会随行情过时的运维参数**
（换股票、调板块不该改代码、不该跑全量测试）。现改为读 `config/sector_fallback.json`。

## 两条必须钉住的语义

1. **配置坏了不许变成空池**：那会让降级路径返回 0 只候选，
   比"用内置池"更糟 —— 而且页面会显示成"选股无结果"，把运维事故伪装成行情结论；
2. **键必须与 `nl_screener.SECTORS` 一致**：两处各自演化过一次，用断言防再漂移。
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)


class TestSectorFallbackConfig:
    def test_repo_config_loads_and_is_non_empty(self):
        """仓库里的配置必须能读出来且非空（外置后最常见的翻车方式就是路径写错）。"""
        from core import screener as S
        pools = S._load_sector_fallback()
        assert pools, "配置读出来是空的 ⇒ 降级路径会返回 0 只候选"
        assert all(isinstance(v, list) and v for v in pools.values())
        assert pools == S.SECTOR_HARDCODED, "模块级缓存与实时读取不一致"

    def test_keys_match_nl_screener(self):
        """⭐ 防漂移：兜底池的板块键必须覆盖 `nl_screener.SECTORS`。"""
        from core import nl_screener, screener as S
        missing = set(nl_screener.SECTORS) - set(S.SECTOR_HARDCODED)
        assert not missing, f"这些板块在兜底池里没有对应池子：{missing}"

    def test_broken_config_falls_back_to_builtin_not_empty(self, tmp_path):
        """⭐⭐ 核心：配置损坏 ⇒ 退回**内置默认**，**绝不是空池**。"""
        from core import screener as S
        bad = tmp_path / "broken.json"
        bad.write_text('{ not json', encoding="utf-8")
        got = S._load_sector_fallback(str(bad))
        assert got == S._SECTOR_FALLBACK_BUILTIN, "损坏的配置没有退回内置默认"
        assert got, "退回结果不能是空池"

    def test_missing_file_falls_back_to_builtin(self, tmp_path):
        from core import screener as S
        got = S._load_sector_fallback(str(tmp_path / "nope.json"))
        assert got == S._SECTOR_FALLBACK_BUILTIN

    def test_empty_sectors_falls_back(self, tmp_path):
        """配置合法但**内容为空**同样要退回（"空"是最危险的"看起来正常"）。"""
        from core import screener as S
        f = tmp_path / "empty.json"
        f.write_text(json.dumps({"sectors": {}}, ensure_ascii=False), encoding="utf-8")
        assert S._load_sector_fallback(str(f)) == S._SECTOR_FALLBACK_BUILTIN

    def test_valid_config_overrides_builtin(self, tmp_path):
        """⚠️ 鉴别力：合法配置必须**真的生效**（否则"外置"等于摆设）。"""
        from core import screener as S
        f = tmp_path / "ok.json"
        f.write_text(json.dumps({"sectors": {"测试板块": ["600000", "000001"]}},
                                ensure_ascii=False), encoding="utf-8")
        got = S._load_sector_fallback(str(f))
        assert got == {"测试板块": ["600000", "000001"]}, got
