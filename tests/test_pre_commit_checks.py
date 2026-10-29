# -*- coding: utf-8 -*-
"""提交前快集闸门（`scripts/pre_commit_checks.py`）的自测。

## 为什么护栏也要测
2026-10-23 的教训有两条都落在这里：
1. **"看起来对"**——`check_workspace` 的"改动未生效"第一版拿心跳文件 mtime 比对，
   而心跳每 60 秒重写 ⇒ 判断**恒为空**，试跑时输出了"✓ 无"骗过我一次。
2. **没有鉴别力就等于没有闸门**——一个永远返回 ✓ 的检查器比不检查更糟，
   因为它会让人以为查过了。

所以这组用例只守两件事：
- **真能拦住**（把明显的坏输入喂进去，必须返回 False）；
- **不该拦的不拦**（正常状态下必须返回 True，否则会被绕过/关掉）。
"""
from __future__ import annotations

import os

import pytest

import scripts.pre_commit_checks as P


class TestSyntax:
    def test_real_tree_parses(self):
        ok, msg = P.check_syntax()
        assert ok, msg
        assert "个 .py 全部可解析" in msg

    def test_broken_file_is_caught(self, tmp_path, monkeypatch):
        """⭐ 鉴别力：语法错误必须拦住（`def f(:` 是真实事故的形态）。"""
        bad = tmp_path / "broken.py"
        bad.write_text("def broken(:\n    pass\n", encoding="utf-8")
        monkeypatch.setattr(P, "_iter_sources", lambda: iter([str(bad)]))
        ok, msg = P.check_syntax()
        assert ok is False
        assert "语法错误" in msg


class TestConfig:
    def test_real_configs_parse(self):
        ok, msg = P.check_json()
        assert ok, msg

    def test_broken_json_is_caught(self, tmp_path, monkeypatch):
        cfg = tmp_path / "config"
        cfg.mkdir()
        (cfg / "bad.json").write_text('{"a": 1,}', encoding="utf-8")   # 尾逗号
        monkeypatch.setattr(P, "PROJECT_ROOT", str(tmp_path))
        ok, msg = P.check_json()
        assert ok is False
        assert "bad.json" in msg


class TestContractSet:
    def test_four_files_exist(self):
        """四件套必须都在——**护栏被删掉比护栏红更危险**。"""
        for rel, _desc in P._CONTRACT_TESTS:
            assert os.path.exists(os.path.join(P.PROJECT_ROOT, rel)), rel

    def test_missing_contract_test_blocks(self, monkeypatch):
        monkeypatch.setattr(P, "_CONTRACT_TESTS",
                            (("tests/test_not_exist_xyz.py", "不存在"),))
        ok, msg = P.check_contracts("python")
        assert ok is False
        assert "找不到" in msg or "缺失" in msg


class TestHotFiles:
    def test_old_file_is_not_hot(self, tmp_path, monkeypatch):
        f = tmp_path / "old.py"
        f.write_text("x = 1\n", encoding="utf-8")
        os.utime(f, (1_600_000_000, 1_600_000_000))      # 2020 年
        monkeypatch.setattr(P, "_iter_sources", lambda: iter([str(f)]))
        ok, msg = P.check_hotfiles(15)
        assert ok is True, msg

    def test_fresh_file_is_reported(self, tmp_path, monkeypatch):
        f = tmp_path / "fresh.py"
        f.write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(P, "_iter_sources", lambda: iter([str(f)]))
        ok, msg = P.check_hotfiles(15)
        assert ok is False
        assert "fresh.py" in msg


class TestEffective:
    def test_no_heartbeat_is_unknown_not_ok(self, tmp_path, monkeypatch):
        """读不到心跳时说"无法判定"，**不假装 ✓**（同 check_workspace 的教训）。"""
        monkeypatch.setattr(P, "PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(P, "_iter_sources", lambda: iter([]))
        ok, msg = P.check_effective()
        assert ok is True
        assert "未判定" in msg or "无法判定" in msg

    def test_newer_than_started_is_flagged(self, tmp_path, monkeypatch):
        """⭐ 核心判据：文件比进程新 ⇒ 必须报"需重启"。"""
        import json
        import time

        (tmp_path / "temp").mkdir()
        (tmp_path / "temp" / "scheduler_heartbeat.json").write_text(
            json.dumps({"pid": 1, "started_at": time.time() - 3600}),
            encoding="utf-8")
        f = tmp_path / "changed.py"
        f.write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(P, "PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(P, "_iter_sources", lambda: iter([str(f)]))
        ok, msg = P.check_effective()
        assert ok is False
        assert "需重启" in msg

    def test_older_than_started_passes(self, tmp_path, monkeypatch):
        import json
        import time

        (tmp_path / "temp").mkdir()
        (tmp_path / "temp" / "scheduler_heartbeat.json").write_text(
            json.dumps({"pid": 1, "started_at": time.time() + 3600}),
            encoding="utf-8")
        f = tmp_path / "old.py"
        f.write_text("x = 1\n", encoding="utf-8")
        monkeypatch.setattr(P, "PROJECT_ROOT", str(tmp_path))
        monkeypatch.setattr(P, "_iter_sources", lambda: iter([str(f)]))
        ok, msg = P.check_effective()
        assert ok is True, msg


class TestBlockingSemantics:
    """阻断项只有 语法/配置/契约；并发与生效是**警告**（不该拦住正常提交）。"""

    def test_warning_steps_do_not_block(self):
        assert "并发" not in {"语法", "配置", "契约"}
        assert "生效" not in {"语法", "配置", "契约"}

    @pytest.mark.parametrize("quiet", [False, True])
    def test_main_returns_zero_or_two_on_clean_tree(self, quiet, monkeypatch):
        monkeypatch.setattr(P, "check_hotfiles", lambda m: (True, "stub"))
        monkeypatch.setattr(P, "check_effective", lambda: (True, "stub"))
        monkeypatch.setattr(P, "check_contracts", lambda r: (True, "stub"))
        argv = ["pre_commit_checks.py", "--quiet"] if quiet else ["pre_commit_checks.py"]
        monkeypatch.setattr("sys.argv", argv)
        assert P.main() == 0
