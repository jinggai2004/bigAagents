"""推送消息模板（可编辑）的测试（2026-10-23）

钉住的判定：
1. **模板语法**：`{{var}}` 替换、未知占位符**原样保留**（暴露拼写错误）、
   已知但缺值的变量渲染成 `—`（不留空）、`{% if var %}…{% endif %}` 条件块；
2. **不用 `str.format`**：模板里出现字面花括号（JSON 示例）不得炸
   —— 本项目在 LLM 提示词上已经踩过这个坑；
3. **坏配置退回默认**：库里读不到 / 正文为空 ⇒ 用内置默认，不抛异常；
4. **默认模板与改造前逐字一致**（否则用户会发现"没改配置，消息却变了"）；
5. **改完即时生效**：保存后 `get_template` 立刻返回新内容（不重启进程）。
"""

import pytest

from core import push_templates as T
from tests.conftest import use_template_db


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path, monkeypatch):
    use_template_db(tmp_path, monkeypatch, "push_templates.db")


class TestRender:
    def test_replaces_known_variable(self):
        assert T.render("你好 {{name}}", {"name": "小王"}) == "你好 小王"

    def test_unknown_placeholder_is_kept_verbatim(self):
        """不认识的占位符**原样保留** —— 让用户一眼看出自己拼错了变量名。

        ⚠️ 只有在**给了合法变量清单**（`known`）时才能区分"拼错"与"缺值"；
        不给清单时一律当已知（缺值 → `—`）。
        """
        out = T.render("{{not_exist}}", {"name": "x"}, known={"name"})
        assert out == "{{not_exist}}"

    def test_known_but_missing_value_becomes_dash(self):
        assert T.render("值={{x}}", {}, known={"x"}) == "值=—"
        # 不给清单时也按"已知"处理 ⇒ 缺值 → "—"
        assert T.render("值={{x}}", {}) == "值=—"

    def test_empty_value_becomes_dash(self):
        assert T.render("值={{x}}", {"x": None}) == "值=—"
        assert T.render("值={{x}}", {"x": ""}) == "值=—"

    def test_literal_braces_do_not_crash(self):
        """⚠️ 不用 str.format 的理由：模板里出现字面花括号不得炸。"""
        out = T.render('JSON 示例：{"a": 1} {{x}}', {"x": "ok"})
        assert '{"a": 1}' in out and "ok" in out

    def test_conditional_block_removed_when_empty(self):
        tpl = "前{% if rel %}　【{{rel}}】{% endif %}后"
        assert T.render(tpl, {"rel": "持仓"}) == "前　【持仓】后"
        assert T.render(tpl, {"rel": ""}) == "前后"

    def test_conditional_block_inside_item_template(self):
        tpl = T.DEFAULT_TEMPLATES[T.TPL_ALERT_ITEM]
        out = T.render(tpl, {"icon": "🔴", "name": "茅台", "code": "600519",
                             "message": "涨停", "price": "1329.04",
                             "quote_time": "14:30:00", "relation": "持仓", "note": "有备注"})
        assert "【持仓】" in out and "└ 有备注" in out
        out2 = T.render(tpl, {"icon": "🔴", "name": "茅台", "code": "600519",
                              "message": "涨停", "price": "1329.04",
                              "quote_time": "14:30:00", "relation": "", "note": ""})
        assert "【" not in out2 and "└" not in out2


class TestStorage:
    def test_default_used_when_no_override(self):
        assert T.get_template(T.TPL_ALERT) == T.DEFAULT_TEMPLATES[T.TPL_ALERT]

    def test_save_takes_effect_immediately(self):
        assert T.save_template(T.TPL_ALERT, "# 自定义 {{count}}")
        assert T.get_template(T.TPL_ALERT) == "# 自定义 {{count}}"
        # 渲染也立刻生效（不依赖重启）
        assert T.render(T.get_template(T.TPL_ALERT), {"count": "3"}) == "# 自定义 3"

    def test_reset_restores_default(self):
        T.save_template(T.TPL_ALERT, "自定义")
        assert T.reset_template(T.TPL_ALERT) is True
        assert T.get_template(T.TPL_ALERT) == T.DEFAULT_TEMPLATES[T.TPL_ALERT]

    def test_empty_body_is_treated_as_reset(self):
        T.save_template(T.TPL_ALERT, "x")
        T.save_template(T.TPL_ALERT, "   ")
        assert T.get_template(T.TPL_ALERT) == T.DEFAULT_TEMPLATES[T.TPL_ALERT]

    def test_unknown_key_is_rejected(self):
        assert T.save_template("not_a_template", "x") is False

    def test_broken_storage_falls_back_to_default(self, monkeypatch):
        """⚠️ 配置坏了不能连累推送（坏配置退回默认）。"""
        from db import store

        def _boom():
            raise RuntimeError("db down")

        monkeypatch.setattr(store, "list_push_templates", _boom)
        assert T.get_template(T.TPL_ALERT) == T.DEFAULT_TEMPLATES[T.TPL_ALERT]
        assert T.user_overrides() == {}


class TestDefaultMatchesLegacyOutput:
    """⭐ 默认模板的渲染结果必须与**改造前**逐字一致。"""

    def test_alert_message_unchanged(self):
        from core.alert_engine import format_alerts_markdown
        alerts = [
            {"code": "600519", "name": "贵州茅台", "rule": "limit_up", "level": "danger",
             "message": "涨停 +10.00%", "price": 1329.04, "change_pct": 10.0,
             "quote_time": "14:30:00"},
            {"code": "300750", "name": "宁德时代", "rule": "price_surge", "level": "warn",
             "message": "涨幅 +7.80%", "price": 236.96, "change_pct": 7.8,
             "quote_time": "14:30:00"},
        ]
        out = format_alerts_markdown(alerts, title="bigAagents 盘中异动 14:30",
                                     relations={"600519": "持仓·自选"},
                                     notes={"600519": "持仓 100 股，止损 1245.00（现价差 -6.3%）"})
        expected = (
            "# bigAagents 盘中异动 14:30（2 条）\n\n"
            "- 🔴 **贵州茅台(600519)** 涨停 +10.00% · 现价 1329.04 · 14:30:00　【持仓·自选】\n"
            "　　└ 持仓 100 股，止损 1245.00（现价差 -6.3%）\n"
            "- 🟠 **宁德时代(300750)** 涨幅 +7.80% · 现价 236.96 · 14:30:00\n\n"
            "*程序化监控，仅供参考，不构成投资建议。*"
        )
        assert out == expected

    def test_sample_contexts_render_without_error(self):
        """配置页"预览"按钮用的示例上下文必须都能渲染（否则预览是坏的）。"""
        for key in T.TEMPLATE_META:
            ctx = T.SAMPLE_CONTEXT.get(key) or T.SAMPLE_ITEM_CONTEXT.get(key) or {}
            out = T.render(T.DEFAULT_TEMPLATES[key], ctx)
            assert out.strip(), f"{key} 的预览是空的"
            assert "{{" not in out, f"{key} 的示例上下文有变量没填：{out}"
