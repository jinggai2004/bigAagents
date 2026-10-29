"""工具契约测试：防止「Mock 掩盖真实接口」这一类静默故障。

## 为什么单独建这个文件

项目的血泪史里有一整类 bug 是同一个形状：**测试把被测对象整个 mock 掉，
于是"名字写错 / 参数对不上 / 方法不存在"全都测不出来，而真实运行必然是坏的。**

已经踩过的两次：

1. `data/news.py::NewsFlowDataFetcher` 根本没有 `get_news_flow_data()` 方法，
   `core/agents.py` 却一直在调它 → 舆情主源永久 AttributeError → 实际 0 条新闻，
   而模型照常输出"交叉核查沪深交易所/巨潮资讯网"。测试用
   `@patch("data.news.NewsFlowDataFetcher")` 后写
   `mock.return_value.get_news_flow_data.return_value = ...` ——
   **Mock 上任何属性都成立**，方法名写错也"通过"。
   （契约断言现已在 `tests/test_news_stock_source.py`）

2. 2026-09-28：`core/agent_tools.py` 的 `_tool` 包装层签名是 `def wrapper(**kwargs)`，
   只收关键字参数；`core/agents.py` 的基本面节点写的是
   `tool_financials(stock_code, periods=4)` → 位置参数抛 TypeError，
   被工具自己的 except 兜成 `{"error": ...}` → 基本面**永远走"未评估"分支**。
   集成测试当时 patch 的是 `core.agent_tools.tool_financials` **本身**，
   真实函数体从未执行，所以全绿。

## 本文件的原则

**只 mock 叶子依赖（最底层的取数函数），不 mock 被测工具自身。**
这样 `_tool` 包装层 + 工具函数体 + 参数传递路径全部真实执行，
签名一旦对不上就立刻红。
"""

import inspect
from unittest.mock import patch

import pytest

from core import agent_tools
from core.agent_tools import TOOL_FUNCS, TOOL_SCHEMAS, _tool, execute_tool


# ===== 1. 包装层：必须接受位置参数与关键字参数 =====

def test_tool_wrapper_forwards_positional_args():
    """`_tool` 包装层必须能转发位置参数。

    失败即意味着 Python 侧调用方一旦用位置传参，就会拿到
    `TypeError: wrapper() takes 0 positional arguments but N were given`。
    该异常在参数绑定阶段抛出，wrapper 的 except 接不住，会直接抛给调用方
    → 调用方的 try/except 把它记成"取不到数据"，静默降级。
    """
    @_tool
    def _probe(a, b=2):
        return {"a": a, "b": b}

    res = _probe(1, 3)
    assert res["a"] == 1 and res["b"] == 3, f"位置参数未正确转发: {res}"


def test_tool_wrapper_forwards_keyword_args_and_adds_as_of():
    @_tool
    def _probe(a, b=2):
        return {"a": a, "b": b}

    res = _probe(a=1, b=3)
    assert res["a"] == 1 and res["b"] == 3
    assert res["as_of"], "工具结果必须带 as_of 时间戳"


def test_tool_wrapper_swallows_exception_into_error_dict():
    """包装层的兜底语义不能变：异常 → {"error": ...}，绝不抛到 Agent 循环。"""
    @_tool
    def _boom():
        raise ValueError("炸了")

    res = _boom()
    assert "error" in res and "炸了" in res["error"]


def test_tool_wrapper_preserves_original_signature():
    """必须用 functools.wraps 保留原签名（`inspect.signature` 会沿 __wrapped__ 还原）。

    这既是可读性，也是下面第 3 组测试能自动发现"schema 声明了函数没有的参数"的前提。
    """
    sig = inspect.signature(agent_tools.tool_financials)
    assert list(sig.parameters) == ["symbol", "periods"], (
        f"包装后签名被抹平了: {sig} —— 请确认 _tool 用了 functools.wraps")


# ===== 2. 注册表契约：schema 声明的参数，函数必须真的收得下 =====

def _schema_params(schema: dict) -> list:
    params = (schema.get("function", {}).get("parameters") or {})
    return list((params.get("properties") or {}).keys())


def test_every_schema_name_has_a_registered_function():
    names = [s["function"]["name"] for s in TOOL_SCHEMAS]
    missing = [n for n in names if n not in TOOL_FUNCS]
    assert not missing, f"schema 声明了但注册表里没有的工具: {missing}"


def test_every_registered_function_has_a_schema():
    """注册表里多出没有 schema 的函数 → LLM 永远不会调用它（等于死代码）。"""
    declared = {s["function"]["name"] for s in TOOL_SCHEMAS}
    extra = [n for n in TOOL_FUNCS if n not in declared]
    assert not extra, f"已注册但未声明 schema 的工具（模型无法调用）: {extra}"


def test_schema_params_are_accepted_by_the_function():
    """**schema ↔ 函数签名**的契约。

    schema 是给 LLM 看的参数说明，函数是真正执行的一方。
    两边只要有一边写错（多一个/少一个参数名），模型给的参数就会变成
    `TypeError: unexpected keyword argument`，被兜成 {"error": ...}
    → 该工具对模型而言"永远没有数据"，且不报错。
    """
    problems = []
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        fn = TOOL_FUNCS.get(name)
        if fn is None:
            continue
        accepted = set(inspect.signature(fn).parameters)
        declared = set(_schema_params(schema))
        unknown = declared - accepted
        if unknown:
            problems.append(f"{name}: schema 声明了函数不接受的参数 {sorted(unknown)}"
                            f"（函数签名: {sorted(accepted)}）")
    assert not problems, "\n".join(problems)


def test_required_schema_params_have_no_default_on_the_function():
    """schema 标为 required 的参数，函数里不应该有默认值。

    否则"模型忘了传"会被默认值静默兜住（如 symbol 默认成空串再报"代码须为6位数字"），
    错误信息指向错误的地方。
    """
    problems = []
    for schema in TOOL_SCHEMAS:
        fn = schema["function"]["name"]
        func = TOOL_FUNCS.get(fn)
        if func is None:
            continue
        params = inspect.signature(func).parameters
        required = (schema["function"]["parameters"] or {}).get("required") or []
        for pname in required:
            p = params.get(pname)
            if p is not None and p.default is not inspect.Parameter.empty:
                problems.append(f"{fn}.{pname} 被标为 required 但有默认值 {p.default!r}")
    assert not problems, "\n".join(problems)


# ===== 3. 执行路径：execute_tool → 真实函数体（不 mock 被测工具）=====

def test_execute_tool_unknown_name_returns_error():
    assert "error" in execute_tool("tool_does_not_exist", {})


def test_execute_tool_bad_json_returns_error():
    assert "error" in execute_tool("tool_stock_profile", "{不是json")


def test_execute_tool_rejects_unexpected_param_without_raising():
    """纵深防御：参数名对不上时返回 error 而不是抛异常。"""
    res = execute_tool("tool_stock_profile", {"nope": 1})
    assert "error" in res


# ===== 4. tool_financials 的真实调用路径（叶子级 mock）=====

def test_tool_financials_positional_call_reaches_real_body():
    """**位置传参必须抵达真实函数体**——这是 2026-09-28 那个 bug 的回归测试。

    做法：只 mock 最底层的取数函数 `StockDataFetcher.get_financial_data`，
    让它返回一个**带标记的错误**。如果调用真的到了 `tool_financials` 函数体，
    这个标记会被原样包进 `财务数据不可用: ...` 返回；如果调用死在包装层的签名检查上，
    则在**调用处**直接抛 `TypeError: wrapper() takes 0 positional arguments` ——
    注意参数绑定失败发生在 wrapper 函数体之前，**wrapper 自己的 except 接不住**，
    异常会直接冒出来（旧实现下本用例是 ERROR 而不是断言失败）。
    """
    marker = "CONTRACT-MARKER-叶子错误"
    with patch("data.stock_data.StockDataFetcher.get_financial_data",
               return_value={"error": marker}):
        res = agent_tools.tool_financials("600211", periods=4)      # 位置传参

    assert "error" in res
    assert marker in res["error"], (
        f"位置传参没有抵达真实函数体，returned={res!r} "
        "（典型症状：TypeError: wrapper() takes 0 positional arguments）")


def test_tool_financials_keyword_call_reaches_real_body():
    marker = "CONTRACT-MARKER-叶子错误"
    with patch("data.stock_data.StockDataFetcher.get_financial_data",
               return_value={"error": marker}):
        res = agent_tools.tool_financials(symbol="600211", periods=4)

    assert marker in res.get("error", "")


def test_tool_financials_produces_report_without_mocking_the_tool():
    """端到端（除叶子外全真）：给出真实形状的三表 → 必须产出可读财务摘要。

    这条用例保证 `tool_financials` 的**函数体**真的能跑通，
    而不是只有"被 mock 时才通过"。
    """
    raw = {
        "symbol": "600211",
        "income_statement": [
            {"报告日": "20260630", "营业收入": 1.2e9, "净利润": 2.0e8, "_source": "新浪"},
            {"报告日": "20250630", "营业收入": 1.0e9, "净利润": 1.7e8, "_source": "新浪"},
        ],
        "balance_sheet": [
            {"报告日": "20260630", "资产总计": 5.0e9, "负债合计": 1.5e9},
            {"报告日": "20250630", "资产总计": 4.6e9, "负债合计": 1.4e9},
        ],
        "cash_flow": [{"报告日": "20260630", "经营活动产生的现金流量净额": 3.0e8}],
    }
    with patch("data.stock_data.StockDataFetcher.get_financial_data", return_value=raw):
        res = agent_tools.tool_financials("600211", periods=4)

    assert "error" not in res, f"真实函数体执行失败: {res}"
    assert res["报告期"], "应返回报告期列表"
    assert res["利润表(亿元)"]["营业收入"] == [12.0, 10.0], res.get("利润表(亿元)")
    # 同比 = 2026H1 vs 2025H1 → +20%
    assert res["同比%"]["营业收入"] == 20.0


# ===== 5. 基本面节点必须真的把财务数据喂进去 =====

@pytest.fixture()
def _leaf_financials():
    raw = {
        "symbol": "600211",
        "income_statement": [
            {"报告日": "20260630", "营业收入": 1.2e9, "净利润": 2.0e8, "_source": "新浪"},
            {"报告日": "20250630", "营业收入": 1.0e9, "净利润": 1.7e8, "_source": "新浪"},
        ],
        "balance_sheet": [{"报告日": "20260630", "资产总计": 5.0e9}],
        "cash_flow": [],
    }
    with patch("data.stock_data.StockDataFetcher.get_financial_data", return_value=raw):
        yield


def test_fundamental_node_marks_financials_available(_leaf_financials):
    """基本面节点在**只 mock 叶子**的情况下，必须判定"财务报表已取到"。

    上一版这条路径恒为「未评估」，因为节点是用位置参数调工具、
    而包装层只收关键字参数 —— 但集成测试把工具整个 mock 掉了，看不见。
    """
    from core.agents import fundamental_analyst_node

    captured = {}

    class _FakeClient:
        def fundamental_analysis(self, *a, **kw):
            captured.update(kw)
            return "**基本面报告**\n（测试替身）"

    with patch("core.agents.DeepSeekClient", return_value=_FakeClient()):
        out = fundamental_analyst_node({
            "stock_code": "600211", "stock_info": {"name": "西藏药业", "pe_ratio": 12.0},
            "analysis_date": "2026-09-28", "data_through": "2026-09-24",
            "data_availability": {}, "errors": [], "fatal_errors": [],
        })

    assert out["data_availability"]["财务报表"] is True, (
        f"财务报表未被判定为已取到: {out.get('errors')}")
    fin = captured.get("financial_data") or {}
    assert fin.get("报告期"), f"财务数据没有传给 LLM: {captured.keys()}"
    assert captured.get("analysis_date") == "2026-09-28", "时间锚点没传下去"
    assert captured.get("data_through") == "2026-09-24"


def test_fundamental_node_falls_back_only_when_leaf_has_no_data():
    """叶子确实取不到时，才允许走"未评估"固定文案，且必须写明原因。"""
    from core.agents import fundamental_analyst_node

    with patch("data.stock_data.StockDataFetcher.get_financial_data",
               return_value={"error": "源不可用"}):
        out = fundamental_analyst_node({
            "stock_code": "600211", "stock_info": {"name": "西藏药业"},
            "data_availability": {}, "errors": [], "fatal_errors": [],
        })

    assert out["data_availability"]["财务报表"] is False
    assert "未评估" in out["fundamental_report"]
    assert "源不可用" in " ".join(out.get("errors") or [])
