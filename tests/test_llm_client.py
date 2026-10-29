"""测试 llm/deepseek_client.py 的失败语义（v2/M0 0.3.4）

背景：原 call_api 把异常吞成 "API调用失败: ..." 字符串返回，造成三重后果：
  - Agent 节点把错误文本当作分析正文，继续喂给下游多空辩论；
  - state.errors 始终为空，调度日志显示"完成、errors=0"；
  - 配置页的"LLM 连通性测试"把失败显示为"连通正常"。
本文件锁死新语义：失败必须抛 LLMError。
"""

import pytest
from unittest.mock import MagicMock, patch

from llm.deepseek_client import DeepSeekClient, LLMError, reset_client_cache


@pytest.fixture(autouse=True)
def _isolated_db(tmp_path, monkeypatch):
    """call_api 会把每次调用写进 llm_calls 表（Phase 6.2），测试必须隔离数据库，
    否则会在真实的 data/portfolio.db 里留下假的调用记录。"""
    import db.models as models
    monkeypatch.setattr(models, "DB_PATH", str(tmp_path / "ui.db"))
    yield


@pytest.fixture
def client():
    with patch("llm.deepseek_client._get_client_for_model") as mock_get:
        c = DeepSeekClient()
        c._openai = mock_get.return_value
        yield c


def _fake_response(content="OK", reasoning=None):
    msg = MagicMock()
    msg.content = content
    msg.reasoning_content = reasoning
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


def test_call_api_returns_content(client):
    client._openai.chat.completions.create.return_value = _fake_response("多头的论点")
    assert client.call_api([{"role": "user", "content": "hi"}]) == "多头的论点"


def test_call_api_raises_on_exception(client):
    """回归：调用失败必须抛 LLMError，不得返回错误字符串"""
    client._openai.chat.completions.create.side_effect = RuntimeError("429 Too Many Requests")
    with pytest.raises(LLMError) as ei:
        client.call_api([{"role": "user", "content": "hi"}])
    assert "429" in str(ei.value)


def test_call_api_raises_on_empty_content(client):
    """空响应同样是失败，不能返回占位文本让下游当成报告"""
    client._openai.chat.completions.create.return_value = _fake_response("")
    with pytest.raises(LLMError):
        client.call_api([{"role": "user", "content": "hi"}])


def test_error_text_is_never_returned_as_content(client):
    """最关键的回归：任何情况下都不得让错误信息变成「分析正文」"""
    client._openai.chat.completions.create.side_effect = RuntimeError("boom")
    try:
        out = client.call_api([{"role": "user", "content": "hi"}])
    except LLMError:
        return
    pytest.fail(f"失败时应抛 LLMError，实际返回了 {out!r}")


def test_reasoning_content_is_prefixed(client):
    client._openai.chat.completions.create.return_value = _fake_response("结论", reasoning="思考")
    out = client.call_api([{"role": "user", "content": "hi"}])
    assert "【推理过程】" in out and "结论" in out


def test_reset_client_cache_is_callable():
    """在线改 Key 后需要能清空客户端缓存（否则沿用旧密钥直到重启）"""
    reset_client_cache()
