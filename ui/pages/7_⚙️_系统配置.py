"""系统配置页面 — API Key 管理 / 连通性测试 / 调度状态 / 参数总览"""

import sys, os, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
LATEST_PATH = os.path.join(PROJECT_ROOT, "output", "daily_picks", "latest.json")

st.set_page_config(page_title="系统配置", page_icon="⚙️", layout="wide")
st.title("⚙️ 系统配置")

# 环境文件读写（统一 UTF-8；读取兼容 GBK 旧文件）
def _read_env_text() -> str:
    if not os.path.exists(ENV_PATH):
        return ""
    raw = open(ENV_PATH, "rb").read()
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def _write_env_text(text: str):
    with open(ENV_PATH, "w", encoding="utf-8") as f:
        f.write(text)


def read_env_vars() -> dict:
    vars_ = {}
    for line in _read_env_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            vars_[k.strip()] = v.strip()
    return vars_


def update_env_vars(updates: dict):
    """更新指定变量，保留其余行（含注释），末尾补缺失项"""
    lines = _read_env_text().splitlines()
    out, seen = [], set()
    for line in lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.strip().startswith("#") else None
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)
    for k, v in updates.items():
        if k not in seen:
            out.append(f"{k}={v}")
    _write_env_text("\n".join(out) + "\n")


def mask(value: str) -> str:
    if not value:
        return "未配置"
    return f"已配置 (…{value[-4:]})" if len(value) > 4 else "已配置"


def flash(msg: str, level: str = "success"):
    """保存后要 st.rerun() 刷新表格，而 rerun 会把这一轮的消息冲掉，
    所以先把提示存进 session_state，由下一轮渲染时弹出来。"""
    st.session_state["_cfg_flash"] = (level, msg)


def render_flash():
    item = st.session_state.pop("_cfg_flash", None)
    if item:
        getattr(st, item[0], st.info)(item[1])


def split_list(text: str) -> list:
    """逗号（中英文均可）分隔 → 去空去重保序的列表。"""
    out, seen = [], set()
    for part in (text or "").replace("，", ",").split(","):
        v = part.strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


env = read_env_vars()
render_flash()

# ===== 配置状态总览 =====

from ui.components.section_nav import section_nav, visible

# ⚠️ `_SEP` / `_prov` 被**多个区块**使用 ⇒ 必须留在模块级；
#    否则「只选中某一区块」时它们不存在 ⇒ NameError（点下去才炸）。
SEC_API = "🔑 API 配置状态"
SEC_PROV = "🧩 模型接入（多厂商）"
SEC_FEATURE = "🎚️ 功能模型分配"
SEC_KEYS = "✏️ 更新密钥"
SEC_TPL = "📨 推送消息模板"
SEC_LLM = "🧪 LLM 连通性测试"
SEC_HEALTH = "🩺 运行健康"
SEC_SCHED = "⏰ 定时调度"
SEC_JOBS = "⏱️ 自定义定时任务（改完不用重启）"
SEC_CALLS = "🧠 LLM 调用链（近 7 天）"
SEC_DATA = "📚 数据链路与参数"
_SECTIONS = (SEC_API, SEC_PROV, SEC_FEATURE, SEC_KEYS, SEC_TPL, SEC_LLM, SEC_HEALTH,
             SEC_SCHED, SEC_JOBS, SEC_CALLS, SEC_DATA)

_SEP = " · "
_prov = None
try:
    from config import providers as _prov
except Exception as _prov_err:                           # noqa: BLE001
    st.error(f"厂商注册表模块不可用：{_prov_err}")


@st.cache_data(ttl=1800, show_spinner=False)
def panel_overview() -> dict:
    """本地面板概况（P0-3，2026-10-18）——读落盘概况，**不现算、不联网**。

    为什么要它：`data_coverage` 之前只有 1 行陈旧记录（got=7/total=12），
    健康页看不出"历史行情底库到底覆盖了多少"。
    ⚠️ 不能调 `panel_status()`：它内部 `load_panel()` 要全量 dedup+sort，
    11.3M 行实测**分钟级**（页 7 冒烟因此 AppTest 超时 120s）。
    改为读 `data/panel/overview.json`（回填收尾生成），毫秒级。
    ⚠️ TTL=30 分钟：概况每天回填后变一次；TTL 只是防同一进程长期不刷新。
    """
    try:
        import data.panel_store as _ps
        ov = _ps.read_overview()
    except Exception as e:                               # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if not ov:
        return {"error": "尚无面板概况（运行一次回填或 "
                         "`python -c \"import data.panel_store as p; p.write_overview()\"`）"}
    return ov


def main():
    _SEL = section_nav(_SECTIONS, key="cfg_section",
                        group_label="⚙️ 系统配置 · 板块")
    if visible(_SEL, SEC_API):
        st.subheader("🔑 API 配置状态")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("DeepSeek Key", mask(env.get("DEEPSEEK_API_KEY", "")))
            st.metric("Qwen Key", mask(env.get("QWEN_API_KEY", "")))
        with c2:
            st.metric("Tushare Token", mask(env.get("TUSHARE_TOKEN", "")))
            st.metric("通知 Webhook", "已配置" if (env.get("DINGTALK_WEBHOOK") or env.get("FEISHU_WEBHOOK")) else "未配置")
        with c3:
            st.metric("默认模型", env.get("DEFAULT_MODEL_NAME") or env.get("DEEPSEEK_MODEL", "deepseek-chat（默认）"))
            st.caption("厂商路由由 `config/llm_providers.json` 决定，见下方「🧩 模型接入」")

        # ===== 模型接入（多厂商）=====
        # 改造前厂商路由硬编码在 llm/deepseek_client.py 的 if/else 里
        # （"qwen-" 前缀 → DashScope，其余 → DeepSeek），接第三家厂商必须改代码。
        # 现在清单外置成 JSON，本区块就是它的编辑器。
        # 密钥仍然只写在 .env —— 注册表里存的是「密钥的环境变量名」，不是密钥本身，
        # 这样清单文件可以安全地进版本库。
    if visible(_SEL, SEC_PROV):
        st.subheader("🧩 模型接入（多厂商）")


        if _prov is not None:
            _registry_path = _prov.registry_path()
            _providers = _prov.get_providers()
            _default_pid = _prov.default_provider_id()

            st.caption(
                f"清单文件：`{_registry_path}`　｜　密钥：`.env`。"
                "任意 **OpenAI 兼容端点**都能接入：Kimi / 智谱 GLM / 豆包 / OpenAI / "
                "本地 Ollama（`http://localhost:11434/v1`）/ 自建网关。"
            )

            if not _prov.registry_file_exists():
                st.info("尚未创建清单文件，当前使用**内置厂商**（DeepSeek / Qwen），"
                        "路由行为与改造前完全一致。要新增厂商请先初始化。")
                if st.button("📄 初始化清单文件"):
                    try:
                        _prov.init_registry_file()
                        flash(f"已创建 {_registry_path}")
                        st.rerun()
                    except Exception as e:                       # noqa: BLE001
                        st.error(f"创建失败：{e}")

            # 厂商总览。
            # 密钥列在**未配置时也要写出环境变量名**：只显示"未配置"的话，
            # 用户不知道该往 .env 里填哪个变量，等于把最后一步留成谜题。
            _rows = []
            for _p in _providers:
                _env_name = _p.get("api_key_env") or ""
                if _prov.provider_api_key(_p):
                    _key_cell = f"✅ {_env_name}" if _env_name else "✅ 已配置"
                else:
                    _key_cell = f"❌ {_env_name}（未配置）" if _env_name else "❌ 未配置"
                _rows.append({
                    "厂商": _p["name"] + ("　⭐默认" if _p["id"] == _default_pid else ""),
                    "ID": _p["id"],
                    "Base URL（实际生效）": _prov.provider_base_url(_p) or "—",
                    "密钥": _key_cell,
                    "已声明模型": "、".join(_p["models"]) or "—",
                    "匹配模式": "、".join(_p["match"]) or "—",
                    "状态": "启用" if _p.get("enabled", True) else "已停用",
                })
            st.dataframe(_rows, width="stretch", hide_index=True)

            # ── 默认模型 ──
            # 深度分析、AI 助手在未显式指定模型时用它（core/analysis_history 也按它记账）。
            _all_models = _prov.available_models()
            _labels = [f"{m['model']}{_SEP}{m['provider_name']}"
                       + ("" if m["ready"] else "（密钥未配置）") for m in _all_models]
            with st.form("llm_default_model_form"):
                _cur_default = env.get("DEFAULT_MODEL_NAME") or env.get("DEEPSEEK_MODEL", "")
                _dc1, _dc2 = st.columns(2)
                with _dc1:
                    _pick = st.selectbox("从已声明模型中选择", ["（保持不变）"] + _labels, index=0)
                with _dc2:
                    _manual = st.text_input("或手填模型名（优先级更高）",
                                            placeholder=f"当前：{_cur_default or 'deepseek-chat'}")
                if st.form_submit_button("💾 设为默认模型"):
                    _new_model = _manual.strip() or (
                        _pick.split(_SEP)[0] if _pick != "（保持不变）" else "")
                    if not _new_model:
                        st.warning("没有选择或填写任何模型")
                    else:
                        try:
                            from config.settings import reload_config
                            from llm.deepseek_client import reset_client_cache
                            update_env_vars({"DEFAULT_MODEL_NAME": _new_model})
                            reload_config()
                            reset_client_cache()
                            flash(f"默认模型已设为 {_new_model}"
                                  f"（厂商：{_prov.describe_model(_new_model)}）")
                            st.rerun()
                        except Exception as e:                   # noqa: BLE001
                            st.error(f"保存失败：{e}")

            # ── 添加厂商 ──
            with st.expander("➕ 添加厂商", expanded=False):
                with st.form("add_provider_form"):
                    _ac1, _ac2 = st.columns(2)
                    with _ac1:
                        _f_id = st.text_input("厂商 ID", placeholder="kimi（小写字母/数字/_/-）")
                        _f_name = st.text_input("显示名", placeholder="月之暗面 Kimi")
                        _f_key = st.text_input("API Key", type="password",
                                               placeholder="留空则稍后填（也可只填环境变量名）")
                    with _ac2:
                        _f_url = st.text_input("Base URL", placeholder="https://api.moonshot.cn/v1")
                        _f_models = st.text_input("已声明模型（逗号分隔）",
                                                  placeholder="moonshot-v1-8k, kimi-k2-0711-preview")
                        _f_match = st.text_input("匹配模式（逗号分隔，可选）",
                                                 placeholder="moonshot-*, kimi-*")
                    _f_default = st.checkbox("同时设为默认厂商（所有未命中的模型都走它）", value=False)
                    st.caption("API Key 会写入 `.env` 的 `<ID>_API_KEY`，**不会**进入清单文件。")
                    if st.form_submit_button("➕ 添加", type="primary"):
                        try:
                            from config.settings import reload_config
                            from llm.deepseek_client import reset_client_cache
                            _pid = _prov.validate_provider_id(_f_id)
                            if any(p["id"] == _pid for p in _providers):
                                raise ValueError(f"厂商 ID「{_pid}」已存在")
                            if not _f_url.strip():
                                raise ValueError("Base URL 必填，例如 https://api.moonshot.cn/v1")
                            _env_name = f"{_pid.upper().replace('-', '_')}_API_KEY"
                            _entry = {
                                "id": _pid,
                                "name": _f_name.strip() or _pid,
                                "base_url": _f_url.strip(),
                                "api_key_env": _env_name,
                                "models": split_list(_f_models),
                                "match": split_list(_f_match),
                                "enabled": True,
                            }
                            _prov.save_registry({
                                "default_provider": _pid if _f_default else _default_pid,
                                "providers": _providers + [_entry],
                            })
                            if _f_key.strip():
                                update_env_vars({_env_name: _f_key.strip()})
                                reload_config()
                            reset_client_cache()
                            flash(f"已添加厂商「{_entry['name']}」，密钥变量 {_env_name}")
                            st.rerun()
                        except Exception as e:                   # noqa: BLE001
                            st.error(f"添加失败：{e}")

            # ── 编辑 / 删除 ──
            with st.expander("✏️ 编辑 / 删除厂商", expanded=False):
                _opts = [f"{p['name']}（{p['id']}）" for p in _providers]
                _sel = st.selectbox("选择厂商", _opts, key="cfg_edit_provider")
                _cur = _providers[_opts.index(_sel)]
                _sid = _cur["id"]
                with st.form("edit_provider_form"):
                    _ec1, _ec2 = st.columns(2)
                    with _ec1:
                        _e_name = st.text_input("显示名", value=_cur["name"], key=f"cfg_e_name_{_sid}")
                        _e_models = st.text_input("已声明模型（逗号分隔）", value=", ".join(_cur["models"]),
                                                  key=f"cfg_e_models_{_sid}")
                        _e_key = st.text_input("更新 API Key", type="password",
                                               placeholder="留空则不修改", key=f"cfg_e_key_{_sid}")
                    with _ec2:
                        _e_url = st.text_input(
                            "Base URL", value=_cur["base_url"], key=f"cfg_e_url_{_sid}",
                            help="注意：.env 里若存在对应的 <ID>_BASE_URL，它会优先生效（见上方表格的「实际生效」列）",
                        )
                        _e_match = st.text_input("匹配模式（逗号分隔）", value=", ".join(_cur["match"]),
                                                 key=f"cfg_e_match_{_sid}")
                        _e_env = st.text_input("密钥环境变量名", value=_cur.get("api_key_env", ""),
                                               key=f"cfg_e_env_{_sid}",
                                               help=".env 里保存密钥的变量名，如 MOONSHOT_API_KEY")
                    _ec3, _ec4 = st.columns(2)
                    with _ec3:
                        _e_on = st.checkbox("启用该厂商", value=_cur.get("enabled", True),
                                            key=f"cfg_e_on_{_sid}")
                        _e_def = st.checkbox("设为默认厂商", value=(_sid == _default_pid),
                                             key=f"cfg_e_def_{_sid}")
                    with _ec4:
                        _e_confirm = st.checkbox("我确认要删除该厂商", key=f"cfg_e_del_{_sid}")

                    _sb1, _sb2 = st.columns(2)
                    _save = _sb1.form_submit_button("💾 保存修改", type="primary")
                    _del = _sb2.form_submit_button("🗑️ 删除该厂商")

                    if _save:
                        try:
                            from config.settings import reload_config
                            from llm.deepseek_client import reset_client_cache
                            _entry = dict(_cur)
                            _entry.update({
                                "name": _e_name.strip() or _sid,
                                "base_url": _e_url.strip(),
                                "api_key_env": _e_env.strip(),
                                "models": split_list(_e_models),
                                "match": split_list(_e_match),
                                "enabled": _e_on,
                            })
                            _prov.save_registry({
                                "default_provider": _sid if _e_def else _default_pid,
                                "providers": [_entry if p["id"] == _sid else p for p in _providers],
                            })
                            if _e_key.strip():
                                update_env_vars({
                                    (_e_env.strip() or f"{_sid.upper()}_API_KEY"): _e_key.strip()
                                })
                                reload_config()
                            reset_client_cache()
                            flash(f"已保存「{_entry['name']}」")
                            st.rerun()
                        except Exception as e:                   # noqa: BLE001
                            st.error(f"保存失败：{e}")

                    if _del:
                        if not _e_confirm:
                            st.error("删除前请先勾选上面的「我确认要删除该厂商」")
                        elif len(_providers) <= 1:
                            st.error("至少要保留一个厂商，否则所有 LLM 调用都会失败")
                        else:
                            try:
                                from llm.deepseek_client import reset_client_cache
                                _rest = [p for p in _providers if p["id"] != _sid]
                                _prov.save_registry({
                                    "default_provider": (_default_pid if _default_pid != _sid
                                                         else _rest[0]["id"]),
                                    "providers": _rest,
                                })
                                reset_client_cache()
                                flash(f"已删除厂商「{_cur['name']}」", "warning")
                                st.rerun()
                            except Exception as e:               # noqa: BLE001
                                st.error(f"删除失败：{e}")

        # ===== 功能模型分配（功能级路由）=====
        # 改造前「哪个功能用哪个模型」无处可配：多数节点读 DEFAULT_MODEL_NAME，而 AI 助手/复盘
        # 用的是**导入期固化的模块常量** —— 改 .env 后不重启进程不生效（10-13 踩过）。
        # 现在外置成 config/llm_feature_models.json（mtime 缓存，**改完即时生效**）。
    if visible(_SEL, SEC_FEATURE):
        st.subheader("🎚️ 功能模型分配")

        _mr = None
        try:
            from config import model_routing as _mr
        except Exception as _mr_err:                             # noqa: BLE001
            st.error(f"功能路由模块不可用：{_mr_err}")

        if _mr is not None:
            _mr_path = _mr.feature_models_path()
            _mr_overrides = _mr.get_feature_models()
            st.caption(
                f"映射文件：`{_mr_path}`　｜　**改完即时生效，无需重启进程**（按 mtime 热加载）。"
                "未指定的功能「跟随默认」（多数走 `DEFAULT_MODEL_NAME`；AI 助手/复盘/市场观点"
                "走各自的默认档位，见下表）。"
            )

            if not _mr.feature_models_file_exists():
                st.info("尚未创建映射文件：当前**所有功能都跟随默认模型**，行为与改造前完全一致。")
                if st.button("📄 初始化映射文件"):
                    if _mr.init_feature_models_file():
                        flash(f"已创建 {_mr_path}")
                        st.rerun()
                    else:
                        st.error("创建失败（详见日志；不会静默）")

            _mr_models = _prov.available_models() if _prov is not None else []

            def _mr_status(model: str) -> tuple:
                """(厂商名, 状态文案)。未声明的模型名 / 密钥未配置的厂商都必须**可见**。"""
                if _prov is None:
                    return "—", "—"
                if not any(m["model"].lower() == model.lower() for m in _mr_models):
                    return _prov.describe_model(model), "⚠️ 未声明（仍会按兜底厂商路由）"
                provider = _prov.resolve_provider(model)
                if provider is not None and not _prov.provider_ready(provider):
                    return provider["name"], "⚠️ 密钥未配置（将回退默认厂商）"
                return _prov.describe_model(model), "✅"

            # ── 当前分配总览 ──
            _mr_rows = []
            for _key, _meta in _mr.FEATURES.items():
                _model = _mr.resolve_model(_key)
                if _key in _mr_overrides:
                    _source = "⭐ 覆盖"
                elif _meta.get("reserved"):
                    _source = "预留位（无调用点）"
                else:
                    _source = "跟随默认"
                _vendor, _state = _mr_status(_model)
                _mr_rows.append({
                    "功能": _meta["label"], "分组": _meta["group"], "功能键": _key,
                    "当前使用模型": _model, "来源": _source, "厂商": _vendor, "状态": _state,
                })
            st.dataframe(_mr_rows, width="stretch", hide_index=True)

            # ── 逐功能指定 / 恢复 ──
            _mr_pick_labels = [f"{m['model']}{_SEP}{m['provider_name']}"
                               + ("" if m["ready"] else "（密钥未配置）") for m in _mr_models]
            _mr_feature_opts = [f"{meta['label']}（{key}）" for key, meta in _mr.FEATURES.items()]
            _mr_feature_by_opt = {opt: key for opt, key in zip(_mr_feature_opts, _mr.FEATURES)}

            with st.expander("✏️ 为某功能指定模型 / 恢复默认", expanded=False):
                with st.form("mr_set_form"):
                    _fc1, _fc2 = st.columns(2)
                    with _fc1:
                        _sel_feature = st.selectbox("功能", _mr_feature_opts, key="mr_sel_feature")
                    with _fc2:
                        _sel_model = st.selectbox(
                            "模型（「跟随默认」= 清除该功能的覆盖）",
                            ["（跟随默认）"] + _mr_pick_labels, key="mr_sel_model")
                    st.caption("也可以手填模型名（任意 OpenAI 兼容端点都允许，但必须在上方「模型接入」里已声明厂商）。")
                    _mr_manual = st.text_input("或手填模型名（优先级更高）", key="mr_manual_model")
                    _b1, _b2 = st.columns(2)
                    _mr_save = _b1.form_submit_button("💾 保存", type="primary")
                    _mr_reset = _b2.form_submit_button("↩️ 恢复默认")
                    if _mr_save or _mr_reset:
                        _target = _mr_feature_by_opt[_sel_feature]
                        _value = "" if _mr_reset else (_mr_manual.strip() or (
                            "" if _sel_model == "（跟随默认）" else _sel_model.split(_SEP)[0]))
                        if _mr.set_feature_model(_target, _value):
                            flash(f"「{_mr.feature_label(_target)}」已{'恢复默认' if not _value else '指定为 ' + _value}")
                            st.rerun()
                        else:
                            st.error("保存失败：写入未成功（详见日志；不会静默）")

            # ── 逐功能连通性测试（用该功能**当前解析到的模型**）──
            st.markdown("**逐功能连通性测试**（用该功能当前解析到的模型发一条最小请求）")
            for _key, _meta in _mr.FEATURES.items():
                _rc1, _rc2, _rc3, _rc4 = st.columns([3, 3, 2, 1])
                _model = _mr.resolve_model(_key)
                _rc1.write(_meta["label"] + ("　（预留位）" if _meta.get("reserved") else ""))
                _rc2.code(_model, language=None)
                _rc3.caption(_mr_status(_model)[1])
                if _rc4.button("测试", key=f"mr_test_{_key}"):
                    with st.spinner(f"调用 {_model} 中…"):
                        try:
                            from llm.deepseek_client import DeepSeekClient, TEMP_DETERMINISTIC
                            _cli = DeepSeekClient(model=_model)
                            _reply = _cli.call_api(
                                [{"role": "user", "content": "请只回复两个字母：OK"}],
                                temperature=TEMP_DETERMINISTIC, purpose="config.test")
                            st.success(f"{_meta['label']}：连通正常（{_model}）｜回复 {str(_reply)[:40]}")
                        except Exception as _e:                  # noqa: BLE001
                            st.error(f"{_meta['label']}：连通失败 — {_e}")

            # ── 按功能成本统计（R12：llm_calls 的 token；2026-10-23 起带金额）──
            with st.expander("💰 按功能成本统计（读 llm_calls）", expanded=False):
                _usage_hours = st.selectbox(
                    "统计窗口", [("近 24 小时", 24), ("近 7 天", 24 * 7), ("近 30 天", 24 * 30)],
                    format_func=lambda x: x[0], key="mr_usage_window")[1]
                _usage = _mr.feature_usage(hours=_usage_hours)
                if not _usage:
                    st.caption("读取 llm_calls 失败（数据库暂不可用），本面板这次拿不到数据。")
                else:
                    try:
                        from core import llm_cost as _cost
                        _costs = _cost.feature_costs(hours=_usage_hours)
                        _budget = _cost.budget_status()
                    except Exception as _ce:                # noqa: BLE001
                        _costs, _budget = {}, {}
                        st.caption(f"金额列不可用（{_ce}）；token 数据仍然有效。")
                    _tot_c = sum(u["count"] for u in _usage)
                    _tot_t = sum(u["tokens"] for u in _usage)
                    _tot_f = sum(u["failed"] for u in _usage)
                    _tot_money = sum(_costs.values())
                    uc1, uc2, uc3, uc4 = st.columns(4)
                    uc1.metric("调用次数", f"{_tot_c:,}")
                    uc2.metric("失败", f"{_tot_f:,}")
                    uc3.metric("tokens", f"{_tot_t:,}")
                    uc4.metric("金额", f"¥{_tot_money:.2f}" if _costs else "—")
                    if _budget.get("enabled"):
                        # 预算条：ratio > 1 时 st.progress 会报错，先夹住
                        _r = min(max(float(_budget.get("ratio") or 0), 0.0), 1.0)
                        st.progress(_r, text=(
                            f"今日 ¥{_budget.get('spent', 0):.2f} / 预算 ¥{_budget.get('cap', 0):.2f}"
                            f"（{_r*100:.0f}%）"
                            + ("　⚠️ 已超预算：盘后批量分析会被跳过" if _budget.get("exceeded") else "")))
                    else:
                        st.caption("日预算闸门未启用（`LLM_DAILY_BUDGET_CNY` ≤ 0）")
                    _rows = []
                    for u in _usage:
                        _m = _costs.get(u["feature"])
                        _rows.append({"功能": u["label"], "分组": u["group"],
                                      "功能键": u["feature"], "次数": u["count"],
                                      "失败": u["failed"], "tokens": u["tokens"],
                                      "金额¥": (round(_m, 4) if _m is not None else None)})
                    st.dataframe(_rows, width="stretch", hide_index=True)
                    st.caption("「未登记功能」= 出现在 llm_calls 里但不在 FEATURES 注册表中的 purpose"
                               "（例如新增调用点忘了登记）——它**不会**被功能路由覆盖，请补登记。"
                               "　金额按**调用发生时**的单价记账（峰谷/阶梯已计入），"
                               "老行没有金额则按当前单价估算。")

        st.divider()

        # ===== API Key 编辑 =====
    if visible(_SEL, SEC_KEYS):
        st.subheader("✏️ 更新密钥")
        st.caption("留空表示不修改该键。保存写入项目根目录 .env（UTF-8）；**API Key 保存后立即生效**（自动重载配置并清空 LLM 客户端缓存）。")
        with st.form("update_keys"):
            col_a, col_b = st.columns(2)
            with col_a:
                nk1 = st.text_input("DEEPSEEK_API_KEY", type="password", placeholder="保持不变则留空")
                nk2 = st.text_input("QWEN_API_KEY", type="password", placeholder="保持不变则留空")
                nk3 = st.text_input("TUSHARE_TOKEN", type="password", placeholder="可选，启用 tushare 降级链")
            with col_b:
                nk4 = st.text_input("DEEPSEEK_BASE_URL", placeholder=env.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
                nk5 = st.text_input("QWEN_BASE_URL", placeholder=env.get("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"))
                nk6 = st.text_input("DINGTALK_WEBHOOK", type="password", placeholder="可选")
            submitted = st.form_submit_button("💾 保存到 .env", type="primary")
            if submitted:
                updates = {}
                for key, val in [("DEEPSEEK_API_KEY", nk1), ("QWEN_API_KEY", nk2), ("TUSHARE_TOKEN", nk3),
                                 ("DEEPSEEK_BASE_URL", nk4), ("QWEN_BASE_URL", nk5), ("DINGTALK_WEBHOOK", nk6)]:
                    if val.strip():
                        updates[key] = val.strip()
                if updates:
                    try:
                        update_env_vars(updates)
                        # 热重载：config 的字段在导入时固化，必须显式 reload；
                        # LLM 客户端按模型名 lru_cache 了旧密钥，必须清缓存，
                        # 否则新 Key 在进程重启前不会生效。
                        from config.settings import reload_config
                        from llm.deepseek_client import reset_client_cache
                        reload_config()
                        reset_client_cache()
                        st.success(f"已写入 {len(updates)} 项并立即生效：{', '.join(updates)}")
                    except Exception as e:
                        st.error(f"写入失败: {e}")
                else:
                    st.info("未填写任何新值")

        # ===== 推送消息模板 =====
    if visible(_SEL, SEC_TPL):
        st.subheader("📨 推送消息模板")
        st.caption("这里的正文就是飞书/钉钉**推送出去的内容**。改完**即时生效**"
                   "（模板每次推送都从库里读，不用重启调度器进程）。"
                   "占位符写成 {{变量名}}；**未知的占位符会原样保留**，方便你发现拼错。")
        from core import push_templates as _tpl
        _overrides = _tpl.user_overrides()
        # ⚠️ "恢复默认"要重建 text_area（否则它还显示旧内容）。
        #    Streamlit 里 widget 的 session_state 在**实例化后不可改** ⇒ 用
        #    "版本号进 key"的方式换掉 widget，而不是去改它的 state。
        _vers = st.session_state.get("_tpl_ver") or {}
        for _key, (_label, _vars) in _tpl.TEMPLATE_META.items():
            _custom = _key in _overrides
            with st.expander(f"{_label}{'　✏️ 已自定义' if _custom else ''}", expanded=False):
                st.caption(f"可用变量：{_vars}")
                _body = st.text_area("模板正文", value=_tpl.get_template(_key),
                                     height=180, key=f"tpl_{_key}_{_vers.get(_key, 0)}")
                _c1, _c2, _c3 = st.columns([1, 1, 3])
                with _c1:
                    if st.button("💾 保存", key=f"tpl_save_{_key}", type="primary"):
                        if _tpl.save_template(_key, _body):
                            flash(f"已保存「{_label}」，下次推送即生效")
                        else:
                            flash("保存失败，请看日志", "error")
                with _c2:
                    def _reset(_k=_key, _lbl=_label):
                        _tpl.reset_template(_k)
                        _v = dict(st.session_state.get("_tpl_ver") or {})
                        _v[_k] = _v.get(_k, 0) + 1
                        st.session_state["_tpl_ver"] = _v
                        flash(f"已恢复「{_lbl}」的默认文案")
                    st.button("↩️ 恢复默认", key=f"tpl_reset_{_key}", on_click=_reset)
                with _c3:
                    if st.button("👀 预览", key=f"tpl_prev_{_key}"):
                        _ctx = (_tpl.SAMPLE_CONTEXT.get(_key)
                                or _tpl.SAMPLE_ITEM_CONTEXT.get(_key) or {})
                        st.markdown("**预览（示例数据；实际内容随行情变化）**")
                        st.markdown(_tpl.render(_body, _ctx))
                        with st.expander("查看渲染后的源码", expanded=False):
                            st.code(_tpl.render(_body, _ctx), language="markdown")

    # ===== LLM 连通性测试 =====
    if visible(_SEL, SEC_LLM):
        st.subheader("🧪 LLM 连通性测试")
        # 接了几家厂商之后，"能不能连通"必须能**逐家**验证——否则某个新厂商密钥填错，
        # 只有等它真的被用到的任务跑失败才会暴露。
        _test_labels = []
        if _prov is not None:
            try:
                _test_labels = [f"{m['model']}{_SEP}{m['provider_name']}"
                                + ("" if m["ready"] else "（密钥未配置）")
                                for m in _prov.available_models()]
            except Exception:                                    # noqa: BLE001
                _test_labels = []
        if not _test_labels:
            _test_labels = ["deepseek-chat"]
        _tc1, _tc2 = st.columns([2, 1])
        with _tc1:
            _test_model = st.selectbox("测试哪个模型", _test_labels, key="cfg_test_model").split(_SEP)[0]
        with _tc2:
            st.caption("逐个模型测一遍，能直接定位是哪家厂商的密钥 / 地址配错了。")
        if st.button("发送测试消息"):
            with st.spinner(f"调用 {_test_model} 中（约1-3秒）..."):
                try:
                    from llm.deepseek_client import DeepSeekClient
                    client = DeepSeekClient(model=_test_model)
                    from llm.deepseek_client import TEMP_DETERMINISTIC
                    reply = client.call_api([{"role": "user", "content": "请只回复两个字母：OK"}],
                                            temperature=TEMP_DETERMINISTIC,
                                            purpose="config.test")
                    _vendor = _prov.describe_model(_test_model) if _prov else "—"
                    st.success(f"连通正常 ｜ 模型 {_test_model} ｜ 厂商 {_vendor} ｜ "
                               f"回复：{str(reply)[:60]}")
                except Exception as e:
                    st.error(f"连通失败：{e}")

        st.divider()

        # ===== 运行健康（调度器在不在跑 / 数据在不在长）=====
        # 为什么放最前面：调度器曾长期"从未启动"而无人察觉（入口脚本启动即崩），
        # 结果 analyses/screen_picks 永远是空表、胜率统计无从谈起。
        # "服务在跑"必须是页面第一眼能看到的状态。
    if visible(_SEL, SEC_HEALTH):
        st.subheader("🩺 运行健康")
        try:
            from core import scheduler_health as health_mod
            _health = health_mod.check_health()
            _sched = _health["scheduler"]
            hc1, hc2 = st.columns([1, 3])
            hc1.metric("调度器", "运行中" if _sched["running"] else "未运行",
                       delta="心跳正常" if _sched["running"] else _sched["reason"],
                       delta_color="normal" if _sched["running"] else "inverse")
            with hc2:
                for w in _health.get("warnings", []):
                    st.warning(f"⚠️ {w}")
                if not _health.get("warnings"):
                    st.caption(f"检查时间 {_health['checked_at']}｜所有定时任务与数据链路正常。")
            _rows = _health.get("data") or {}
            if _rows:
                cols = st.columns(len(_rows))
                for col, (t, info) in zip(cols, _rows.items()):
                    col.metric(t, f"{info['rows']:,}",
                               delta=f"最新 {(info.get('latest') or '—')[:10]}",
                               delta_color="off")
            # ── 本地面板（P0-3，2026-10-18）──
            # ⚠️ 只报"rows 1,130 万"不够：估值列覆盖率 12/5830 = 0.21% 必须显式，
            #    否则会被读成"历史行情齐了"（缺就是缺）。
            _pn = panel_overview()
            if _pn.get("error"):
                st.caption(f"本地面板统计不可用：{_pn['error']}")
            elif _pn.get("rows_disk"):
                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("本地面板股票数", f"{_pn['codes']:,} 只",
                           delta=f"至 {_pn['date_max']}", delta_color="off")
                pc2.metric("面板行数（日线）", f"{_pn['rows_disk']:,} 行",
                           delta=f"自 {_pn['date_min']}", delta_color="off")
                pc3.metric("估值列已覆盖", f"{_pn.get('valued_codes', 0):,} 只",
                           delta=("待 BaoStock 解封补齐"
                                  if _pn.get("valued_codes", 0) < _pn["codes"] else "已齐"),
                           delta_color="off")
                st.caption(f"数据源：tushare 按日（价量完整）＋ BaoStock（估值列，"
                           f"受黑名单中断 ⇒ 仅少量股票有 PE/PB）。"
                           f"概况生成于 {_pn.get('at') or '—'}（`data/panel/overview.json`），"
                           f"落盘行数含少量重复分片。")
            # ── 采集覆盖率（P2-7，2026-10-17）──
            # ⚠️ 只显示"落了 N 行"会把 54.8% 的覆盖看成 100%（实测 10-16：2991/5459）。
            #    这里把「实取 / 全市场 / 覆盖率 / 降级原因」摆出来。
            try:
                from db import store as _db
                _cov_rows = _db.load_data_coverage(limit=12)
                if _cov_rows:
                    st.markdown("**采集覆盖率**（实取 ÷ 全市场；“未提供/不可评” **不等于** 正常）")
                    st.dataframe([{
                        "数据日期": r["data_date"], "数据集": r["dataset"],
                        "来源": r.get("source") or "—",
                        "实取": f"{r['got']:,}" if r.get("got") is not None else "—",
                        "全市场": f"{r['total']:,}" if r.get("total") else "未提供",
                        "覆盖率": (f"{r['ratio']:.1%}" if r.get("ratio") is not None
                                   else "不可评"),
                        "状态": r.get("status") or "—",
                        "说明": (r.get("note") or "")[:60] or "—",
                    } for r in _cov_rows], width="stretch", hide_index=True)
                else:
                    st.caption("尚无覆盖率记录（15:40 快照任务成功跑过一次后出现）。")
            except Exception as _ce:                        # noqa: BLE001
                st.caption(f"覆盖率读取失败: {_ce}")

            # ── 手动重放阶段（P0-6，2026-10-18）──
            # 原状：只有 CLI `--once`（整个盘后全流程）⇒ 15:40 快照挂了 / 15:45 复盘崩了，
            # 当天就没有第二次机会。这里按 job_id 精确重跑，`task_locks` 防重入。
            # ⚠️ 用 subprocess 调 CLI（而不是直接 import run_job）：
            #    ① 与手动在命令行跑**行为完全一致**（可复现）；② 异常/退出码不炸页面；
            #    ③ 不污染 Streamlit 进程的全局状态。
            st.markdown("**手动重放阶段**（当天补救：某阶段挂了不必等下一个交易日）")
            try:
                from core.scheduler import JOBS as _JOBS
                _opts = [(jid, desc) for jid, _kw, _m, desc in _JOBS]
            except Exception as _je:                        # noqa: BLE001
                _opts = []
                st.caption(f"任务清单读取失败：{_je}")
            if _opts:
                _c1, _c2, _c3 = st.columns([2, 1.4, 1])
                _pick = _c1.selectbox(
                    "阶段", [f"{d}（{j}）" for j, d in _opts],
                    key="replay_pick", label_visibility="collapsed")
                _date_in = _c2.text_input(
                    "数据日期（仅快照）", value="", placeholder="YYYY-MM-DD，留空=当天",
                    key="replay_date", label_visibility="collapsed")
                _go = _c3.button("重跑本阶段", key="replay_go", width="stretch")
                if _go:
                    import re as _re
                    import subprocess
                    import sys as _sys
                    _jid = _re.search(r"（(.+?)）$", _pick).group(1)
                    _cmd = [_sys.executable, "scripts/run_scheduler.py", "--job", _jid]
                    if _date_in.strip():
                        _cmd += ["--date", _date_in.strip()]
                    with st.spinner(f"正在重跑 {_jid}…（快照约 1~2 分钟；复盘含 LLM 会更久）"):
                        try:
                            # ⚠️ 源码模式下 `_sys.executable` 是 python.exe（控制台程序），
                            # 不传 CREATE_NO_WINDOW 会在重跑期间一直挂着一个黑框（1~2 分钟）。
                            # 打包后是 GUI 子系统，此参数无副作用。
                            _flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                            _r = subprocess.run(
                                _cmd, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=900,
                                cwd=PROJECT_ROOT, creationflags=_flags)
                            _rc, _err = _r.returncode, (_r.stdout or "") + (_r.stderr or "")
                        except subprocess.TimeoutExpired:
                            _rc, _err = -1, "900s 超时（长任务建议用命令行跑）"
                            st.warning(_err)
                    if _rc == 0:
                        st.session_state["replay_msg"] = f"✅ {_jid} 已重跑完成（详见 logs/scheduler_*.log）"
                    else:
                        st.session_state["replay_msg"] = (
                            f"❌ {_jid} 重跑失败（退出码 {_rc}）：{_err.strip()[-300:]}")
                    st.rerun()
                # ⚠️ st.success 紧跟 st.rerun 会被抹掉 ⇒ 提示存 session_state 下轮渲染
                if st.session_state.get("replay_msg"):
                    st.info(st.session_state.pop("replay_msg"))
        except Exception as e:
            st.caption(f"运行健康检查不可用: {e}")

        st.divider()

        # ===== 定时调度状态 =====
    if visible(_SEL, SEC_SCHED):
        st.subheader("⏰ 定时调度")
        from core.scheduler import JOBS


        def _cron_field(value, default="0") -> str:
            """cron 字段可能是 int，也可能是 "9-11,13-15" / "*" 这类字符串（如盘中盯盘任务）"""
            if value is None:
                value = default
            return f"{value:02d}" if isinstance(value, int) else str(value)


        for job_id, cron_kwargs, mode, desc in JOBS:
            col1, col2, col3 = st.columns([1, 1, 2])
            col1.write(f"**{desc}**")
            if "hour" in cron_kwargs:
                col2.write(f"周一至周五 {_cron_field(cron_kwargs['hour'])}:"
                           f"{_cron_field(cron_kwargs.get('minute'), '*')}")
            else:
                col2.write(f"每天每小时 {_cron_field(cron_kwargs.get('minute'))} 分")
            col3.caption(f"模式: {mode} | job_id: {job_id}")

        st.caption("启动常驻调度：在项目根目录执行 `python scripts/run_scheduler.py`；"
                   "调试单次全流程：`python scripts/run_scheduler.py --once`。"
                   "LLM 深度分析不进定时任务（成本控制），请到「深度分析」页手动触发。")

        # ===== 自定义定时任务（L2 扩展层）=====
        # 上面那张表是**内置**任务（写在 core/scheduler.py::JOBS 里，改时间要改代码并重启）。
        # 这一段是**自定义**任务：存在 scheduled_jobs 表里，调度进程每 60s 热加载一次，
        # 改时间/开关**不用重启**。
        #
        # 为什么值得做：本项目两次栽在"改排期"上——新增 risk_check 忘重启导致健康检查
        # 报"跑的是旧代码"；snapshot_flow(15:40) 与 daily_review(15:45) 只隔 5 分钟而
        # 快照慢时跑 9 分钟，复盘被挤到重复采集（9-28 实测采了两次）。
        #
        # 刻意**不支持"任意 prompt 定时执行"**：LLM 是真实成本，自由文本的定时任务
        # 等于给账单开一个无人看守的阀门。可选任务限定在白名单内，深度分析不在其中。
    if visible(_SEL, SEC_JOBS):
        st.subheader("⏱️ 自定义定时任务（改完不用重启）")

        _JM = None
        try:
            from core import scheduled_jobs as _JM
        except Exception as _jm_err:                             # noqa: BLE001
            st.error(f"自定义任务模块不可用：{_jm_err}")

        if _JM is not None:
            _DAY_CN = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            _DAY_EN = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

            def _days_to_cn(cron_days) -> list:
                got = _JM._expand_days(cron_days)
                if not got:
                    return []
                return [_DAY_CN[d] for d in sorted(got)]

            def _cn_to_days(names) -> str:
                idx = sorted({_DAY_CN.index(n) for n in names if n in _DAY_CN})
                if len(idx) == 7:
                    return "*"
                return ",".join(_DAY_EN[i] for i in idx)

            # 运行中的调度进程上实际挂了哪些自定义任务（来自心跳）。
            # None = 心跳没有这个字段 → 那个进程是加这一层之前的代码，改动不会生效。
            _hb_custom = None
            try:
                from core import scheduler_health as _health_mod
                _hb_custom = _health_mod.read_heartbeat().get("custom_job_ids")
            except Exception:                                    # noqa: BLE001
                _hb_custom = None

            try:
                from db import store as _db_store
                _jobs = _db_store.list_scheduled_jobs()
            except Exception as _e:                              # noqa: BLE001
                _jobs = []
                st.warning(f"读取自定义任务失败：{_e}")

            # 只列**用户可自建**的任务类型；`schedulable=False` 的是内置专用
            # （如盘后深度分析：LLM 重链路，自建多个会打爆账单）
            _mode_rows = {k: v for k, v in _JM.JOB_MODES.items()
                          if v.get("schedulable", True)}
            _MODE_LABEL = {k: v["label"] for k, v in _mode_rows.items()}
            _MODE_OPTS = [f"{v['label']}｜{k}" for k, v in _mode_rows.items()]
            _MODE_BY_OPT = {opt: opt.rsplit("｜", 1)[-1] for opt in _MODE_OPTS}

            st.caption(
                f"共 {len(_jobs)} 个自定义任务　｜　"
                "存于 `scheduled_jobs` 表，调度进程**每 60 秒热加载一次**，保存后无需重启。"
                "可选任务限定在白名单内（深度分析等 LLM 重链路刻意不开放，避免账单失控）。"
            )

            if _jobs:
                _rows = []
                for _j in _jobs:
                    _cron = _j.get("cron") or {}
                    _bad = _j.get("error") or ""
                    _conf = []
                    if _cron:
                        try:
                            _conf = [c for c in _JM.find_conflicts(
                                _cron, mode=_j.get("mode", ""), exclude_id=_j["job_id"],
                                custom_jobs=_jobs) if c.get("same_mode")]
                        except Exception:                        # noqa: BLE001
                            _conf = []
                    if _bad:
                        _status = "❌ 定义有误"
                    elif not _j.get("enabled"):
                        _status = "⏸ 已停用"
                    elif _hb_custom is None:
                        _status = "⏳ 未生效（调度进程是旧版代码，需重启一次）"
                    elif _j["job_id"] in (_hb_custom or []):
                        _status = "✅ 已生效"
                    else:
                        _status = "⏳ 待生效（60 秒内）"
                    _nxt = _JM.next_fire_time(_cron) if _cron else None
                    _rows.append({
                        "名称": _j.get("name") or _j["job_id"],
                        "执行任务": _MODE_LABEL.get(_j.get("mode"), _j.get("mode") or "—"),
                        "排期": _JM.describe_cron(_cron) if _cron else f"❌ {_bad}",
                        "下次执行": f"{_nxt:%m-%d %H:%M}" if _nxt else "—",
                        "状态": _status,
                        "⚠️ 与内置任务同时刻": "、".join(c["name"] or c["job_id"] for c in _conf) or "—",
                        "备注": _j.get("note") or "",
                        "job_id": _j["job_id"],
                    })
                st.dataframe(_rows, width="stretch", hide_index=True)
            else:
                st.info("还没有自定义任务。下面可以新建一个，例如「复盘补跑 16:00」"
                        "或把内置任务换个时间再跑一遍。")

            # ── 新建 ──
            with st.expander("➕ 新建自定义任务", expanded=not _jobs):
                st.caption("提示：同一天同一分钟跑了**同一个任务两遍**会被标红——"
                           "本项目真实发生过「快照跑太久把复盘挤到重复采集」，"
                           "排期前先看下方的冲突提示。")
                with st.form("add_custom_job_form"):
                    _ac1, _ac2 = st.columns(2)
                    with _ac1:
                        _an = st.text_input("任务名称", placeholder="例如：复盘补跑")
                        _am_opt = st.selectbox("要执行的任务", _MODE_OPTS, key="cjob_add_mode")
                    with _ac2:
                        _adays = st.multiselect("重复（星期）", _DAY_CN,
                                                default=_DAY_CN[:5], key="cjob_add_days")
                        _at = st.columns(2)
                        _ah = _at[0].number_input("小时", min_value=0, max_value=23, value=15,
                                                  step=1, key="cjob_add_h")
                        _ami = _at[1].number_input("分钟", min_value=0, max_value=59, value=0,
                                                   step=1, key="cjob_add_m")
                    _anote = st.text_input("备注（可选）", placeholder="为什么要加这个任务")
                    _amode = _MODE_BY_OPT[_am_opt]
                    _mode_info = _mode_rows[_amode]
                    st.caption(f"**{_mode_info['label']}**：{_mode_info['desc']}"
                               + ("　｜　⚠️ 该任务依赖当日行情，非交易日会自动跳过。"
                                  if _amode in _JM.TRADING_DAY_MODES else ""))
                    if st.form_submit_button("➕ 创建", type="primary"):
                        try:
                            from db import store as _db
                            _cron_in = {"hour": int(_ah), "minute": int(_ami),
                                        "day_of_week": _cn_to_days(_adays)}
                            _taken = set(_JM.builtin_job_ids()) | {j["job_id"] for j in _jobs}
                            _new = _JM.validate_job({"name": _an, "mode": _amode, "cron": _cron_in},
                                                    taken_ids=_taken)
                            _conf = _JM.find_conflicts(_cron_in, mode=_amode,
                                                       custom_jobs=_jobs)
                            _db.upsert_scheduled_job(_new)
                            _msg = (f"已创建「{_new['name']}」（{_JM.describe_cron(_new['cron'])}"
                                    f"　{_MODE_LABEL[_new['mode']]}），60 秒内生效")
                            if _conf:
                                _names = "、".join(c["name"] or c["job_id"] for c in _conf)
                                _msg += f"。⚠️ 与 {_names} 时间重叠，注意资源争抢"
                            flash(_msg, "warning" if _conf else "success")
                            st.rerun()
                        except Exception as e:                   # noqa: BLE001
                            st.error(f"创建失败：{e}")

            # ── 编辑 / 停用 / 删除 ──
            if _jobs:
                with st.expander("✏️ 编辑 / 停用 / 删除", expanded=False):
                    _opts = [f"{j.get('name') or j['job_id']}（{j['job_id']}）" for j in _jobs]
                    _sel = st.selectbox("选择任务", _opts, key="cjob_edit_pick")
                    _cur = _jobs[_opts.index(_sel)]
                    _cid = _cur["job_id"]
                    _ccron = _cur.get("cron") or {}
                    with st.form("edit_custom_job_form"):
                        _ec1, _ec2 = st.columns(2)
                        with _ec1:
                            _en = st.text_input("任务名称", value=_cur.get("name") or "",
                                                key=f"cjob_e_name_{_cid}")
                            _cur_mode = _cur.get("mode") if _cur.get("mode") in _mode_rows else next(
                                iter(_mode_rows))
                            _em_opt = st.selectbox(
                                "要执行的任务", _MODE_OPTS,
                                index=list(_mode_rows).index(_cur_mode),
                                key=f"cjob_e_mode_{_cid}")
                        with _ec2:
                            _edays = st.multiselect(
                                "重复（星期）", _DAY_CN,
                                default=_days_to_cn(_ccron.get("day_of_week")) or _DAY_CN[:5],
                                key=f"cjob_e_days_{_cid}")
                            _et = st.columns(2)
                            _eh = _et[0].number_input(
                                "小时", min_value=0, max_value=23,
                                value=int(_ccron.get("hour")) if isinstance(_ccron.get("hour"), int) else 15,
                                step=1, key=f"cjob_e_h_{_cid}")
                            _emi = _et[1].number_input(
                                "分钟", min_value=0, max_value=59,
                                value=int(_ccron.get("minute")) if isinstance(_ccron.get("minute"), int) else 0,
                                step=1, key=f"cjob_e_m_{_cid}")
                        _enote = st.text_input("备注", value=_cur.get("note") or "",
                                               key=f"cjob_e_note_{_cid}")
                        _ec3, _ec4 = st.columns(2)
                        with _ec3:
                            _eon = st.checkbox("启用该任务", value=bool(_cur.get("enabled")),
                                               key=f"cjob_e_on_{_cid}")
                        with _ec4:
                            _edel = st.checkbox("我确认要删除该任务", key=f"cjob_e_del_{_cid}")

                        _sb1, _sb2 = st.columns(2)
                        _save_j = _sb1.form_submit_button("💾 保存修改", type="primary")
                        _del_j = _sb2.form_submit_button("🗑️ 删除该任务")

                        if _save_j:
                            try:
                                from db import store as _db
                                _cron_in = {"hour": int(_eh), "minute": int(_emi),
                                            "day_of_week": _cn_to_days(_edays)}
                                _upd = _JM.validate_job(
                                    {"job_id": _cid, "_original_id": _cid, "name": _en,
                                     "mode": _MODE_BY_OPT[_em_opt], "cron": _cron_in,
                                     "enabled": _eon, "note": _enote},
                                    taken_ids={j["job_id"] for j in _jobs} | set(_JM.builtin_job_ids()),
                                    require_id=True)
                                _upd.pop("trading_day_locked", None)
                                _db.upsert_scheduled_job(_upd)
                                _conf = [c for c in _JM.find_conflicts(
                                    _cron_in, mode=_upd["mode"], exclude_id=_cid,
                                    custom_jobs=_jobs) if c.get("same_mode")]
                                _msg = f"已保存「{_upd['name']}」（{_JM.describe_cron(_upd['cron'])}）"
                                if not _eon:
                                    _msg += "，已停用"
                                if _conf:
                                    _msg += (f"。⚠️ 与 {'、'.join(c['name'] or c['job_id'] for c in _conf)}"
                                             " 同一时刻跑同一个任务，会重复执行")
                                flash(_msg, "warning" if _conf else "success")
                                st.rerun()
                            except Exception as e:               # noqa: BLE001
                                st.error(f"保存失败：{e}")

                        if _del_j:
                            if not _edel:
                                st.error("删除前请先勾选上面的「我确认要删除该任务」")
                            else:
                                try:
                                    from db import store as _db
                                    _db.delete_scheduled_job(_cid)
                                    flash(f"已删除「{_cur.get('name') or _cid}」，60 秒内从调度中移除",
                                          "warning")
                                    st.rerun()
                                except Exception as e:           # noqa: BLE001
                                    st.error(f"删除失败：{e}")

            st.caption("⚠️ 自定义任务**不会**改变内置任务；两者同时刻跑同一任务会重复执行。"
                       "改完若状态一直是「未生效」，说明调度进程还没重启过（旧版代码不认识这张表）。")


        # 最新流水线结果
        st.markdown("**最新流水线结果** (`output/daily_picks/latest.json`)")
        if os.path.exists(LATEST_PATH):
            try:
                latest = json.load(open(LATEST_PATH, encoding="utf-8"))
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("生成时间", latest.get("generated_at", "N/A"))
                m2.metric("模式", latest.get("mode", "N/A"))
                m3.metric("候选数", len(latest.get("picks", [])))
                m4.metric("错误数", len(latest.get("errors", [])))
                if latest.get("skipped"):
                    st.warning(f"该次运行被跳过：{latest.get('skip_reason', '')}")
                picks = latest.get("picks") or []
                if picks:
                    rows = [{"代码": p.get("股票代码"), "名称": p.get("股票名称") or "—",
                             "评分": p.get("score"), "策略": p.get("策略", ""),
                             "板块": p.get("板块", "")} for p in picks]
                    st.dataframe(rows, width="stretch", hide_index=True)
                if latest.get("backtest"):
                    bt = latest["backtest"]
                    st.caption(f"回测: {bt.get('window', ['',''])[0]} ~ {bt.get('window', ['',''])[-1]} | "
                               f"总收益 {bt.get('total_return', 0):+.2f}% | 交易 {bt.get('total_trades', 0)} 笔 | "
                               f"报告文件: {bt.get('report_file', '—')}")
                for err in latest.get("errors", []):
                    st.error(err)
            except Exception as e:
                st.error(f"读取失败: {e}")
        else:
            st.info("尚无流水线产出。运行 `python scripts/run_scheduler.py --once` 或等待定时任务触发。")

        # 每日复盘 —— 已独立为专门页面（2026-09-28）
        #
        # 复盘是每个交易日都要看的产出，此前唯一入口是本页底部的 expander。
        # 本页职责是「API Key / 调度 / 运行健康」这类低频配置，把高频产物放在这里，
        # 结果是文件明明生成了、用户却在界面上找不到。现改为只保留状态 + 跳转，
        # 正文阅读、历史存档、预测命中率、手动生成统一收归「每日复盘」页。
        st.markdown("**每日复盘**")
        REVIEW_PATH = os.path.join(PROJECT_ROOT, "output", "daily_reports", "latest.md")
        # ⚠️ 这里不用 st.page_link：在子页面里它需要 url_pathname（入口脚本才注入），
        #    AppTest 下会 KeyError: 'url_pathname'，冒烟测试直接红。导航交给侧边栏/首页。
        st.info("📰 复盘正文、历史存档、观察点命中率与手动生成，请到侧边栏或首页的「**每日复盘**」页。")
        if os.path.exists(REVIEW_PATH):
            _mtime = datetime.fromtimestamp(os.path.getmtime(REVIEW_PATH))
            st.caption(f"最新报告：{_mtime:%Y-%m-%d %H:%M} ｜ "
                       f"{os.path.getsize(REVIEW_PATH):,} 字节 ｜ "
                       f"调度任务「每日复盘」每个交易日 15:45 自动生成")
        else:
            st.caption("尚无复盘报告。调度任务「每日复盘」在交易日 15:45 自动生成。")

        # 数据统计
        st.markdown("**数据统计**（SQLite 持久化）")
        try:
            from db import store as db_store
            stats = db_store.store_stats()
            label_map = {"news_items": "📰 新闻快讯", "chat_logs": "💬 对话消息",
                         "daily_features": "📈 相似日特征", "runs": "⚙️ 运行记录",
                         "llm_calls": "🧠 LLM 调用", "alerts": "🔔 异动预警",
                         "analyses": "🔍 分析历史", "chat_sessions": "🗂️ 对话会话"}
            stat_cols = st.columns(len(stats))
            for col, (table, info) in zip(stat_cols, stats.items()):
                col.metric(label_map.get(table, table),
                           f"{info['rows']:,}",
                           delta=f"最新 {info['latest'][:10] if info['latest'] else '—'}",
                           delta_color="off")
        except Exception as e:
            st.caption(f"数据统计不可用: {e}")

        st.divider()

        # ===== LLM 调用链可观测性（Phase 6.2）=====
    if visible(_SEL, SEC_CALLS):
        st.subheader("🧠 LLM 调用链（近 7 天）")
        try:
            from db import store as db_store
            ls = db_store.llm_call_stats(recent_days=7)
            if not ls["total"]:
                st.caption("近 7 天没有调用记录。之后的每次调用都会写入 `llm_calls` 表"
                           "（模型 / 用途 / 耗时 / token / 失败原因），不再只能翻日志文件。")
            else:
                lc1, lc2, lc3, lc4 = st.columns(4)
                lc1.metric("调用次数", f"{ls['total']:,}")
                lc2.metric("失败", f"{ls['failed']:,}",
                           delta=f"{ls['failure_rate']:.1%}", delta_color="inverse")
                lc3.metric("平均延迟", f"{ls['avg_latency_ms'] / 1000:.1f}s")
                lc4.metric("tokens", f"{ls['prompt_tokens'] + ls['completion_tokens']:,}",
                           delta=f"入 {ls['prompt_tokens']:,} / 出 {ls['completion_tokens']:,}",
                           delta_color="off")
                st.markdown("**按用途**")
                st.dataframe(
                    [{"用途": g["key"], "次数": g["count"], "失败": g["failed"],
                      "平均耗时(ms)": g["avg_latency_ms"], "tokens": g["total_tokens"]}
                     for g in ls["by_purpose"]], width="stretch", hide_index=True)
                if len(ls["by_model"]) > 1:
                    st.markdown("**按模型**")
                    st.dataframe(
                        [{"模型": g["key"], "次数": g["count"], "失败": g["failed"],
                          "平均耗时(ms)": g["avg_latency_ms"], "tokens": g["total_tokens"]}
                         for g in ls["by_model"]], width="stretch", hide_index=True)

                recent = db_store.get_recent_llm_calls(limit=50)
                with st.expander(f"最近调用明细（{len(recent)} 条）", expanded=False):
                    st.dataframe(
                        [{"时间": r["created_at"], "模型": r["model"], "用途": r["purpose"],
                          "结果": "✅" if r["ok"] else "❌", "耗时(ms)": r["latency_ms"],
                          "tokens": r["total_tokens"], "错误": (r["error"] or "")[:60]}
                         for r in recent], width="stretch", hide_index=True)
                failures = db_store.get_recent_llm_calls(limit=10, only_failures=True)
                if failures:
                    with st.expander(f"⚠️ 最近失败（{len(failures)} 条）", expanded=True):
                        for r in failures:
                            st.caption(f"- `{r['created_at']}` **{r['purpose']}** / {r['model']}："
                                       f"{r['error'] or '未记录原因'}")
        except Exception as e:
            st.caption(f"LLM 调用统计不可用: {e}")

        st.divider()

        # ===== 数据链路与参数说明 =====
    if visible(_SEL, SEC_DATA):
        st.subheader("📚 数据链路与参数")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**资金流降级链**（已实测）")
            st.caption("1️⃣ 东财逐股历史接口 → 2️⃣ Tushare moneyflow（需Token） → "
                       "3️⃣ 同花顺全市场快照（本环境当前生效，快照模式评分）\n\n"
                       "东财接口连续失败 10 分钟内自动熔断跳过。")
            st.markdown("**基准收益**")
            st.caption("沪深300 净值（新浪→腾讯降级链），接入后信息比率真实计算。")
        with c2:
            st.markdown("**核心参数（代码常量）**")
            st.caption("`scripts/backtest_rule_based.py`：SCORE_THRESHOLD=60（买入阈值）、"
                       "HOLDING_DAYS_MAX=20（最长持仓）、ATR_MULTIPLIER=2.5（止损倍数）\n\n"
                       "`core/screener.py`：STRATEGIES（五路本地条件描述）、"
                       "`core/local_screen.py`（本地初筛实现）、"
                       "SECTOR_HARDCODED（最后兜底精选池）\n\n"
                       "修改需编辑对应文件后重启。")

        st.divider()
        st.caption("⚠️ 密钥仅保存在本地 .env（已列入 .gitignore，不会提交到仓库）")


main()
