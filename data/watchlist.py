"""
自选股管理：**双轨**（用户 `user` / 阿深 `ashen`），落 SQLite `watchlist_items`。

## 为什么必须迁出 JSON（2026-10-27）

JSON 是"整文件替换"语义。一旦有两个写入方（用户 + 阿深），谁后写谁赢，
**丢掉的那一方不会得到任何报错**——它只是安静地少了十几行。
SQLite 用 `(owner, code)` 主键托管两套，天然隔离、天然幂等。

同一只票可以同时在两套里（用户提名给阿深后大概率如此），这是**正常状态**，
不是数据冲突 ⇒ 不能用"全局 code 唯一"来建模。

## 关注池 `attention_pool()`：两套自选 ∪ 今日候选 ∪ 当前持仓

> "盯盘范围"和"新闻关联范围"都只允许有一个定义，否则两处会各自演化——
> 过去新闻入库只关联自选股（`related_codes` 对全市场候选恒为空），
> 舆情因子只能靠"简称出现在正文里"兜底（见 data/news_feed 的 0.2.3 修复）。

⚠️ **`POOL_SOURCES` 必须同时含 `watchlist_user` 与 `watchlist_ashen`。**
漏掉阿深那套的后果是**静默**的：它的票进不了新闻关联 ⇒ 舆情因子永远走 0.2
兜底，而看盘页看起来完全正常（看盘页直读自选表，根本不走关注池）。
"""

import json
import logging
import os
from datetime import datetime

_logger = logging.getLogger(__name__)

# ⚠️ 这两个路径在打包后**不在同一个目录**，必须分开取：
#   · `PICKS_PATH`  是**数据** ⇒ 跟数据根走（`config/paths.data_root()`）
#   · `WATCHLIST_PATH` 是**配置** ⇒ 默认随包分发（`bundle_root()`，只读包内）；
#     但允许用户在数据目录放一份覆盖（exe 同级 `config/watchlist.json`），
#     否则自选股清单在 exe 里永远改不了。
# 2026-10-25 事故：这里原本都用 `dirname(dirname(__file__))`，打包后两者都指向
# 只读包 ⇒ 自选股读到空清单、选股路径读不到 latest.json。
# 2026-10-27：`WATCHLIST_PATH` 现在**只用于一次性迁移**，不再是数据源。
from config.paths import bundle_root as _paths_bundle_root  # noqa: E402
from config.paths import data_root as _paths_data_root  # noqa: E402

_WATCHLIST_OVERRIDE = os.path.join(_paths_data_root(), "config", "watchlist.json")
WATCHLIST_PATH = (_WATCHLIST_OVERRIDE if os.path.isfile(_WATCHLIST_OVERRIDE)
                  else os.path.join(_paths_bundle_root(), "config", "watchlist.json"))
PICKS_PATH = os.path.join(_paths_data_root(), "output", "daily_picks", "latest.json")

# 关注池上限：新闻关联是"每条新闻 × 池内每只股票"的子串匹配，必须封顶
MAX_POOL = 500

# 两套自选的 owner。`user` = 你，`ashen` = 阿深。
OWNERS = ("user", "ashen")
OWNER_LABELS = {"user": "我的自选", "ashen": "阿深自选"}

# 关注池来源。⚠️ 顺序即去重优先级：同一只票先出现的来源赢。
# 两套自选都排在最前 —— 人主动加的票比机器候选更需要稳定的关系标签。
POOL_SOURCES = ("watchlist_user", "watchlist_ashen", "picks", "positions")

# 回测 WATCHLIST 的默认自选池（与 scripts/backtest_rule_based.py 保持一致）
DEFAULT_WATCHLIST = [
    "000001", "600036", "601318",  # 金融
    "600519", "000858", "002714",  # 消费
    "002230", "300474", "000063",  # 科技
    "300750", "002594", "601012",  # 新能源
    "600276", "300760", "000963",  # 医药
]

# 默认名称表（展示兜底，实时行情返回的名称优先）
DEFAULT_NAMES = {
    "000001": "平安银行", "600036": "招商银行", "601318": "中国平安",
    "600519": "贵州茅台", "000858": "五粮液", "002714": "牧原股份",
    "002230": "科大讯飞", "300474": "景嘉微", "000063": "中兴通讯",
    "300750": "宁德时代", "002594": "比亚迪", "601012": "隆基绿能",
    "600276": "恒瑞医药", "300760": "迈瑞医疗", "000963": "华东医药",
}

# 迁移标记的 flag 名（存在库里 `app_flags`，见 `db/store.set_flag`）
_MIGRATED_FLAG = "watchlist_migrated_v1"


# ================================================================
# owner 规范化
# ================================================================
def normalize_owner(owner) -> str:
    """白名单校验。非法值**回退 user 并记 error**。

    为什么回退而不是抛：owner 只可能来自 API 白名单或内部常量，出现非法值
    说明是程序 bug。回退能让页面继续可用，而 `error` 级日志保证它被看见。
    ⚠️ 但也正因为它会回退，**写操作绝不能拿用户输入直接当 owner**——
    先过这个函数拿到规范化结果，再用它。
    """
    o = str(owner or "user").strip().lower()
    if o in OWNERS:
        return o
    _logger.error("[自选股] 非法 owner=%r，已按 'user' 处理（调用方应传 %s）",
                  owner, "/".join(OWNERS))
    return "user"


def _store():
    """延迟导入 `db.store`：本模块被 `data/news_feed` 等在启动早期导入，
    把 DB 依赖压到调用时，避免循环导入与"导入即建表"。"""
    from db import store
    return store


# ================================================================
# 一次性迁移（JSON → watchlist_items）
# ================================================================
def _read_legacy_json() -> list | None:
    """读旧 JSON。不存在/损坏 ⇒ None（**不返回默认池**，默认池由调用方决定）。"""
    if not os.path.exists(WATCHLIST_PATH):
        return None
    try:
        with open(WATCHLIST_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 旧 JSON 解析失败，迁移将改用默认池: %s", e)
        return None
    stocks = data.get("stocks", []) if isinstance(data, dict) else []
    if stocks and isinstance(stocks[0], str):              # 兼容纯代码列表
        stocks = [{"code": c, "name": DEFAULT_NAMES.get(c, ""), "note": ""} for c in stocks]
    return stocks or None


def _default_stocks() -> list:
    return [{"code": c, "name": DEFAULT_NAMES.get(c, ""), "note": "", "source": "default"}
            for c in DEFAULT_WATCHLIST]


def _ensure_migrated() -> None:
    """首次启动把旧 JSON（或默认池）导进 `user` 轨，然后在**库里**写标记。**幂等**。

    - 标记存在 ⇒ 直接返回。**用户把自选全删光也绝不会被重新灌回来**，
      这是加标记的唯一理由（只看"表里是否为空"会在清空后误判为未迁移）。
    - 表里已有行但没标记（手工建库、测试夹具）⇒ 只补标记，不覆盖。
    - 标记存 `app_flags` 而不是库旁边的文件：**库被替换/重建时标记必须一起消失**，
      否则新库会跳过迁移、用户自选原地清空（且不报错）。顺带单测换临时库也天然隔离。

    ⚠️ 迁移**只读**旧 JSON，改名/删除源文件是刻意不做的（2026-10-27 实测教训）：
    只要有一个调用方忘了 patch `WATCHLIST_PATH`，归档动作就会作用到**真实的
    `config/watchlist.json`** 上——而它当时连接的可能是个临时库。结果是
    "数据迁走了、源文件也没了"，连回滚的余地都没有。标记已经足够防止重复导入，
    文件留着只会变成一份不会被读取的备份，没有任何危害。
    """
    st = _store()
    try:
        if st.get_flag(_MIGRATED_FLAG):
            return
        existing = st.count_watchlist("user")
        if existing > 0:
            st.set_flag(_MIGRATED_FLAG, f"existing_db:{existing}")
            return
        stocks = _read_legacy_json()
        if stocks is None:
            source, stocks = "default", _default_stocks()
        else:
            source = "json"
            for s in stocks:
                s.setdefault("source", "migrated")
        count = st.replace_watchlist("user", stocks)
        st.set_flag(_MIGRATED_FLAG, f"{source}:{count}")
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 迁移失败（下次启动会重试）: %s: %s", type(e).__name__, e)
        return False


# ================================================================
# 读写
# ================================================================
def load_watchlist(owner: str = "user") -> list[dict]:
    """读取某轨自选。返回 `[{"code","name","note","source","added_at"}]`。

    读失败返回 `[]` 并记 warning —— **绝不返回默认池冒充**。默认池只在首次
    迁移时写一次；此后"读到空"就是真的空（用户删光了）。
    """
    owner = normalize_owner(owner)
    try:
        _ensure_migrated()
        return _store().list_watchlist(owner)
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 读取失败 owner=%s: %s: %s", owner, type(e).__name__, e)
        return []


def save_watchlist(stocks: list[dict], owner: str = "user") -> int:
    """**全量替换**某轨清单，返回最终条数。

    ⚠️ 这个"全量替换"语义必须保留：`core/agent_write_guard` 的 `watchlist_save`
    （AI 助手的 add/remove/update 及其**回滚**）传的就是"变更后的完整列表"，
    靠替换落地。改成增量语义会让回滚变成空操作。
    """
    owner = normalize_owner(owner)
    count = _store().replace_watchlist(owner, stocks or [])
    _logger.info("[自选股] 已保存 owner=%s 共 %d 只", owner, count)
    return count


def _load_for_write(owner: str = "user") -> list[dict] | None:
    """写前读取。返回 `None` ⇒ **库读失败，调用方必须中止写入**。

    历史语义（JSON 时代）是"文件损坏返回 None，禁止用默认池顶替去写"，
    内部函数（`core/agent_tools.py`）依赖它做 fail-closed 判断，签名保持不变。
    """
    owner = normalize_owner(owner)
    try:
        _ensure_migrated()
        return _store().list_watchlist(owner)
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 写前读取失败 owner=%s，已中止写入: %s: %s",
                        owner, type(e).__name__, e)
        return None


def add_stock(code: str, name: str = "", note: str = "",
              owner: str = "user", source: str = "manual") -> tuple[bool, str]:
    """添加自选股。返回 (是否成功, 提示消息)。库不可读时中止，不用默认池覆盖。"""
    owner = normalize_owner(owner)
    code = str(code).strip()
    if not (code.isdigit() and len(code) == 6):
        return False, "股票代码须为6位数字"
    stocks = _load_for_write(owner)
    if stocks is None:
        return False, "自选股库读取失败，已中止写入——请检查数据库"
    if any(s["code"] == code for s in stocks):
        return False, f"{code} 已在{OWNER_LABELS.get(owner, '自选')}中"
    try:
        _store().upsert_watchlist_item(owner, code, name, note, source)
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 添加失败 %s/%s: %s", owner, code, e)
        return False, f"写入失败：{type(e).__name__}"
    return True, f"已添加 {name or code}"


def remove_stock(code: str, owner: str = "user") -> tuple[bool, str]:
    """移除自选股。返回 (是否成功, 提示消息)"""
    owner = normalize_owner(owner)
    code = str(code).strip()
    try:
        ok = _store().delete_watchlist_item(owner, code)
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 移除失败 %s/%s: %s", owner, code, e)
        return False, f"写入失败：{type(e).__name__}"
    if not ok:
        return False, f"{code} 不在{OWNER_LABELS.get(owner, '自选')}中"
    return True, f"已移除 {code}"


def update_stock(code: str, name: str = None, note: str = None,
                 owner: str = "user") -> tuple[bool, str]:
    """更新自选股的名称/备注（2026-10-24，AI 助手 update 动作的数据层实现）。

    语义：None = **不修改**该字段；note 传 ""（且非 None）= 清空备注。
    add/remove 的历史签名是 (bool, msg)，这里保持一致。
    """
    owner = normalize_owner(owner)
    code = str(code).strip()
    if not (code.isdigit() and len(code) == 6):
        return False, "股票代码须为6位数字"
    name_val = str(name).strip() if name is not None and str(name).strip() else None
    note_val = None if note is None else str(note)
    if name_val is None and note_val is None:
        return False, "没有要修改的字段（name/note 至少传一个）"
    stocks = _load_for_write(owner)
    if stocks is None:
        return False, "自选股库读取失败，已中止写入——请检查数据库"
    if not any(s["code"] == code for s in stocks):
        return False, f"{code} 不在{OWNER_LABELS.get(owner, '自选')}中"
    try:
        _store().update_watchlist_item(owner, code, name_val, note_val)
    except Exception as e:                                 # noqa: BLE001
        _logger.warning("[自选股] 更新失败 %s/%s: %s", owner, code, e)
        return False, f"写入失败：{type(e).__name__}"
    changed = [k for k, v in (("名称", name_val), ("备注", note_val)) if v is not None]
    return True, f"已更新 {code} 的{'、'.join(changed)}"


def watchlist_codes(owner: str = "user") -> list[str]:
    """自选股代码列表（便捷方法）"""
    return [s["code"] for s in load_watchlist(owner)]


def both_watchlists() -> dict:
    """两套清单一次性取回：`{"user": [...], "ashen": [...]}`。

    页面与端点用这个而不是两次 `load_watchlist`：迁移只跑一次、且两套必须
    来自**同一次**读取，否则中间夹一次写入会让页面显示"我改了但对方没变"。
    """
    _ensure_migrated()
    return {o: load_watchlist(o) for o in OWNERS}


def ensure_ready() -> None:
    """确保表已建、迁移已做（**不读数据**）。

    给"想自己直读 `db.store` 以便把异常透出去"的调用方用：
    `load_watchlist` 为了不打断新闻采集会吞异常返回 `[]`，而页面端点需要
    区分"真的是空"和"库读挂了"（前者正常，后者要显式报错）。
    """
    _ensure_migrated()


def snapshot(owner: str = "user") -> dict:
    """页面用的**严格**快照：`{owner, label, stocks, counts}`，读失败**抛异常**。

    与 `load_watchlist` 的差别只有一处：它吞异常。两个入口并存是因为两类调用方
    的要求相反 —— 新闻采集不能因为库抖动整批失败，页面不能把"库读挂了"
    显示成"自选股为空"（用户会以为清单被删了）。
    """
    owner = normalize_owner(owner)
    st = _store()
    _ensure_migrated()
    return {
        "owner": owner,
        "label": OWNER_LABELS.get(owner, owner),
        "stocks": st.list_watchlist(owner),
        "counts": {o: st.count_watchlist(o) for o in OWNERS},
    }


# ================================================================
# 关注池：两套自选 ∪ 今日候选 ∪ 当前持仓
# ================================================================
def _pool_from_watchlist(owner: str) -> list:
    return [(s.get("code"), s.get("name", "")) for s in load_watchlist(owner)]


def _pool_from_picks(today: str = None) -> list:
    """今日流水线候选（output/daily_picks/latest.json）。

    **只认当天**：隔日的候选不该继续进新闻关联/盯盘范围（否则池子会越滚越大、
    并把早就过期的标的算成"今日关注"）。
    """
    today = today or datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(PICKS_PATH):
        return []
    with open(PICKS_PATH, encoding="utf-8") as f:
        latest = json.load(f)
    if latest.get("date") != today:
        return []
    return [(p.get("股票代码"), p.get("股票名称", "")) for p in (latest.get("picks") or [])]


def _pool_from_positions() -> list:
    """当前持仓（只读 code/name 两列）。

    刻意**不走 core.portfolio.get_active_positions()**：那条路会 self.load()，
    对每只持仓拉一次现价——新闻采集/盯盘范围判定不该触发行情请求。
    """
    from db.models import get_connection
    conn = get_connection()
    try:
        rows = conn.execute(
            "SELECT code, name FROM positions WHERE status = 'active'").fetchall()
        return [(r["code"], r["name"]) for r in rows]
    finally:
        conn.close()


_POOL_BUILDERS = {
    "watchlist_user": lambda: _pool_from_watchlist("user"),
    "watchlist_ashen": lambda: _pool_from_watchlist("ashen"),
    # 旧名别名（2026-10-27 前只有一套）。保留是为了**未改到的调用点不至于
    # 静默变成空池**——那会让盯盘范围和新闻关联莫名缩小，而日志里只有一条
    # "未知来源" warning，很容易被忽略。新代码一律用带 owner 的名字。
    "watchlist": lambda: _pool_from_watchlist("user"),
    "picks": _pool_from_picks,
    "positions": _pool_from_positions,
}


def attention_pool(sources=POOL_SOURCES, limit: int = MAX_POOL) -> list:
    """关注池：`[{"code", "name", "source"}]`，按 sources 顺序去重保序。

    每个来源**独立容错**：任何一个读不到（库读失败/没有任何持仓/网络无关）
    都只记日志，不影响其它来源，更不影响调用方主流程——新闻采集绝不能因为
    持仓表读不到就整批失败。同名代码出现多次时，用先出现的来源排序，
    名称取第一个非空值（自选股常常没维护名称，候选/持仓里有）。
    """
    order: dict = {}
    for source in sources:
        builder = _POOL_BUILDERS.get(source)
        if builder is None:
            _logger.warning("[关注池] 未知来源: %s", source)
            continue
        try:
            entries = builder()
        except Exception as e:                     # noqa: BLE001
            _logger.warning("[关注池] 来源 %s 读取失败（忽略）: %s", source, e)
            continue
        for code, name in entries or []:
            code = str(code or "").strip()
            if not (code.isdigit() and len(code) == 6):
                continue
            if code in order:
                if name and not order[code]["name"]:
                    order[code]["name"] = str(name).strip()
                continue
            if len(order) >= limit:
                _logger.warning("[关注池] 已达上限 %d，忽略来源 %s 的其余标的", limit, source)
                break
            order[code] = {"code": code, "name": str(name or "").strip(), "source": source}
    return list(order.values())


def attention_codes(sources=POOL_SOURCES) -> list:
    """关注池代码列表（顺序与 attention_pool 一致）"""
    return [s["code"] for s in attention_pool(sources)]
