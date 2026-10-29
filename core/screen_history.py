"""选股候选入库与 T+N 回看（P2：条件命中率回测的样本来源）。

为什么做这个（而不是先做"胜率对照实验"）
-------------------------------------
批次 15 前的实测：`analyses` 只有 1 行样本，`daily_features` 只有**上证指数**一个标的
（没有个股面板）。也就是说：
- P3「胜率对照实验」需要 50+ 样本 → 按深度分析的频率要攒两个月，现在做不了；
- P2「条件命中率回测」原打算用 daily_features 做 → 它是指数技术特征，没有市盈率/资金面，做不了。

而一次选股就产出 5~10 只候选，且天然带**条件命中明细**（NL 选股的 hits/misses）与
**因子得分**——这是唯一能快速攒够样本、又能直接回答"这些条件有没有区分度"的地方。
于是把 analyses 那套"落库 → T+N 回填 → 统计"的闭环复制到选股层：

    save_screen_result()  选股结束即落库（best-effort）
    backfill_screen_picks()  每日收盘后回填 T+5/T+20 + 同期指数基准
    store.condition_stats()  按条件分组看区分度（正 edge = 条件有效）

口径统一：T+N 收益一律走 `core.forward_returns`（与分析历史共用同一实现）。
"""

import logging
import os
import time
from datetime import datetime, timedelta

import db.store as store
from core.forward_returns import default_price_fetcher, fetch_forward_returns

_logger = logging.getLogger(__name__)

# 与分析历史同款：超过这个天数 T+20 已不可能补齐，标 failed 停止每日重试
BACKFILL_MAX_AGE_DAYS = 60

# 只回填「至少这么老」的候选：T+5 需要起点之后 6 根 K 线，5 个交易日 ≈ 7 个自然日。
# ⚠️ 不过滤的后果（2026-10-28 实测）：窗口里全是当天刚选出的行（当天落库 438 行），
#    它们 100% 算出空收益被 skip，于是任务每天白跑 17 分钟、0 条有效回填。
BACKFILL_MIN_AGE_DAYS = 7

_PRICE_KEYS = ("现价", "最新价", "最新", "price", "close", "收盘价", "收盘")


def _pick_price(cand: dict):
    """候选里的现价字段名各策略不统一，兼容取值；取不到返回 None（回填时补基准价）"""
    for k in _PRICE_KEYS:
        v = cand.get(k)
        if v not in (None, "", "-"):
            try:
                return round(float(str(v).replace(",", "").replace("元", "")), 4)
            except (TypeError, ValueError):
                continue
    return None


def _pick_conditions(cand: dict) -> dict:
    """取条件命中明细（自然语言选股才有 nl_report；普通选股为空 dict）"""
    rep = cand.get("nl_report") or cand.get("conditions") or {}
    if not isinstance(rep, dict):
        return {}
    return {k: rep.get(k) for k in ("hits", "misses", "unverified") if rep.get(k)}


def save_screen_result(result: dict, source: str = "ui",
                       pick_date: str = None) -> int:
    """把一次选股的候选落库，返回写入条数。

    Args:
        result: `StockScreener.run()` 的返回（含 market_signal / candidates / skipped）
        source: screen（内置选股页）/ nl_screen（自然语言）/ pipeline
        pick_date: 选股日，默认今天

    跳过（熊市）或没有候选时返回 0——没有"判断"就没有可回看的样本。
    """
    if not result or result.get("skipped"):
        return 0
    # 候选扩池（2026-10-22）：优先落**完整确定性排序**（`ranked`，最多 300 名），
    # 没有该字段时回退 `candidates` —— 后者只做过行业分散的 5~10 只。
    # ⚠️ 为什么分开：`candidates` 是"给用户看的名单"（要行业分散），
    #    `ranked` 是"给深度分析挑的池子"（按分数全排）。用错会导致候选池
    #    仍只有个位数，重合度降不下来（10-21 实测 56%）。
    # ⚠️ 旧 result（无 ranked）行为与改动前一字不差 —— 页面/自然语言选股仍走 candidates。
    candidates = result.get("ranked") or result.get("candidates") or []
    if not candidates:
        return 0
    signal = result.get("market_signal")
    date = pick_date or datetime.now().strftime("%Y-%m-%d")
    # ⚠️ 双格式（2026-10-25 C1）：内存态是 MarketSignal 对象，从 latest.json 读回来是 dict
    #    （core/pipeline._json_default 落盘为 dict），更老的文件是 repr 串。
    #    统一走 `core.market_signal_codec.coerce`（全项目唯一编解码器），
    #    否则 dict 输入会让 market_state/market_score 静默变 None。
    from core.market_signal_codec import coerce
    sig, _fmt = coerce(signal)
    if sig:
        ms_state, ms_score = sig.get("market_state"), sig.get("total_score")
    else:
        # 认不出的类型：保持旧行为（getattr），不因为编解码器升级而丢字段
        ms_state = getattr(signal, "market_state", None)
        ms_score = getattr(signal, "total_score", None)

    picks = []
    for i, c in enumerate(candidates, 1):
        code = c.get("股票代码") or c.get("code") or ""
        if not code:
            continue                      # 没有代码的候选无法回看，直接丢弃
        picks.append({
            "pick_date": date,
            "stock_code": str(code),
            "stock_name": c.get("股票名称") or c.get("name") or "",
            "source": source,
            "strategy": c.get("策略") or c.get("strategy") or "",
            "score": c.get("score"),
            "score_coverage": c.get("score_coverage"),
            "price_at_pick": _pick_price(c),
            "rank_in_run": i,
            "market_state": ms_state,
            "market_score": ms_score,
            "factors": c.get("factors") or {},
            "conditions": _pick_conditions(c),
            # B2③（2026-10-18）：本次评分的 PE 口径（eastmoney / snapshot_sina /
            # None=未评）—— 双口径并存时 IC/质量评估必须能按口径过滤
            "pe_source": c.get("pe_source"),
        })
    try:
        return store.insert_screen_picks(picks)
    except Exception as e:                  # noqa: BLE001
        _logger.warning("[选股历史] 落库失败: %s", e)
        return 0


def recent_top_overlap(codes: list, lookback: int = 5, top_n: int = 5,
                       skip_latest: bool = False) -> dict:
    """本次 TOP 与最近 `lookback` 个批次的 TOP 重合度（2026-10-10）。

    ## 为什么要有这个
    用户最初的问题就是"每次选股都趋于同几只"。当时**没有任何机制能让这件事自己暴露**——
    是人工比对 `screen_picks` 才发现的（跨日 TOP5 重合 4/5、同日 5/5）。
    把"重合度"变成每次选股都会算、会显示的一个数，同类问题下次会自己报警，
    而不是等人去翻数据库。

    ## 口径
    - 批次 = `screen_picks` 里同一 `created_at` 分钟内的行（一次 run 一次落库）
    - 只取各批 `rank_in_run <= top_n` 的股票，与传入 `codes` 比交集
    - 返回 `{}`（而不是抛异常）当库为空/查询失败——这只是提示信息，
      **绝不能因为它读不到数据而让选股失败**

    Returns:
        {"batch_count": int, "max_overlap": int, "avg_overlap": float,
         "top_n": int, "threshold": int, "alert": bool, "details": [{"time","overlap","codes"}]}
    """
    codes = [str(c).strip() for c in (codes or []) if str(c).strip()]
    if not codes:
        return {}
    try:
        conn = store.get_connection()
        try:
            rows = conn.execute(
                """SELECT substr(created_at, 1, 16) AS batch, stock_code
                   FROM screen_picks
                   WHERE rank_in_run <= ?
                   ORDER BY created_at DESC""", (top_n,)).fetchall()
        finally:
            conn.close()
    except Exception as e:                      # noqa: BLE001
        _logger.debug("[选股重合度] 查询失败（跳过提示）: %s", e)
        return {}

    batches: dict = {}
    for r in rows:
        batches.setdefault(r["batch"], []).append(r["stock_code"])
    # 最近 lookback 个批次，各与本次 TOP 比交集。
    # `skip_latest=True`：页面渲染**已落库的本次结果**时，最新批次就是"本次自己"，
    # 自比必然 100% 重合 —— 那不是"历史高度重复"，而是**误报**（2026-10-10 冒烟
    # 测试抓到：重开页面就弹"与最近 1 次选股高度重合 1/1"）。调用方在"结果已落库"
    # 的场景必须传 True；在"落库之前"评估的场景（如调度流水线）保持默认 False。
    keys = sorted(batches.keys(), reverse=True)
    if skip_latest and keys:
        keys = keys[1:]
    mine = list(codes)[:top_n]
    details = []
    for k in keys[:max(1, int(lookback))]:
        codes_k = batches[k][:top_n]
        details.append({"time": k, "codes": codes_k,
                        "overlap": len(set(mine) & set(codes_k))})

    if not details:
        return {}
    overlaps = [d["overlap"] for d in details]
    threshold = max(1, len(mine) - 1)           # ≥4/5 视为"高度重合"
    return {
        "batch_count": len(details),
        "max_overlap": max(overlaps),
        "avg_overlap": round(sum(overlaps) / len(overlaps), 2),
        "top_n": len(mine),
        "threshold": threshold,
        "alert": max(overlaps) >= threshold,
        "details": details,
    }


# 回填名次上限（2026-10-22，P0-1）：只给前 N 名补 T+N 收益。
# ⚠️ 为什么不是全量：扩池后每天落库 348 行（116×3）而 `limit=200` ⇒ 每天净增 148，
#    实测积压 617 条（None 490 + partial 127），`ret_20d` 覆盖永远追不上。
#    选股质量评估只需要前排样本有代表性；名次之外的只服务"给深度分析挑候选"。
BACKFILL_TOP_N = 50


def _mark_skip_beyond_top(limit_top: int) -> int:
    """把名次超出 `limit_top` 且尚未回填完的记录标 `skip`（幂等）。

    ⚠️ 用 `skip` 而不是 `failed`：`failed` 会被 `list_screen_picks(backfill_status='failed')`
       重新捞出来重试，而这些记录**不该重试**——它们只是不需要 T+N 收益。
    ⚠️ 同时处理 `partial`：名次之外的记录即使只算了部分窗口也不再补全，
       否则它们会永远留在"待回填"里，积压数字永远降不下来。
    ⚠️ 名次用 `COALESCE(rank_in_run, 999)`：老数据可能没有名次，按"最末"处理。
    """
    store.init_store()
    conn = store.get_connection()
    try:
        cur = conn.execute(
            "UPDATE screen_picks SET backfill_status = 'skip' "
            " WHERE (backfill_status IS NULL OR backfill_status IN ('', 'partial')) "
            "   AND COALESCE(rank_in_run, 999) > ?", (int(limit_top),))
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def backfill_screen_picks(limit: int = 200, include_failed: bool = False,
                          price_fetcher=None, use_lock: bool = True) -> dict:
    """回填补 T+N 收益与同期指数基准（调度每日收盘后 / UI 手动触发）。

    与分析历史回填同一套约定：跨进程互斥（不同 task 名，可并行）、
    数据源失败保持原状态等次日重试、超龄标 failed 不再重试。

    ⚠️ **只回填前 `BACKFILL_TOP_N` 名**（2026-10-22，P0-1）：扩池后每天落库
    348 行（116 只 × 早/中/尾三次），而这里 `limit=200` ⇒ **每天净增 148 行**，
    实测积压 617 条（`None` 490 + `partial` 127），`ret_20d` 覆盖永远追不上
    —— 而那是整个选股质量闭环的地基。
    名次之外的记录标 `skip`（**不是 `failed`**：它们只是不需要 T+20 收益，
    仍然服务于"给深度分析挑候选"这个用途）。
    """
    holder = f"pid{os.getpid()}-{int(time.time())}"
    if use_lock and not store.acquire_task_lock("backfill_screen", holder):
        _logger.info("[选股历史] 回填任务已在其他进程运行中，本次跳过")
        return {"checked": 0, "updated": 0, "done": 0, "partial": 0,
                "failed": 0, "skipped": 0, "errors": [], "locked": True}

    _marked = _mark_skip_beyond_top(BACKFILL_TOP_N)
    if _marked:
        _logger.info("[选股历史] 名次 > %d 的 %d 条候选标 skip（不做 T+N 回填）",
                     BACKFILL_TOP_N, _marked)

    fetcher = price_fetcher or default_price_fetcher
    # ⚠️ `oldest_first=True` + `max_pick_date` 是 2026-10-28 修的两个方向性问题：
    #    ① 原来按 `id DESC` 取"最新 200 条"，而扩池后每天落库 200~440 行
    #       ⇒ 窗口永远只覆盖当天新候选，**老候选排到 T+5/T+20 到期时早已被挤出**。
    #       实测：最老待回填 `id=43`（09-25），窗口下界已到 `id=1781`；
    #       全表 `ret_20d` 覆盖 0/2074 —— 闭环从未拿到过 20 日收益。
    #    ② 不过滤"是否到期"⇒ 取到的全是当天行，必然空收益、白拉行情。
    cutoff = (datetime.now() - timedelta(days=BACKFILL_MIN_AGE_DAYS)).strftime("%Y-%m-%d")
    rows = store.list_screen_picks(limit=limit, only_backfillable=True,
                                   oldest_first=True, max_pick_date=cutoff)
    if include_failed:
        # 手动重试 `failed` 的走原口径（它们本身就是"要人看过再试"的小批量）
        rows += store.list_screen_picks(limit=limit, backfill_status="failed")

    stats = {"checked": len(rows), "updated": 0, "done": 0, "partial": 0,
             "failed": 0, "skipped": 0, "errors": [], "locked": False}
    try:
        _backfill_rows(rows, fetcher, stats)
    finally:
        if use_lock:
            store.release_task_lock("backfill_screen", holder)
    return stats


def _backfill_rows(rows: list, fetcher, stats: dict) -> None:
    """逐行回填（拆出以便加锁/解锁包住整批）"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 指数基准按日期批量预取 + 进程内缓存：
    # 候选通常集中在少数几个选股日，逐条查/逐条拉会把同一天重复取 N 遍
    # （预演实测：20 条候选触发 5 次重复的指数 K 线请求）。同一天只算一次。
    bench_cache: dict = {}
    try:
        bench_cache.update(store.benchmark_returns_batch(
            [r.get("pick_date") for r in rows]))
    except Exception as e:                      # noqa: BLE001
        _logger.debug("[选股历史] 批量预取基准失败，将逐条兜底: %s", e)
    # 兜底时也要避免重复请求同一日期：先记录"已经尝试过实时拉取"的日期
    live_tried: set = set()

    for row in rows:
        pick_id = row["id"]
        pick_date = row.get("pick_date") or ""
        code = row.get("stock_code") or ""
        if not pick_date or not code:
            store.update_screen_pick_backfill(
                pick_id, {"backfill_status": "failed", "backfilled_at": now})
            stats["failed"] += 1
            continue

        try:
            age_days = (datetime.now() - datetime.strptime(pick_date, "%Y-%m-%d")).days
        except ValueError:
            store.update_screen_pick_backfill(
                pick_id, {"backfill_status": "failed", "backfilled_at": now})
            stats["failed"] += 1
            continue

        if age_days > BACKFILL_MAX_AGE_DAYS:
            store.update_screen_pick_backfill(
                pick_id, {"backfill_status": "failed", "backfilled_at": now})
            stats["failed"] += 1
            continue

        try:
            fr = fetch_forward_returns(code, pick_date, price_fetcher=fetcher)
        except Exception as e:                      # noqa: BLE001
            # 数据源失败（限频/熔断/网络）：保持原状态，次日任务自然重试
            stats["errors"].append(f"{code}: {e}")
            stats["skipped"] += 1
            continue

        rets = {k: v for k, v in fr.items() if k.startswith("ret_")}
        if not rets:
            stats["skipped"] += 1
            continue

        fields = dict(rets)
        # 选股时没取到价格的，用回填时的基准价补上（口径：pick_date 之后第一根收盘）
        if not row.get("price_at_pick") and fr.get("base_price") is not None:
            fields["price_at_pick"] = fr["base_price"]
        # 同期指数基准：优先用已有的指数特征库（免费），查不到就实时拉一次指数 K 线。
        # 为什么需要兜底：daily_features 最后 5/20 根还没走完的日子没有 future_ret，
        # 而指数特征库只在被回填时才更新——若不管，近期候选的**超额收益会长期全缺**，
        # 而超额正是"选股有没有用"唯一可信的指标（牛市里闭眼选都"赢"）。
        bench = bench_cache.get(pick_date)
        if bench is None and pick_date not in live_tried:
            live_tried.add(pick_date)
            try:
                bfr = fetch_forward_returns("sh000001", pick_date, price_fetcher=fetcher)
                bench = {f"bench_{k}": bfr[k] for k in ("ret_5d", "ret_20d") if k in bfr}
            except Exception as e:                  # noqa: BLE001
                _logger.debug("[选股历史] 指数基准补取失败: %s", e)
                bench = {}
            bench_cache[pick_date] = bench           # 缓存（包括"确实取不到"）
        fields.update(bench or {})

        if "ret_20d" in fields:
            fields.update({"backfill_status": "done", "backfilled_at": now})
            stats["done"] += 1
        else:
            fields.update({"backfill_status": "partial", "backfilled_at": now})
            stats["partial"] += 1
        store.update_screen_pick_backfill(pick_id, fields)
        stats["updated"] += 1

    if stats["checked"]:
        _logger.info("[选股历史] T+N 回填: %s", stats)
