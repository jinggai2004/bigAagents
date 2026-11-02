# -*- coding: utf-8 -*-
"""因子实验室的服务层（给 web 路由用，薄封装 + 统一信封）。

为什么这里几乎没有业务逻辑
──────────────────────────
回测/校验/落库全在 `core/factor_lab.py` 与 `core/factor_dsl.py`。这一层只做
三件事：包信封、把异常翻成人话、在没装 pandas 面板时给出可执行的提示。
业务逻辑一旦漏到 web 层，脚本与页面就会各算一个数——本项目踩过。

⚠️ 这里是**人**的通道（页面上点「回测」），不套两阶段确认；但"应用"这一步
被 `factor_lab.set_status` 挡住：没过检验的因子不许 applied，分数类因子必须先过
A/B 才能启用。判据在 core，不在这里重复写一遍。
"""
from __future__ import annotations

import logging

from web.server import envelope

_logger = logging.getLogger("web.factor_lab")


def state() -> dict:
    """页面首屏：面板概况 + 字段清单（带覆盖率）+ 已登记因子。

    ⚠️ **首屏绝不构建面板**。面板分两档（`core/factor_lab.py` 注释里有实测拆解）：
    量价档 0.4 秒、财务档多 21.5 秒（要读 5,582 个 per-code parquet）。
    打开页面只是想看看有哪些字段，为这点信息付 20 多秒没有道理——
    量价档覆盖率直接用（0.4s），财务字段在财务档还没建时如实返回 `coverage=null`
    （"没加载" ≠ "是空的"，页面显示"待回测"）。
    """
    from core import factor_lab as FL
    try:
        info = FL.panel_info(allow_load=False)      # None=还没构建，不触发
        cat = FL.catalog(full=True)                 # 财务未就绪时财务字段给 None
    except FileNotFoundError as e:
        return envelope.fail("missing_data", str(e))
    except Exception as e:                            # noqa: BLE001
        _logger.exception("因子实验室面板读取失败")
        return envelope.fail("panel_error", f"面板读取失败：{e}")

    from db import store
    rows = []
    for r in store.list_user_factors():
        m = r.get("metrics") or {}
        rows.append({
            "factor_id": r["factor_id"], "name": r["name"], "expr": r["expr"],
            "kind": r["kind"], "status": r["status"],
            "direction": r["direction"], "gate_pct": r["gate_pct"],
            "note": r["note"], "created_at": r["created_at"],
            "metrics": {k: m.get(k) for k in
                        ("ic_mean", "t_stat", "ic_ir", "coverage", "n_months",
                         "n_trials", "usable", "verdict", "blockers", "by_year",
                         "year_consistency", "redundancy", "direction_label",
                         "q5_q1_spread", "ran_at")} if m else {},
        })
    return envelope.ok({
        # `panel=None` ⇒ 连量价档都还没建（首次使用）。前端要显示"首次回测时构建"，
        # 不能当成错误。
        "panel": info,
        "panel_ready": info is not None,
        "fin_ready": bool(info and info.get("has_fin")),
        "catalog": cat,
        "factors": rows,
        "thresholds": {"min_abs_ic": FL.MIN_ABS_IC, "min_abs_t": FL.MIN_ABS_T,
                       "min_year_ratio": FL.MIN_YEAR_RATIO,
                       "min_coverage": FL.MIN_COVERAGE,
                       "duplicate_rho": FL.DUPLICATE_RHO,
                       "redundant_rho": FL.REDUNDANT_RHO},
        # `available=None` = 财务字段还没加载（不是"不可用"），单独分组别混进空字段
        "empty_fields": [c["name"] for c in cat if c.get("available") is False],
        "pending_fields": [c["name"] for c in cat if c.get("available") is None],
    })


def submit(action: str, payload: dict) -> dict:
    """统一写入口。`action` 见下方分支，字段见各分支注释。"""
    from core import factor_dsl as D
    from core import factor_lab as FL
    p = payload or {}
    expr = str(p.get("expr") or "").strip()

    try:
        # ── 静态校验：不跑数据，秒回。给"这行字能不能用"的即时反馈 ──
        if action == "validate":
            if not expr:
                return envelope.ok({"valid": False, "error": "表达式为空"})
            try:
                return envelope.ok({"valid": True, **D.describe(expr)})
            except D.FactorExprError as e:
                return envelope.ok({"valid": False, "error": str(e)})

        # ── 预览：最近一个月的截面取值，用来确认量纲与方向 ──
        if action == "preview":
            return envelope.ok(FL.preview(expr, limit=int(p.get("limit") or 12)))

        # ── 回测：纯读，不落库。页面可以随便点，每点一次都算一次试验（见下） ──
        if action == "backtest":
            return envelope.ok(FL.backtest(expr, str(p.get("name") or "临时因子")))

        # ── 保存：回测 + 落库（status 由可用性决定，不让人手工改） ──
        if action == "save":
            out = FL.register(expr, str(p.get("name") or ""),
                              kind=str(p.get("kind") or "gate"),
                              gate_pct=float(p.get("gate_pct") or 0.1),
                              note=str(p.get("note") or ""),
                              factor_id=p.get("factor_id") or None)
            return envelope.ok(out)

        # ── 状态：draft/validated/rejected/applied/disabled ──
        if action == "status":
            fid = str(p.get("factor_id") or "")
            if not fid:
                return envelope.fail("invalid", "缺少 factor_id")
            row = FL.set_status(fid, str(p.get("status") or ""))
            return envelope.ok({"factor": row,
                                "message": f"{row.get('name')} → {row.get('status')}"})

        if action == "delete":
            from db import store
            fid = str(p.get("factor_id") or "")
            ok = store.delete_user_factor(fid)
            return envelope.ok({"deleted": bool(ok)}) if ok else \
                envelope.fail("not_found", "因子不存在")

        return envelope.fail(
            "invalid",
            f"未知 action：{action}（可用 validate/preview/backtest/save/status/delete）")
    except D.FactorExprError as e:
        return envelope.fail("expr_invalid", str(e))
    except ValueError as e:
        return envelope.fail("invalid", str(e))
    except FileNotFoundError as e:
        return envelope.fail("missing_data", str(e))
    except Exception as e:                            # noqa: BLE001
        _logger.exception("因子实验室 %s 失败", action)
        return envelope.fail("error", f"{type(e).__name__}: {e}")
