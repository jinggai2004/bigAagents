"""每日复盘页面 — 复盘正文浏览 / 历史存档回看 / 预测跟踪命中率 / 手动生成

为什么单独一个页面（2026-09-28）：
复盘是「每个交易日都要看」的产出，此前它的唯一入口藏在「系统配置」页最底部的
一个 expander 里 —— 那一页的职责是管 API Key 和调度，把高频产物放在低频配置页
底部，结果是产物明明生成了（output/daily_reports/latest.md）用户却在界面上找不到。
页面职责要按「使用频率」而非「实现归属」来划分。
"""

import sys, os, re, glob
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 报告目录支持环境变量覆盖：测试用临时目录（同 BIGA_UI_POLL_SECONDS 的既有约定），
# 部署时也可指向别的盘。默认仍是项目内的 output/daily_reports。
REPORT_DIR = os.environ.get("BIGA_REVIEW_DIR") or os.path.join(PROJECT_ROOT, "output", "daily_reports")
LATEST_NAME = "latest.md"

st.set_page_config(page_title="每日复盘", page_icon="📰", layout="wide")
st.title("📰 每日复盘")
st.caption("收盘后自动生成（交易日 15:45）：大盘概览 → 资金与板块 → 自选股跟踪 → "
           "历史相似日 → 消息面 → 明日观察点（次日自动核对，形成命中率闭环）")


# ================================================================
# 历史报告清单（review_YYYYMMDD.md，倒序）
# ================================================================
def list_reports() -> list[dict]:
    """列出所有历史复盘报告。返回 [{date, path, mtime, size}]，按日期倒序。"""
    out = []
    for p in glob.glob(os.path.join(REPORT_DIR, "review_*.md")):
        m = re.search(r"review_(\d{8})\.md$", os.path.basename(p))
        if not m:
            continue
        d = m.group(1)
        out.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "path": p,
            "mtime": os.path.getmtime(p),
            "size": os.path.getsize(p),
        })
    out.sort(key=lambda r: r["date"], reverse=True)
    return out


def _latest_path() -> str:
    return os.path.join(REPORT_DIR, LATEST_NAME)


# ================================================================
# 顶部：概览指标
# ================================================================
reports = list_reports()
today_str = datetime.now().strftime("%Y-%m-%d")
has_today = any(r["date"] == today_str for r in reports)

c1, c2, c3, c4 = st.columns(4)
c1.metric("最新复盘日期", reports[0]["date"] if reports else "—")
c2.metric("生成时间",
          datetime.fromtimestamp(reports[0]["mtime"]).strftime("%m-%d %H:%M") if reports else "—")
c3.metric("历史报告数", len(reports))

# 预测跟踪命中率
obs_stats = {}
try:
    from db import store
    obs_stats = store.observation_stats() or {}
except Exception as e:                                  # noqa: BLE001 - 统计失败不影响看正文
    obs_stats = {"_error": f"{type(e).__name__}: {e}"}
if obs_stats.get("命中率") is not None:
    c4.metric("观察点命中率(近30天)", f"{obs_stats['命中率']:.0%}",
              delta=f"{obs_stats.get('达成', 0)}达成/{obs_stats.get('verified', 0)}核对",
              delta_color="off")
else:
    c4.metric("观察点命中率(近30天)", "—",
              delta="尚无核对记录" if not obs_stats.get("_error") else obs_stats["_error"][:40],
              delta_color="off")

if reports and not has_today:
    st.warning("今日（%s）尚未生成复盘。交易日 15:45 由调度任务自动生成；"
               "若刚刚启动调度进程，可能已错过该时间点，可点击下方按钮手动生成。" % today_str)


# ================================================================
# 生成 / 选择
# ================================================================
st.divider()
gen_col1, gen_col2 = st.columns([1, 3])
with gen_col1:
    gen_btn = st.button("📝 立即生成复盘", width="stretch", type="primary")
with gen_col2:
    st.caption("流程：采集全市场资金流快照 → 市场检测14指标 → 自选股/行业资金/要闻/相似日 → "
               "LLM 撰写 → 预测跟踪落库。约 30-60 秒；若当日快照尚未采集，需先拉全市场，可能数分钟。")
if gen_btn:
    with st.spinner("正在生成复盘（未采集快照时会先拉全市场，请耐心等待）..."):
        try:
            from core.review_agent import generate_review
            result = generate_review(push=False)
            if "error" in result:
                st.error(f"生成失败: {result['error']}")
            else:
                tracking = result.get("tracking") or {}
                if tracking:
                    st.success(f"已生成: {result['report_path']} ｜ 核对 {tracking.get('核对条数', 0)} 条、"
                               f"新增观察点 {tracking.get('新增观察点', 0)} 条")
                else:
                    st.success(f"已生成: {result['report_path']}")
                reports = list_reports()
                st.rerun()
        except Exception as e:                          # noqa: BLE001 - 页面级兜底要带异常类型
            st.error(f"生成异常: {type(e).__name__}: {e}")


# ================================================================
# 今日决策三块（实时直读数据库；不依赖 LLM，也不依赖报告是否已生成）
# ================================================================
# 为什么放在这里（2026-10-10 用户反馈"复盘较简略，看不出今日操作及分析是否正确"）：
#   复盘报告是**事后文本**（且要等 LLM 生成），而"今天选了什么/分析了什么/持仓距
#   止损多远"是**即时事实**。两者分开：即使今天的报告还没生成，这三块也能立刻看到。
#   与报告开头的三章同源（`core.decision_digest`），数字不经模型。
st.divider()
st.subheader("📋 今日决策（实时 · 直接读库）")
st.caption("与复盘报告开头的三章同源（`core.decision_digest`）。即使今日报告尚未生成，"
           "这里也能看到今天选了什么、分析了什么、持仓距止损/目标多远。")
try:
    from core.decision_digest import build_digest
    from core.review_agent import render_decision_sections
    st.markdown(render_decision_sections(build_digest()))
except Exception as e:                                  # noqa: BLE001 - 不影响看报告正文
    st.warning(f"决策汇总读取失败：{type(e).__name__}: {e}")


# ================================================================
# 正文
# ================================================================
if not reports:
    st.info("尚无复盘报告。调度任务「每日复盘」每个交易日 15:45 自动生成，"
            "或点击上方「立即生成复盘」。\n\n"
            "提示：该任务需要调度进程在 15:45 时处于运行状态，"
            "否则当日窗口会被错过（可到「系统配置」页查看运行健康）。")
else:
    # 「最新（latest.md）」这一项只在文件真的存在时才出现。
    # 否则会出现：latest.md 不在、但历史存档在，用户一进页面只看到一句
    # "文件不存在"，还得自己去下拉框里翻——和"看不到复盘"是同一类问题。
    latest_exists = os.path.exists(_latest_path())
    labels, paths, descs = [], [], []
    if latest_exists:
        labels.append("📌 最新（latest.md）")
        paths.append(_latest_path())
        descs.append(f"latest.md → {reports[0]['date']}")
    for r in reports:
        labels.append(f"{r['date']}  " + ("（今日）" if r["date"] == today_str else ""))
        paths.append(r["path"])
        descs.append(r["date"])

    choice = st.selectbox("选择报告", labels, index=0)
    idx = labels.index(choice)
    show_path, show_label = paths[idx], descs[idx]

    if not os.path.exists(show_path):
        st.error(f"文件不存在：{show_path}")
    else:
        text = open(show_path, encoding="utf-8", errors="ignore").read()
        st.caption(f"来源：`output/daily_reports/{os.path.basename(show_path)}` ｜ "
                   f"{len(text):,} 字符 ｜ {show_label}")
        st.markdown(text)

    with st.expander("📊 预测跟踪详情（观察点核对闭环）"):
        if obs_stats.get("_error"):
            st.warning(f"统计读取失败：{obs_stats['_error']}")
        else:
            if obs_stats.get("verified"):
                st.markdown(
                    f"- 近 30 天已核对 **{obs_stats['verified']}** 条："
                    f"达成 {obs_stats.get('达成', 0)} ｜ 部分达成 {obs_stats.get('部分达成', 0)} ｜ "
                    f"未发生 {obs_stats.get('未发生', 0)} ｜ 无法验证 {obs_stats.get('无法验证', 0)}\n"
                    f"- 命中率（达成+部分达成）＝ **{obs_stats.get('命中率'):.0%}**")
            else:
                st.caption("近 30 天尚无已核对的观察点。复盘生成的「明日观察点」会在次日复盘中被逐条核对，"
                           "攒够数据后这里会给出命中率。")
            try:
                from db import store
                pending = store.get_unverified_observations()
                if pending:
                    st.markdown(f"**待核对观察点（{len(pending)} 条）**")
                    for o in pending:
                        st.markdown(f"- `{o.get('review_date')}` 【{o.get('kind') or '观察点'}】{o.get('content')}")
                else:
                    st.caption("当前无待核对观察点。")
            except Exception as e:                      # noqa: BLE001
                st.caption(f"待核对观察点读取失败：{type(e).__name__}: {e}")

st.divider()
st.caption("⚠️ 复盘由 AI 基于当日真实数据撰写，仅供参考，不构成投资建议。"
           "报告中的数字口径与「数据截止」以报告正文标注为准。")
