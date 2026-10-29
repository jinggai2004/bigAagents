"""
定时调度入口：启动 APScheduler 常驻进程。

用法:
    python scripts/run_scheduler.py                       # 前台常驻运行
    python scripts/run_scheduler.py --once                # 立即执行一次盘后全流程(rule)后退出
    python scripts/run_scheduler.py --list-jobs           # 列出全部内置 job_id（P0-6）
    python scripts/run_scheduler.py --job snapshot_flow   # 只重跑一个阶段（当天补救）
    python scripts/run_scheduler.py --job snapshot_flow --date 2026-10-16
    python scripts/run_scheduler.py --job daily_review --force   # 非交易日强制执行

为什么有 `--job`（P0-6，2026-10-18）：原来只有 `--once`（跑整个盘后全流程），
15:40 快照挂了 / 15:45 复盘崩了，当天就没有第二次机会，回填还要等下一个交易日。
按 job_id 精确重放 + `task_locks` 防重入（避免和长驻调度进程撞车）。
"""
import sys
import os
import time
import argparse
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scheduler import (create_scheduler, run_job, next_run_times,
                            pending_run_times, custom_job_ids,
                            reload_custom_jobs, JOBS, JOB_IDS)
from core import single_instance


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join("logs", f"scheduler_{datetime.now():%Y%m%d}.log"),
                                encoding="utf-8"),
        ],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="bigAagents 定时调度器")
    parser.add_argument("--once", action="store_true", help="立即执行一次盘后全流程后退出")
    parser.add_argument("--list-jobs", action="store_true",
                        help="列出全部内置 job_id / mode / 说明（重放时按 id 指定）")
    parser.add_argument("--job", metavar="JOB_ID",
                        help="只重跑指定阶段（当天补救）。见 --list-jobs")
    parser.add_argument("--date", metavar="YYYY-MM-DD",
                        help="数据日期（**仅 snapshot_flow 支持**；其余阶段按当天运行）")
    parser.add_argument("--force", action="store_true",
                        help="绕过交易日检查（非交易日重放会产生与交易日不可区分的记录，谨慎）")
    parser.add_argument("--no-budget", action="store_true",
                        help="人工补跑：不扣 AI 联赛当日额度（否则会把 15:58 正式决策轮挤掉）。"
                             "仅对 ai_daily_round 生效")
    args = parser.parse_args()

    setup_logging()
    _logger = logging.getLogger(__name__)

    # ── P0-6：列出任务 ──
    if args.list_jobs:
        print(f"{'JOB_ID':22s} {'MODE':18s} 说明")
        for jid, _kw, mode, desc in JOBS:
            print(f"{jid:22s} {mode:18s} {desc}")
        return 0

    # ── P0-6：按 job_id 重放（当天可补救）──
    if args.job:
        hit = next(((j[0], j[2], j[3]) for j in JOBS if j[0] == args.job), None)
        if hit is None:
            _logger.error("未知 job_id=%r。可用：%s", args.job, ", ".join(JOB_IDS))
            return 2
        jid, mode, desc = hit
        # ── P0-3（2026-10-22）：常驻存活时默认拒绝手动重放 ──
        # ⚠️ 为什么靠心跳而不是锁：常驻任务**各自硬编码锁名**
        #    （`evening_analysis` / `backfill_screen` …），与这里的 `manual_<id>`
        #    **不同名 ⇒ 那道锁形同虚设**（实测可并发跑同一阶段）。
        #    统一在 `run_job` 里加 `job:<id>` 锁又会误伤 `hourly_news` 这类
        #    一天多次的任务（TTL 会挡住第二次，除非改成显式释放，改动面更大）。
        # ⇒ 用最直接的判据：常驻活着就要求人显式确认（`--force`）。
        #    `--force` 原本只表示"绕过交易日检查"，这里共用同一个
        #    「我知道有风险」的语义，提示文案会说明原因。
        if not args.force:
            import core.scheduler_health as _health
            _hb = _health.read_heartbeat()
            if _hb and _health._process_alive(_hb.get("pid")):
                _logger.error(
                    "常驻调度进程 (pid=%s) 正在运行，重放 %s 可能与之并发执行同一阶段。"
                    "\n  确认常驻当前没在跑该阶段后，加 --force 再试。", _hb.get("pid"), jid)
                return 4
        from db import store as _db
        lock_name = f"manual_{jid}"
        holder = f"manual:{os.getpid()}"
        # task_locks 防重入：长驻调度进程若正在跑同一阶段，这里不能再叠一遍
        if not _db.acquire_task_lock(lock_name, holder, ttl_seconds=7200):
            _logger.error("阶段 %s 正在运行（锁 %s 被占用），本次退出", jid, lock_name)
            return 3
        try:
            if jid == "snapshot_flow" and args.date:
                # 唯一支持显式日期的阶段：快照是"逐日一个截面"，补哪一天是明确的
                from core.scheduler import save_daily_market_snapshot
                out = save_daily_market_snapshot(args.date)
                _logger.info("═══ [手动重放 snapshot_flow @ %s] %s ═══", args.date, out)
                ok = bool(out.get("ok"))
            else:
                if args.date:
                    _logger.warning("--date 仅对 snapshot_flow 生效；%s 按**当天**运行"
                                    "（各阶段内部取当前日期）", jid)
                run_job(mode, f"手动重放[{jid}]", force=args.force, job_id=jid,
                        no_budget=args.no_budget)
                ok = True
        finally:
            _db.release_task_lock(lock_name, holder)
        _logger.info("═══ 手动重放[%s] %s ═══", jid, "完成" if ok else "失败（详见上方日志）")
        return 0 if ok else 1

    if args.once:
        # ⚠️ 不能用 JOBS[-1]：JOBS 末尾是"盘中异动预警"(mode=watch)，追加任务后
        # --once 就悄悄变成了跑盯盘而不是跑全流程（且不报错）。按 job_id 精确取。
        mode = next(m for jid, _kw, m, _d in JOBS if jid == "afternoon_full")
        run_job(mode, "手动单次全流程", force=args.force, no_budget=args.no_budget)
        return 0

    # 单实例保护：多个调度进程会重复执行选股/回测（叠加限频与重复落库）
    if not _acquire_singleton():
        _logger.error("已有调度器在运行（见 temp/scheduler_heartbeat.json），本次退出。"
                      "如需重启，请先运行 停止bigAagents.bat")
        return 1

    scheduler = create_scheduler()

    # 防新式待机：让机器别在盘中睡过去。
    # 为什么需要（2026-10-14 实翻）：`powercfg /change standby-timeout-ac 0`
    # （"永不待机"）**已生效**，但当天 13:32:54 系统仍进入新式待机、15:03:14 才退出，
    # 同期 `盘中异动预警` 出现 **90 分钟缺口**（Kernel-Power 506/507 与缺口完全吻合）。
    # ⇒ 该机型**熄屏本身就是入待机路径**，且 `SUB_BUTTONS` 里没有 LIDACTION
    #   （合盖项未暴露）⇒ 改系统设置这条路走不通，必须由进程自己持有电源请求。
    # best-effort：不支持/失败只记一行日志，绝不让它成为"调度器起不来"的新理由。
    _ka = None
    try:
        from core import keep_awake
        _ka = keep_awake
        if keep_awake.acquire():
            _logger.info("已启用防待机保活：%s", keep_awake.describe()["reason"])
        else:
            _logger.warning("防待机保活未启用：%s", keep_awake.describe()["reason"])
    except Exception as e:                      # noqa: BLE001
        _logger.warning("防待机保活初始化失败（不影响调度）: %s", e)

    # ⚠️ 顺序不能反。APScheduler 未启动时 Job **没有** next_run_time 属性，
    # 原先"先打印 next_run_times() 再 start()"→ 启动即 AttributeError →
    # 调度器从未跑起来，所有定时任务一个都没执行过（详见 next_run_times 的注释）。
    # 因此：先用纯计算预估（pending_run_times，不依赖调度器状态），
    # 等 start() 之后再用真实值覆盖。
    _logger.info("═══ bigAagents 调度器启动，共 %d 个任务（内置 %d + 自定义 %d）═══",
                 len(scheduler.get_jobs()), len(JOBS), len(custom_job_ids(scheduler)))
    for name, next_time in pending_run_times(scheduler):
        _logger.info("预计执行: %s @ %s", name, next_time)

    # 心跳：独立线程定期写心跳文件，供「系统配置」页与自检脚本判断调度器是否活着。
    # 有了这个，"调度器挂了却没人知道"就变成可观测状态。
    # 自定义任务单独用 custom_job_ids 上报：它们不在 JOB_IDS 里，
    # 混进去会让健康检查的"旧代码"比对永远报"多出来的任务"（狼来了）。
    _start_heartbeat(scheduler)

    # 自定义任务热加载：库里的增删改 60 秒内生效，不用重启进程。
    _start_custom_job_reloader(scheduler)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        _logger.info("调度器已停止")
    finally:
        if _ka is not None:
            try:
                _ka.release()
            except Exception:                   # noqa: BLE001
                pass
        _clear_heartbeat()
    return 0


# ⚠️ 用 `Global\` 而不是 `Local\`（2026-10-22 修正）：`Local\` 是**会话内**命名空间，
#    桌面双击（session 1）与计划任务（session 0）各起一个时**互相看不到** ⇒ 仍可双开。
#    `Global\` 是全机命名空间。代价：极少数受限账户可能无 `SeCreateGlobalPrivilege`，
#    此时 CreateMutexW 返回 NULL ⇒ 走"创建失败就放行"的分支（可用性优先，不会起不来）。
# 实现已抽到 `core/single_instance`（2026-10-23）：watchdog 也踩同一个 TOCTOU 坑，
# 两处各写一份必然分叉。这里只保留"调度器专用的互斥名 + 向后兼容的入口"，
# 原有调用方（`_acquire_singleton`、测试里的 `R._acquire_os_mutex(name)`）一行不用改。
_MUTEX_NAME = single_instance.mutex_name("scheduler")
_MUTEX_HANDLES = single_instance._MUTEX_HANDLES   # 兼容：测试/诊断会读它


def _acquire_os_mutex(name: str = None) -> bool:
    """OS 级命名互斥：拿到 = 本进程是唯一实例（实现见 `core/single_instance`）。

    ⚠️⚠️ 历史：2026-10-21 三个实例并发跑了三遍选股（各 3211s），根因是
       "读心跳判 pid"的**检查-后-启动**窗口 —— 心跳本来就是启动成功后才写的。
       换成内核互斥后，进程一退出自动释放，没有窗口。
    """
    return single_instance.acquire_os_mutex(name or _MUTEX_NAME)


def _acquire_singleton() -> bool:
    """单实例检查：已有活着的调度进程就不要再起一个。

    为什么需要：选股一次 4~5 分钟（批次 16 实测）并持续拉数据源，两个调度进程
    并行仍会重复跑同一批任务、叠加限频，还会写出重复的候选/分析记录。

    ⚠️ 2026-10-21：判定手段从"读心跳"改为"OS 命名互斥"（见 `_acquire_os_mutex`）。
       拿到互斥权 ⇒ 可以断言**没有其他实例** ⇒ 残留心跳一定是旧进程留下的，
       直接清掉（避免页面显示已死进程的 pid）。
    """
    import core.scheduler_health as health
    if not _acquire_os_mutex():
        return False
    hb = health.read_heartbeat()
    if hb:
        logging.getLogger("scheduler").info(
            "[单实例] 清理残留心跳（pid=%s，本进程 pid=%s）", hb.get("pid"), os.getpid())
        health.clear_heartbeat()
    return True


def _start_heartbeat(scheduler=None, interval: int = 60):
    """后台线程每 interval 秒刷新心跳文件（进程退出时自动停）"""
    import threading
    import core.scheduler_health as health

    stop = threading.Event()

    started_at = time.time()          # 进程启动时刻（判"改动是否已生效"唯一可靠依据）

    def _beat():
        while not stop.wait(interval):
            try:
                health.write_heartbeat(pid=os.getpid(), job_ids=JOB_IDS,
                                       custom_job_ids=custom_job_ids(scheduler),
                                       started_at=started_at)
            except Exception:                   # noqa: BLE001
                pass                            # 心跳失败不能影响调度主循环

    # 带上 job_ids：健康检查据此判断"跑的是不是最新代码"（只带个数看不出少的是哪个）
    health.write_heartbeat(pid=os.getpid(), job_ids=JOB_IDS,
                           custom_job_ids=custom_job_ids(scheduler),
                           started_at=started_at)
    t = threading.Thread(target=_beat, name="scheduler-heartbeat", daemon=True)
    t.start()
    return stop


def _start_custom_job_reloader(scheduler, interval: int = 60):
    """后台线程每 interval 秒把库里的自定义任务同步到调度器。

    这就是"改完不用重启"的实现：界面上改时间/开关 → 下一个 60 秒窗口生效。
    任何异常都在 reload_custom_jobs 内部兜住——同步失败绝不能影响内置任务。
    """
    import threading

    _logger = logging.getLogger(__name__)
    stop = threading.Event()

    def _loop():
        while not stop.wait(interval):
            result = reload_custom_jobs(scheduler)
            if result.get("changed"):
                _logger.info("自定义定时任务已热加载：%s", result)
            for bad in (result.get("invalid") or []):
                _logger.warning("自定义定时任务 %s 定义有误，已跳过：%s",
                                bad.get("job_id"), bad.get("error"))

    t = threading.Thread(target=_loop, name="scheduler-custom-reload", daemon=True)
    t.start()
    return stop



def _clear_heartbeat():
    try:
        import core.scheduler_health as health
        health.clear_heartbeat()
    except Exception:                           # noqa: BLE001
        pass


if __name__ == "__main__":
    sys.exit(main())
