"""任务控制信号的零依赖定义

为什么单独放这里：
    `TaskCancelled` 需要同时被 UI 层（core/ui_tasks.py）和业务层（core/screener.py）
    使用。业务层反向依赖 UI 基建会破坏 `db ← data ← core ← ui` 的分层，也会让
    "跑一个选股任务"必须把 Streamlit 相关模块一起导入进来。
    异常类本身不依赖任何东西，放在 utils 层两边都能干净地 import。

    core/ui_tasks.py 仍 re-export 本类，历史 import 路径不受影响。
"""


class TaskCancelled(Exception):
    """任务被用户取消（在进度点抛出，由工作线程捕获）"""
