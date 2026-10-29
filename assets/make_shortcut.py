# -*- coding: utf-8 -*-
"""在桌面创建 `bigAagents 面板.lnk`（纯文件写入，不依赖 COM/管理员权限）。

     py -3 assets/make_shortcut.py            # 建（或覆盖）桌面快捷方式
     py -3 assets/make_shortcut.py --check    # 只写到 temp/ 并自检，不碰桌面

## 为什么手写 .lnk 而不是 WScript.Shell

`New-Object -ComObject WScript.Shell` 在受限环境里会被安全策略拦（COM 实例化可执行
任意代码）；符号链接又要管理员权限。而 `.lnk` 本身只是一个按 MS-SHLLINK 排布的
二进制文件，写文件不需要任何特权。

## ⚠️⚠️ `LinkTargetIDList` 是**必需**的（2026-10-25 实测）

第一版只写了 `LinkInfo`（绝对路径）+ StringData + TerminalBlock，省掉了
`LinkTargetIDList`。结构上"看起来"完全正确（HeaderSize 0x4C、CLSID 对、LinkInfo
各偏移自洽、LocalBasePath 能解出中文路径、TerminalBlock 也在），但 **Windows 直接
拒收**：双击/`Start-Process` 报"没有应用程序与此操作的指定文件有关联" —— 也就是
"这个文件不是快捷方式"，而不是"目标找不到"。

隔离实验（做一次，5 分钟，别再靠猜）：拿一个**安装程序生成的正品 .lnk**，
① 原样 → 能启动；② 只清掉 `HasLinkTargetIDList` 标志位、把那段字节删掉，**其余字节
一个不动** → 报同一个错。变量只剩这一个，结论无法反驳。
⇒ **IDList 不是提示，是解析的必要条件**，`LinkInfo` 只是缓存提示。所以现在必须生成 IDList：

    ItemID[0]  我的电脑（固定常量：size=0x14, type=0x1F, CLSID {20D04FE0-…}）
    ItemID[1]  卷      （size=0x19, type=0x2F, "D:\\" + 补齐）
    ItemID[2..] 每一级目录/文件（type=0x31 目录 / 0x32 文件）

文件系统条目用**旧式 ANSI 名**格式即可（不必写 `0xBEEF0004` 扩展块）：

    size(2)  type(1)  unknown(1)=0  fileSize(4)  dosDate(4)  attrs(2)  ANSI名+NUL

⚠️ 每个 ItemID 总长必须是**偶数**（Windows 自己生成的也都是偶数）。名字按系统
代码页（`mbcs`）编码，中文目录名必须走这个，否则路径解析不出来。

⚠️ 目录层级必须与目标真实路径一致：IDList 是逐级"命名空间项"，跳级会让 Windows
去解析一个不存在的父项。
"""
from __future__ import annotations

import os
import struct
import sys
import time

# ---- ShellLinkHeader 标志位 ----
_HAS_IDLIST = 0x00000001
_HAS_LINK_INFO = 0x00000002
_HAS_NAME = 0x00000004
_HAS_WORKING_DIR = 0x00000010
_HAS_ICON_LOCATION = 0x00000040
_IS_UNICODE = 0x00000080

_CLSID = bytes.fromhex("0114020000000000C000000000000046")   # 00021401-0000-0000-C000-000000000046

# ItemID：我的电脑（CLSID {20D04FE0-3AEA-1069-A2D8-08002B30309D}），固定 20 字节
_ID_MY_COMPUTER = bytes.fromhex("14001F50E04FD020EA3A6910A2D808002B30309D")


def _ansi(s: str) -> bytes:
    """.lnk 里的 ANSI 字段按系统代码页编码（中文路径/名字必须走这个）。"""
    try:
        return s.encode("mbcs")
    except (LookupError, UnicodeEncodeError):
        return s.encode("utf-8")


def _dos_time(ts: float) -> int:
    """FILETIME → DOS 日期时间（高 16 位日期 / 低 16 位时间）。"""
    lt = time.localtime(ts)
    year = min(max(lt.tm_year, 1980), 2107)
    d = ((year - 1980) << 9) | (lt.tm_mon << 5) | lt.tm_mday
    t = (lt.tm_hour << 11) | (lt.tm_min << 5) | (lt.tm_sec // 2)
    return (d << 16) | t


def _volume_item(drive: str) -> bytes:
    """卷条目：type 0x2F，"D:\\" 后补 0 到 22 字节（合计 25 = 偶数）。"""
    name = _ansi(drive.rstrip("\\") + "\\")
    return b"\x19\x00\x2f" + name.ljust(22, b"\x00")[:22]


def _entry_item(name: str, is_dir: bool, size: int = 0, mtime: float | None = None) -> bytes:
    """一级目录/文件的 ItemID（旧式 ANSI 名，无 0xBEEF0004 扩展块）。"""
    nm = _ansi(name) + b"\x00"
    body = (b"\x00"                                  # unknown
            + struct.pack("<I", size & 0xFFFFFFFF)   # fileSize
            + struct.pack("<I", _dos_time(mtime or time.time()))
            + struct.pack("<H", 0x10 if is_dir else 0x20)
            + nm)
    total = 3 + len(body)
    if total % 2:                                    # ItemID 总长必须偶数
        body += b"\x00"
    return struct.pack("<H", 3 + len(body)) + (b"\x31" if is_dir else b"\x32") + body


def build_idlist(target: str) -> bytes:
    """按目标的真实路径逐级生成 LinkTargetIDList。"""
    target = os.path.abspath(target)
    drive = os.path.splitdrive(target)[0]            # "D:"
    if not drive:
        raise ValueError(f"不是绝对路径（不支持 UNC）: {target}")
    parts = os.path.relpath(target, drive + os.sep).split(os.sep)

    out = _ID_MY_COMPUTER + _volume_item(drive)
    for i, p in enumerate(parts):
        is_last = i == len(parts) - 1
        if is_last:
            st = os.stat(target)
            out += _entry_item(p, is_dir=False, size=st.st_size, mtime=st.st_mtime)
        else:
            sub = os.path.join(drive + os.sep, *parts[:i + 1])
            mt = os.stat(sub).st_mtime if os.path.exists(sub) else time.time()
            out += _entry_item(p, is_dir=True, mtime=mt)
    return out + b"\x00\x00"                          # ItemIDList 终止符


def _string_data(s: str) -> bytes:
    """StringData 项：CountCharacters(2, 含结尾 NUL) + UTF-16LE 内容 + NUL。"""
    return struct.pack("<H", len(s) + 1) + s.encode("utf-16-le") + b"\x00\x00"


def _link_info(target: str) -> bytes:
    """LinkInfo：只带 VolumeID + LocalBasePath（本地绝对路径，ANSI）。"""
    volume_id = struct.pack("<IIII", 0x11, 3, 0, 0x10) + b"\x00"   # 17 字节，DriveType=3(FIXED)
    base = _ansi(target) + b"\x00"
    suffix = b"\x00"                                              # 无公共后缀
    header_size = 0x1C
    vol_off = header_size
    base_off = vol_off + len(volume_id)
    suffix_off = base_off + len(base)
    total = suffix_off + len(suffix)
    head = struct.pack("<IIIIII", total, header_size, 1, vol_off, base_off, 0)
    return head + struct.pack("<I", suffix_off) + volume_id + base + suffix


def build_lnk(target: str, working_dir: str | None = None,
              icon: str | None = None, desc: str = "",
              show_command: int = 1) -> bytes:
    target = os.path.abspath(target)
    working_dir = os.path.abspath(working_dir) if working_dir else os.path.dirname(target)
    icon = icon or target

    flags = (_HAS_IDLIST | _HAS_LINK_INFO | _HAS_NAME | _HAS_WORKING_DIR
             | _HAS_ICON_LOCATION | _IS_UNICODE)

    header = (
        struct.pack("<I", 0x4C)
        + _CLSID
        + struct.pack("<I", flags)
        + struct.pack("<I", 0x20)                        # FILE_ATTRIBUTE_ARCHIVE
        + struct.pack("<Q", 0) + struct.pack("<Q", 0) + struct.pack("<Q", 0)
        + struct.pack("<I", os.path.getsize(target) & 0xFFFFFFFF)
        + struct.pack("<i", 0)                           # IconIndex
        + struct.pack("<i", show_command)                # SW_SHOWNORMAL
        + struct.pack("<h", 0) + struct.pack("<h", 0)     # HotKey / Reserved
        + struct.pack("<i", 0) + struct.pack("<i", 0)     # Reserved2 / Reserved3
    )
    assert len(header) == 0x4C, f"header 长度应为 0x4C，实际 {len(header)}"

    idlist = build_idlist(target)
    # StringData 顺序必须与标志位顺序一致：NAME → WORKING_DIR → ICON_LOCATION
    strings = _string_data(desc) + _string_data(working_dir) + _string_data(icon)
    return (header + struct.pack("<H", len(idlist)) + idlist
            + _link_info(target) + strings + struct.pack("<I", 0))


def _desktop() -> str:
    """桌面目录（兼容 OneDrive 重定向）。"""
    prof = os.environ.get("USERPROFILE", os.path.expanduser("~"))
    onedrive = os.path.join(prof, "OneDrive", "Desktop")
    plain = os.path.join(prof, "Desktop")
    if os.path.isdir(onedrive) and not os.path.isdir(plain):
        return onedrive
    return plain if os.path.isdir(plain) else onedrive


def _selfcheck(blob: bytes) -> list[str]:
    """结构性自检。⚠️ 只检查"结构自洽"，**不检查"Windows 肯不肯收"** ——
    少了 IDList 的版本一样能通过下面每一条，却仍然打不开（见 docstring）。
    ⚠️ 也不要按"每个 ItemID 必须偶数"判：正品 .lnk 里卷条目（0x2F）就是 25 字节、
    IDList 总长也可能是奇数（Edge 样本 245）—— 奇偶不是可靠判据，别自己加戏。
    """
    errs: list[str] = []
    if struct.unpack("<I", blob[0:4])[0] != 0x4C:
        errs.append("HeaderSize != 0x4C")
    if blob[4:20] != _CLSID:
        errs.append("CLSID 不对")
    flags, = struct.unpack("<I", blob[20:24])
    if not flags & _HAS_IDLIST:
        errs.append("缺少 HasLinkTargetIDList（Windows 会拒收）")
    off = 0x4C
    n, = struct.unpack("<H", blob[off:off + 2])
    raw, pos, cnt = blob[off + 2:off + 2 + n], 0, 0
    while pos < len(raw) - 1:
        sz, = struct.unpack("<H", raw[pos:pos + 2])
        if sz == 0:
            break
        if pos + sz > len(raw):
            errs.append(f"第 {cnt} 个 ItemID 越界（size={sz}，剩余 {len(raw) - pos}）")
            return errs
        pos += sz
        cnt += 1
    if raw[pos:] != b"\x00\x00":
        errs.append(f"IDList 终止符缺失（结尾是 {raw[pos:].hex()}）")
    if cnt < 3:
        errs.append(f"ItemID 只有 {cnt} 个（至少 我的电脑+卷+目标）")
    return errs
    if raw[pos:] != b"\x00\x00":
        errs.append("IDList 终止符缺失")
    if cnt < 3:
        errs.append(f"ItemID 只有 {cnt} 个（至少 我的电脑+卷+目标）")
    return errs


def main(argv: list[str]) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    target = os.path.join(root, "dist", "bigAagents", "bigAagents.exe")
    if not os.path.isfile(target):
        print(f"[shortcut] 找不到 exe：{target}（先跑 scripts/build_exe.py）", file=sys.stderr)
        return 2

    blob = build_lnk(target, working_dir=os.path.dirname(target),
                     desc="bigAagents 量化研究面板")
    errs = _selfcheck(blob)
    if errs:
        print("[shortcut] 自检失败：" + "；".join(errs), file=sys.stderr)
        return 1

    check = "--check" in argv
    out_dir = os.path.join(root, "temp") if check else _desktop()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bigAagents 面板.lnk")
    with open(path, "wb") as f:
        f.write(blob)
    print(f"[shortcut] 已写 {path}（{len(blob)} 字节，自检通过）")
    if check:
        print("[shortcut] --check 模式：未写入桌面")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
