"""把阿深的脸拍下来，四种表情并排对比 —— 只靠读代码判断"表情改好了没有"是不可靠的。

判据分两层：
  1. 机械：腮红必须贴在脸面内（z < 0.4286），不能再是浮在脸外的球；
  2. 肉眼：四张图并排，能看出眉/嘴/眼睛的差别（人来看）。
"""
import json
import sys
from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8799/companion/face_preview.html?noblink=1"
MOODS = ["neutral", "happy", "sad", "think"]


def _preflight():
    """页面验证的前提是本地服务在跑；不检查会得到 `ERR_CONNECTION_REFUSED`，
    看起来像"页面代码坏了"。

    用 socket 探端口而不是 try/except urlopen：这里没有任何需要记录日志的
    异常情况（"没起服务"就是没起），异常控制流程还会被静默 except 审计记一笔。
    """
    import socket
    with socket.socket() as s:
        s.settimeout(1.5)
        if s.connect_ex(("127.0.0.1", 8799)) != 0:
            print("本地服务没起。先跑：")
            print("  python scripts/serve_web.py --port 8799")
            sys.exit(2)


def main():
    _preflight()
    shots = {}
    facts = {}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", args=["--use-gl=angle", "--enable-webgl"])
        for m in MOODS:
            pg = b.new_page(viewport={"width": 420, "height": 420})
            pg.goto(f"{BASE}&mood={m}", wait_until="networkidle")
            pg.wait_for_timeout(900)
            ok = pg.evaluate("() => document.documentElement.dataset.ready === '1'")
            if not ok:
                print(f"[{m}] 页面未就绪")
            facts[m] = pg.evaluate("() => (window.__face ? window.__face() : null)")
            pg.screenshot(path=f"temp/face_{m}.png")
            shots[m] = f"temp/face_{m}.png"
            pg.close()
        b.close()

    print("=== 机械事实 ===")
    for m in MOODS:
        f = facts.get(m)
        print(f"{m:8}", json.dumps(f, ensure_ascii=False) if f else "NULL")

    base = facts.get("neutral") or {}
    bad = []
    if not base:
        bad.append("neutral 没有拿到 __face()，页面脚本可能没跑起来")
    else:
        if base.get("eyeCount") != 2:
            bad.append(f"眼睛数量 {base.get('eyeCount')} ≠ 2")
        if base.get("browCount") != 2:
            bad.append(f"眉毛数量 {base.get('browCount')} ≠ 2")
        if base.get("blushCount") != 2:
            bad.append(f"腮红数量 {base.get('blushCount')} ≠ 2")
        for z in (base.get("blushZ") or []):
            if z is not None and z > 0.4286:
                bad.append(f"腮红浮在脸外：z={z} > 0.4286")

    # 表情必须真的不一样：眉毛角/嘴型/睁眼度至少各有一处变化
    def diff(a, b, key, path=None):
        fa, fb = (facts.get(a) or {}), (facts.get(b) or {})
        return fa.get(key) != fb.get(key)

    if not diff("neutral", "happy", "mouth"):
        bad.append("happy 的嘴型与 neutral 相同 ⇒ 笑不出来")
    if not diff("neutral", "sad", "browZ"):
        bad.append("sad 的眉毛与 neutral 相同 ⇒ 看不出难过")
    if not diff("neutral", "happy", "eyeOpen"):
        bad.append("happy 的睁眼度与 neutral 相同 ⇒ 没笑眼")

    # 眉毛必须**左右镜像对称**：两侧"内端比外端高多少"要同号且接近。
    # 只看 local rotation.z 判不出来（左右 wrap 的世界朝向相同、不是镜像，
    # 所以同一条弧在两侧的"内/外端"映射是反的）—— 必须读世界坐标。
    # 2026-10-28 实际发生过：四格全是"左眉朝外斜、右眉朝内斜"，肉眼像单侧挑眉，
    # 而 browZ 读出来只是 [-0.22, 0.22] 这种"看起来很正常"的数字。
    for m in MOODS:
        ends = (facts.get(m) or {}).get("browEnds") or []
        if len(ends) != 2:
            bad.append(f"{m} 没拿到 browEnds（face_preview.html 的 __face 缺字段）")
            continue
        dy = [e.get("innerDy") for e in ends]
        if any(v is None for v in dy):
            bad.append(f"{m} 的 innerDy 为空")
            continue
        if dy[0] * dy[1] <= 0:
            bad.append(f"{m} 两眉异号：左 {dy[0]:+.4f} / 右 {dy[1]:+.4f} ⇒ 单侧挑眉")
        elif abs(dy[0] - dy[1]) > 0.005:
            bad.append(f"{m} 两眉不对称：左 {dy[0]:+.4f} / 右 {dy[1]:+.4f}")
    # 语义方向：难过的内侧要比开心更抬高（八字眉），否则"情绪"只落在嘴上
    sad_dy = ((facts.get("sad") or {}).get("browEnds") or [{}])[0].get("innerDy")
    happy_dy = ((facts.get("happy") or {}).get("browEnds") or [{}])[0].get("innerDy")
    if sad_dy is not None and happy_dy is not None and sad_dy <= happy_dy:
        bad.append(f"sad 内端 {sad_dy:+.4f} 不高于 happy 的 {happy_dy:+.4f} ⇒ 八字眉没做出来")

    print("\n=== 判据 ===")
    print("五官结构:", "OK" if not bad else "FAIL")
    for x in bad:
        print("  ✗", x)
    print("\n截图:", ", ".join(shots.values()))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
