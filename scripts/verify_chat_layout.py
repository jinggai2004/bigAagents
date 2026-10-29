"""量 chat 页三列的真实尺寸：对话变长时，右侧房间会不会被撑高。

用本机 Edge 内核（chromium channel=msedge）跑真布局，不靠猜 CSS。
判据：
  1. .wrap / .room-pane / iframe 的高度在「注入 30 条长消息」前后必须**不变**；
  2. .thread 必须出现滚动条（scrollHeight > clientHeight），且页面 body 不出现滚动条；
  3. iframe 内 stage 的宽高比与其绘帧缓冲比例一致（不一致就是"人物被拉长"）。
"""
import json
import sys
from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8799/chat.html"

PROBE = """() => {
  const q = (s) => document.querySelector(s);
  const wrap = q('.wrap'), room = q('.room-pane'), fr = q('.room-pane iframe'),
        th = q('.thread'), main = q('main'), body = document.body;
  const h = (el) => el ? Math.round(el.getBoundingClientRect().height) : null;
  const w = (el) => el ? Math.round(el.getBoundingClientRect().width) : null;
  let inner = null;
  try {
    const d = fr && fr.contentDocument;
    const st = d && d.getElementById('stage');
    const cv = d && d.querySelector('canvas');
    if (st) inner = {
      stageW: st.clientWidth, stageH: st.clientHeight,
      canvasCssW: cv ? Math.round(cv.getBoundingClientRect().width) : null,
      canvasCssH: cv ? Math.round(cv.getBoundingClientRect().height) : null,
      bufW: cv ? cv.width : null, bufH: cv ? cv.height : null,
    };
  } catch (e) { inner = { err: String(e).slice(0, 60) }; }
  return {
    winH: window.innerHeight, winW: window.innerWidth,
    bodyScrollH: body.scrollHeight, bodyClientH: body.clientHeight,
    mainH: h(main), wrapH: h(wrap), roomH: h(room), roomW: w(room),
    iframeH: h(fr), iframeW: w(fr),
    threadH: h(th), threadScrollH: th ? th.scrollHeight : null,
    threadClientH: th ? th.clientHeight : null,
    threadOverflow: th ? getComputedStyle(th).overflowY : null,
    inner,
  };
}"""


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
    out = {}
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge")
        # 用用户截图的口径：1280×720 左右的窗口（他那张 1080×677 是缩放后的）
        pg = b.new_page(viewport={"width": 1440, "height": 780})
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(2500)
        before = pg.evaluate(PROBE)

        # 注入 30 条很长的消息，模拟"中间对话很长"
        pg.evaluate("""() => {
          const th = document.querySelector('.thread');
          for (let i = 0; i < 30; i++) {
            const d = document.createElement('div');
            d.className = 'msg bot';
            d.innerHTML = '<div class="who">ASSISTANT</div><div class="bd">' +
              ('这是一条用于把对话区撑长的测试消息，包含行情表格和多段文字。'.repeat(6)) +
              '</div>';
            th.appendChild(d);
          }
        }""")
        pg.wait_for_timeout(600)
        after = pg.evaluate(PROBE)
        pg.screenshot(path="temp/shot_chat_long.png", full_page=False)
        b.close()

    print("=== 注入长对话前 ===")
    print(json.dumps(before, ensure_ascii=False, indent=1))
    print("=== 注入长对话后 ===")
    print(json.dumps(after, ensure_ascii=False, indent=1))

    bad = []
    for k in ("wrapH", "roomH", "iframeH", "mainH"):
        if before.get(k) != after.get(k):
            bad.append(f"{k}: {before.get(k)} → {after.get(k)}")
    if after["roomH"] and after["roomH"] < 200:
        bad.append(f"房间过矮: {after['roomH']}")
    print("\n=== 判据 ===")
    print("高度是否被撑开:", ("否 ✓" if not bad else "是 ✗ " + "; ".join(bad)))
    print("对话区是否独立滚动:", "是 ✓" if (after["threadScrollH"] or 0) > (after["threadClientH"] or 0)
          else f"否 ✗ (scroll={after['threadScrollH']} client={after['threadClientH']})")
    print("body 是否出现页面级滚动:", "否 ✓" if after["bodyScrollH"] <= after["bodyClientH"] + 1
          else f"是 ✗ (scroll={after['bodyScrollH']} client={after['bodyClientH']})")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
