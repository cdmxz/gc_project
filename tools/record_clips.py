"""真实运行录屏（Playwright 页面录制 + 手绘光标，不受窗口遮挡影响）。

用法：
    python tools/record_clips.py tb      # 约 80 秒 → video/_build/clip_tb.mp4
    python tools/record_clips.py web     # 约 118 秒 → video/_build/clip_web.mp4
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "video" / "_build"
BUILD.mkdir(parents=True, exist_ok=True)

TB_PORT = 6006
WEB_PORT = 5000
VW, VH = 1536, 864
FFMPEG = None

CURSOR_JS = """
() => {
  if (window.__cursorReady) return;
  window.__cursorReady = true;
  const c = document.createElement('div');
  c.id = '__cursor';
  c.style.cssText = 'position:fixed;left:-40px;top:-40px;width:22px;height:28px;z-index:2147483647;' +
                    'pointer-events:none;';
  c.innerHTML = '<svg width="22" height="28" viewBox="0 0 22 28">' +
    '<path d="M2 1 L2 21 L7.2 16.4 L10.6 24.6 L14 23.2 L10.6 15.2 L17.6 15.2 Z" ' +
    'fill="#ffffff" stroke="#111111" stroke-width="1.6" stroke-linejoin="round"/></svg>';
  document.documentElement.appendChild(c);
  window.__setCursor = (x, y) => { c.style.left = x + 'px'; c.style.top = y + 'px'; };
  window.__clickFx = (x, y) => {
    const r = document.createElement('div');
    r.style.cssText = 'position:fixed;left:' + (x - 18) + 'px;top:' + (y - 18) + 'px;width:36px;height:36px;' +
      'border:3px solid #1f7a4d;border-radius:50%;z-index:2147483646;pointer-events:none;' +
      'opacity:.9;transition:transform .45s ease-out,opacity .45s ease-out;';
    document.documentElement.appendChild(r);
    requestAnimationFrame(() => { r.style.transform = 'scale(1.9)'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 700);
  };
}
"""


def ffmpeg():
    global FFMPEG
    if FFMPEG is None:
        import imageio_ffmpeg
        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return FFMPEG


def wait_url(url, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2).read(200)
            return True
        except Exception:
            time.sleep(1)
    return False


def probe_duration(path):
    out = subprocess.run([ffmpeg(), "-i", str(path)], capture_output=True, text=True, errors="ignore").stderr
    for line in out.splitlines():
        if "Duration:" in line:
            hms = line.split("Duration:")[1].split(",")[0].strip()
            h, m, s = hms.split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    return 0.0


class Demo:
    """封装页面 + 手绘光标，所有交互都通过它执行。"""

    def __init__(self, page):
        self.page = page
        self.page.evaluate(CURSOR_JS)

    def move(self, x, y, steps=12, pause=0.35):
        self.page.mouse.move(x, y, steps=steps)
        self.page.evaluate("([x, y]) => window.__setCursor(x, y)", [x, y])
        time.sleep(pause)

    def click(self, x, y, selector=None):
        self.move(x, y, steps=10, pause=0.5)
        self.page.evaluate("([x, y]) => window.__clickFx(x, y)", [x, y])
        if selector:
            self.page.click(selector)
        else:
            self.page.mouse.click(x, y)
        time.sleep(0.8)

    def wheel(self, dy, steps=14, delay=0.14):
        for _ in range(steps):
            self.page.mouse.wheel(0, dy / steps)
            time.sleep(delay)


def record_tb():
    from playwright.sync_api import sync_playwright

    if not wait_url(f"http://127.0.0.1:{TB_PORT}/", timeout=3):
        print("[启动] TensorBoard ...")
        subprocess.Popen([sys.executable, "-m", "tensorboard.main", "--logdir", str(ROOT / "runs"),
                          "--port", str(TB_PORT), "--host", "127.0.0.1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert wait_url(f"http://127.0.0.1:{TB_PORT}/"), "TensorBoard 启动失败"
    print("[ok] TensorBoard 已就绪")

    raw_dir = BUILD / "raw_tb"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": VW, "height": VH},
                                  record_video_dir=str(raw_dir),
                                  record_video_size={"width": VW, "height": VH})
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{TB_PORT}/#scalars", wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(7000)
        demo = Demo(page)
        t0 = time.time()

        demo.move(1200, 200, steps=16, pause=1.5)
        time.sleep(1.2)
        hdr = page.get_by_text("Loss", exact=True).first
        hdr.scroll_into_view_if_needed()
        time.sleep(1.2)
        hdr.click()          # Loss 默认折叠，点一次展开
        time.sleep(3.5)
        page.wait_for_timeout(1000)
        demo.move(760, 500, steps=8, pause=0.6)
        demo.wheel(700, steps=18, delay=0.16)     # 滑到底部，让 Loss 两张图完整显示
        time.sleep(2.5)

        anchor = page.get_by_text("Loss/train").first.bounding_box()
        ax = (anchor["x"] + 170) if anchor else 700
        ay = (anchor["y"] + 120) if anchor else 330
        page.screenshot(path=str(BUILD / "_tb_check1.png"))

        demo.move(int(ax) - 120, int(ay) - 60, steps=14, pause=1.2)
        for i in range(8):                              # 沿 Loss 曲线逐点读取数值
            demo.move(int(ax) - 60 + 42 * i, int(ay) + (i % 3) * 14, steps=6, pause=1.35)
        time.sleep(2)
        demo.move(int(ax) + 300, int(ay) + 40, steps=12, pause=2)
        page.screenshot(path=str(BUILD / "_tb_check2.png"))
        demo.wheel(280, steps=14, delay=0.16)           # 下移到 Loss/val 与后续分组
        time.sleep(3)
        demo.wheel(-300, steps=14, delay=0.16)
        time.sleep(2.5)
        demo.move(int(ax) - 40, int(ay) - 40, steps=10, pause=1.5)
        for i in range(5):
            demo.move(int(ax) + 30 + 55 * i, int(ay) - 30, steps=5, pause=1.25)
        time.sleep(2)
        while time.time() - t0 < 78:
            time.sleep(0.3)
        video = page.video.path()
        ctx.close()
        browser.close()

    out = BUILD / "clip_tb.mp4"
    subprocess.run([ffmpeg(), "-y", "-i", str(video), "-r", "25",
                    "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[完成] {out}  {probe_duration(out):.1f}s")


def record_web():
    from playwright.sync_api import sync_playwright

    if not wait_url(f"http://127.0.0.1:{WEB_PORT}/", timeout=3):
        subprocess.Popen([sys.executable, str(ROOT / "src" / "app.py"), "--port", str(WEB_PORT)],
                         cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        assert wait_url(f"http://127.0.0.1:{WEB_PORT}/"), "Flask 启动失败"
    print("[ok] 前端服务已就绪")

    img1 = ROOT / "dataset" / "samples" / "plastic_1_plastic100.jpg"
    img2 = ROOT / "dataset" / "samples" / "cardboard_1_cardboard101.jpg"
    bad = BUILD / "_bad.txt"
    bad.write_text("this is not an image", encoding="utf-8")

    raw_dir = BUILD / "raw_web"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": VW, "height": VH},
                                  record_video_dir=str(raw_dir),
                                  record_video_size={"width": VW, "height": VH})
        page = ctx.new_page()
        page.goto(f"http://127.0.0.1:{WEB_PORT}/", wait_until="networkidle", timeout=60000)
        page.wait_for_timeout(4000)
        demo = Demo(page)
        t0 = time.time()

        demo.move(760, 300, steps=14, pause=1.5)
        time.sleep(1.5)
        demo.move(300, 470, steps=10, pause=0.6)
        page.set_input_files("#fileInput", str(img1))  # 真实上传第一张图
        time.sleep(5)
        demo.move(1180, 648, steps=12, pause=1.2)
        demo.click(1180, 648, selector="#predictBtn")  # 真实点击“开始识别”
        page.wait_for_selector("#resultBox:not(.hidden)", timeout=30000)
        time.sleep(6)
        demo.move(1000, 520, steps=10, pause=2.5)
        demo.move(760, 470, steps=10, pause=2)
        demo.wheel(330, steps=12, delay=0.14)          # 滚到样例图
        time.sleep(2.5)
        demo.move(300, 520, steps=12, pause=1)
        page.locator("#sampleRow figure").nth(1).click()   # 点样例 → 自动上传并识别
        page.wait_for_timeout(6000)
        time.sleep(4)
        demo.wheel(-300, steps=10, delay=0.14)
        time.sleep(2)
        demo.move(1180, 648, steps=10, pause=1)
        page.set_input_files("#fileInput", str(bad))   # 异常文件（非图片）
        time.sleep(6)
        demo.move(760, 380, steps=10, pause=2)
        page.set_input_files("#fileInput", str(img2))  # 再上传一张纸板
        time.sleep(3)
        demo.click(1180, 648, selector="#predictBtn")
        page.wait_for_timeout(5000)
        time.sleep(4)
        demo.wheel(330, steps=10, delay=0.14)
        time.sleep(4)

        while time.time() - t0 < 116:
            time.sleep(0.3)
        video = page.video.path()
        ctx.close()
        browser.close()

    bad.unlink(missing_ok=True)
    out = BUILD / "clip_web.mp4"
    subprocess.run([ffmpeg(), "-y", "-i", str(video), "-r", "25",
                    "-vf", "scale=1920:1080:flags=lanczos", "-c:v", "libx264",
                    "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", str(out)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[完成] {out}  {probe_duration(out):.1f}s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    if mode in ("tb", "all"):
        record_tb()
    if mode in ("web", "all"):
        record_web()
