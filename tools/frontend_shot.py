"""启动 Flask 服务并用 Playwright 截图，保存前端运行效果（报告与视频素材）。

用法：python tools/frontend_shot.py
输出：outputs/frontend_home.png、frontend_result.png、frontend_error.png、frontend_result_2.png
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PORT = 5000

DEMO_IMAGES = [
    ROOT / "dataset" / "samples" / "plastic_1_plastic100.jpg",
    ROOT / "dataset" / "samples" / "cardboard_1_cardboard101.jpg",
]


def wait_up(url, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(url, timeout=2).read(200)
            return True
        except Exception:
            time.sleep(0.7)
    return False


def main():
    from playwright.sync_api import sync_playwright

    server = subprocess.Popen([sys.executable, str(ROOT / "src" / "app.py"), "--port", str(PORT)],
                              cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert wait_up(f"http://127.0.0.1:{PORT}/api/model-info"), "Flask 启动失败"
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome", None):
                try:
                    browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                    break
                except Exception as exc:
                    print("launch failed:", channel, exc)
            page = browser.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
            page.goto(f"http://127.0.0.1:{PORT}/", wait_until="networkidle")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT / "frontend_home.png"))
            print("saved frontend_home.png")

            # 上传 -> 识别
            page.set_input_files("#fileInput", str(DEMO_IMAGES[0]))
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / "frontend_preview.png"))
            page.click("#predictBtn")
            page.wait_for_selector("#resultBox:not(.hidden)", timeout=30000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / "frontend_result.png"))
            print("saved frontend_result.png")

            # 第二张图片
            page.set_input_files("#fileInput", str(DEMO_IMAGES[1]))
            page.wait_for_timeout(800)
            page.click("#predictBtn")
            page.wait_for_selector("#resultBox:not(.hidden)", timeout=30000)
            page.wait_for_timeout(1200)
            page.screenshot(path=str(OUT / "frontend_result_2.png"))
            print("saved frontend_result_2.png")

            # 异常输入：非法文件类型
            tmp = OUT / "_bad.txt"
            tmp.write_text("not an image", encoding="utf-8")
            page.set_input_files("#fileInput", str(tmp))
            page.wait_for_timeout(1000)
            page.screenshot(path=str(OUT / "frontend_error.png"))
            print("saved frontend_error.png")
            tmp.unlink(missing_ok=True)

            browser.close()
    finally:
        server.terminate()
        print("[done]")


if __name__ == "__main__":
    main()
