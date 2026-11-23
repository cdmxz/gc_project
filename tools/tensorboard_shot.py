"""启动 TensorBoard 并用 Playwright 截取 SCALARS 页面，作为报告中的曲线截图。

用法：python tools/tensorboard_shot.py
输出：outputs/tb_overview.png、tb_loss_compare.png、tb_acc_compare.png、tb_single_exp.png
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
PORT_ALL = 6006
PORT_ONE = 6007
def best_exp():
    """按验证集 Macro-F1 选出最终模型对应的实验目录。"""
    import json
    best, best_f1 = None, -1
    for d in (ROOT / "runs").iterdir():
        f = d / "result.json"
        if f.exists():
            r = json.loads(f.read_text(encoding="utf-8"))
            if r.get("best_val_f1", -1) > best_f1:
                best, best_f1 = d.name, r["best_val_f1"]
    return best or "E3_resnet18_lr3e-4_aug"


FINAL_EXP = best_exp()


def start_tb(logdir, port):
    return subprocess.Popen(
        [sys.executable, "-m", "tensorboard.main", "--logdir", str(logdir),
         "--port", str(port), "--host", "127.0.0.1", "--reload_interval", "3"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def wait_up(port, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).read(100)
            return True
        except Exception:
            time.sleep(1)
    return False


def trim(path):
    """裁掉截图四周的纯白边。"""
    from PIL import Image, ImageChops
    im = Image.open(path).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg)
    bbox = diff.getbbox()
    if bbox:
        pad = 12
        box = (max(0, bbox[0] - pad), max(0, bbox[1] - pad),
               min(im.width, bbox[2] + pad), min(im.height, bbox[3] + pad))
        im.crop(box).save(path)
    return path


def shoot_section(page, header_text, out_path, next_text=None):
    """展开指定分组（Loss / Accuracy）并按内容自动裁剪。"""
    import numpy as np
    from PIL import Image

    header = page.get_by_text(header_text, exact=True).first
    header.scroll_into_view_if_needed()
    page.wait_for_timeout(500)
    expanded = page.get_by_text(f"{header_text}/train").count() > 0
    if not expanded:
        header.click()
        page.wait_for_timeout(3500)
    else:
        page.wait_for_timeout(800)
    box = header.bounding_box()
    scale = 2
    shot = page.screenshot(full_page=True)
    open(str(out_path), "wb").write(shot)

    im = Image.open(out_path).convert("RGB")
    arr = np.array(im)
    top = max(0, int((box["y"] - 14) * scale)) if box else 0
    left = int(302 * scale)
    sub = arr[top:, left:, :]
    nonwhite = (sub < 248).any(axis=2)
    rows = np.where(nonwhite.sum(axis=1) > 4)[0]
    limit = arr.shape[0]
    if next_text:
        try:
            nxt = page.get_by_text(next_text, exact=True).first.bounding_box()
            if nxt:
                limit = int((nxt["y"] - 7) * scale)
        except Exception:
            pass
    if not next_text:
        try:
            hp = page.get_by_text("hparam", exact=True).first.bounding_box()
            if hp:
                limit = min(limit, int((hp["y"] - 4) * scale))
        except Exception:
            pass
    if len(rows):
        bottom = min(top + rows[-1] + 16, limit)
    else:
        bottom = min(limit, top + 400 * scale)
    crop = im.crop((left, top, im.width, min(bottom, im.height)))
    crop.save(out_path)
    trim(out_path)
    print("saved", out_path.name, crop.size)


def main():
    from playwright.sync_api import sync_playwright

    tb_all = start_tb(ROOT / "runs", PORT_ALL)
    tb_one = start_tb(ROOT / "runs" / FINAL_EXP, PORT_ONE)
    try:
        assert wait_up(PORT_ALL) and wait_up(PORT_ONE), "TensorBoard 启动失败"
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome", None):
                try:
                    browser = p.chromium.launch(channel=channel) if channel else p.chromium.launch()
                    break
                except Exception as exc:
                    print("launch failed:", channel, str(exc)[:80])

            page = browser.new_page(viewport={"width": 1500, "height": 1500}, device_scale_factor=2)
            page.goto(f"http://127.0.0.1:{PORT_ALL}/#scalars", wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(6000)
            page.screenshot(path=str(OUT / "tb_overview.png"))
            print("saved tb_overview.png")

            shoot_section(page, "Loss", OUT / "tb_loss_compare.png")
            shoot_section(page, "Accuracy", OUT / "tb_acc_compare.png", "F1_macro")

            page2 = browser.new_page(viewport={"width": 1500, "height": 1500}, device_scale_factor=2)
            page2.goto(f"http://127.0.0.1:{PORT_ONE}/#scalars", wait_until="networkidle", timeout=90000)
            page2.wait_for_timeout(6000)
            shoot_section(page2, "Loss", OUT / "tb_single_exp.png")
            page2.get_by_text("Loss", exact=True).first.click()
            page2.wait_for_timeout(1000)
            shoot_section(page2, "Accuracy", OUT / "tb_single_acc.png", "F1_macro")
            browser.close()
    finally:
        for proc in (tb_all, tb_one):
            proc.terminate()
        print("[done]")


if __name__ == "__main__":
    main()
