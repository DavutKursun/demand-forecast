#!/usr/bin/env python
"""Record the README demo GIF with Playwright: Forecast page, two random items, lead time and
service level changes, switch to Model quality, scroll. Frames are screenshots assembled with Pillow
(no ffmpeg needed), downscaled and palette-quantised to stay small.

    python scripts/demo_gif.py                      # local headless streamlit on a free port
    python scripts/demo_gif.py --url https://<app>.streamlit.app

Dev-only tooling, not in any requirements file:  pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import io
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from screenshots import free_port, wait_healthy  # noqa: E402


def app_scope(page, timeout: float = 240.0):
    """The page itself, or the iframe Streamlit Community Cloud wraps the app in (its URL contains /~/+/)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for frame in page.frames:
            if "/~/+/" in frame.url:
                return frame
        if page.locator(".js-plotly-plot").count():
            return page
        time.sleep(0.5)
    return page


class Recorder:
    """Collects screenshots at a steady cadence while the script drives the page.

    ``page`` takes the screenshots; ``scope`` (the page or the app's iframe) is where widgets live.
    """

    def __init__(self, page, scope, fps: float):
        self.page, self.scope, self.frames, self.dt = page, scope, [], 1.0 / fps

    def hold(self, seconds: float) -> None:
        """Capture frames for ``seconds`` of real time (the page may still be settling)."""
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            t0 = time.monotonic()
            self.frames.append(self.page.screenshot(type="png"))
            time.sleep(max(0.0, self.dt - (time.monotonic() - t0)))

    def settle(self) -> None:
        """Wait for a Streamlit rerun to finish (status widget gone), capturing frames meanwhile."""
        status = self.scope.locator('[data-testid="stStatusWidget"]')
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if status.count() == 0:
                break
            self.frames.append(self.page.screenshot(type="png"))
            time.sleep(self.dt)


def drive(url: str, fps: float, width: int, height: int) -> list[bytes]:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)

        page.goto(url, wait_until="domcontentloaded")
        # a sleeping Community Cloud app (free tier) shows a wake-up button first
        wake = page.get_by_role("button", name=re.compile("get this app back up", re.I))
        try:
            wake.first.wait_for(timeout=8_000)
            wake.first.click()
            print("app was asleep: clicked the wake-up button")
        except Exception:  # noqa: BLE001 - no wake-up page, the app is already running
            pass
        app = app_scope(page)
        print("app renders in:", "an iframe (Community Cloud shell)" if app is not page else "the page itself")
        app.wait_for_selector(".js-plotly-plot", timeout=240_000)
        page.wait_for_timeout(1500)
        rec = Recorder(page, app, fps)
        rec.hold(2.0)

        random_btn = app.get_by_role("button", name="Random item")
        for _ in range(2):
            random_btn.click()
            rec.settle()
            rec.hold(2.0)

        # Streamlit's slider is a react-aria widget: a visually hidden <input type="range"> takes the keyboard
        slider = app.locator('[data-testid="stSlider"]').first
        thumb_value = slider.locator('[data-testid="stSliderThumbValue"]').first
        slider.locator('input[type="range"]').first.focus()
        for _ in range(2):  # lead time 2 → 4
            before = thumb_value.inner_text()
            page.keyboard.press("ArrowRight")
            rec.settle()
            rec.hold(0.8)
            if thumb_value.inner_text() == before:  # keyboard did not move it: click the right end of the track
                box = slider.locator('[data-orientation="horizontal"]').first.bounding_box()
                page.mouse.click(box["x"] + box["width"] - 2, box["y"] + box["height"] / 2)
                rec.settle()
                rec.hold(0.8)
        print("lead time now:", thumb_value.inner_text())
        rec.hold(1.0)

        app.get_by_text("P50", exact=True).click()
        rec.settle()
        rec.hold(2.0)

        app.get_by_role("link", name="Model quality").click()
        app.wait_for_selector("h1:has-text('Model quality')", timeout=30_000)
        app.wait_for_selector(".js-plotly-plot", timeout=30_000)
        rec.settle()
        rec.hold(2.0)

        page.mouse.move(width // 2 + 150, height // 2)
        for _ in range(4):
            page.mouse.wheel(0, 450)
            rec.hold(1.0)
        rec.hold(1.5)
        browser.close()
    return rec.frames


def to_gif(frames: list[bytes], out: Path, fps: float, scale: float, colors: int) -> int:
    from PIL import Image

    imgs = [Image.open(io.BytesIO(b)).convert("RGB") for b in frames]
    w, h = imgs[0].size
    size = (int(w * scale), int(h * scale))
    imgs = [im.resize(size, Image.LANCZOS) for im in imgs]
    # one global palette (built from a mosaic of evenly spaced frames) so Pillow can store only the
    # changed rectangle of each frame instead of a full redraw
    sample = imgs[:: max(1, len(imgs) // 8)]
    mosaic = Image.new("RGB", (size[0], size[1] * len(sample)))
    for i, im in enumerate(sample):
        mosaic.paste(im, (0, i * size[1]))
    palette = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    quantised = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    quantised[0].save(out, save_all=True, append_images=quantised[1:], duration=int(1000 / fps), loop=0,
                      optimize=True, disposal=1)
    return out.stat().st_size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="existing app URL; if omitted, a headless streamlit is started locally")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img" / "demo.gif")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=800)
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument("--max-mb", type=float, default=3.0)
    a = ap.parse_args(argv)

    proc = None
    url = a.url.rstrip("/") if a.url else None
    if url is None:
        port = free_port()
        url = f"http://127.0.0.1:{port}"
        proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
                                 "--server.port", str(port), "--browser.gatherUsageStats", "false"],
                                cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if proc is not None:
            wait_healthy(url)
        t0 = time.monotonic()
        frames = drive(url, a.fps, a.width, a.height)
        print(f"{len(frames)} frames captured over {time.monotonic() - t0:.0f} s of driving")
    finally:
        if proc is not None:
            proc.terminate()
            proc.wait(timeout=15)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for scale, colors in ((a.scale, 128), (a.scale, 64), (a.scale * 0.85, 64), (a.scale * 0.7, 48)):
        size = to_gif(frames, a.out, a.fps, scale, colors)
        print(f"{a.out.relative_to(ROOT)}: {len(frames)} frames, {len(frames) / a.fps:.1f} s, scale {scale:.2f}, "
              f"{colors} colours → {size / 1e6:.2f} MB")
        if size <= a.max_mb * 1e6:
            return 0
    print("could not get under the size budget", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
