#!/usr/bin/env python
"""Capture dashboard screenshots with Playwright (headless Chromium) for the README.

    python scripts/screenshots.py                          # starts streamlit on a free port, writes docs/img/*.png
    python scripts/screenshots.py --url http://localhost:8501   # against an app that is already running

Dev-only tooling, not in any requirements file:  pip install playwright && playwright install chromium
"""

from __future__ import annotations

import argparse
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = {"forecast": "", "quality": "quality", "how": "how"}  # file name -> URL path; the default page lives at the root


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_healthy(url: str, timeout: float = 90) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"{url}/_stcore/health", timeout=2).read().strip() == b"ok":
                return
        except Exception:  # noqa: BLE001 - server not up yet
            time.sleep(0.5)
    raise TimeoutError(f"streamlit at {url} did not become healthy in {timeout}s")


def shrink(path: Path, max_kb: int) -> None:
    """Palette-quantize the PNG until it fits the size budget (dashboards compress very well)."""
    from PIL import Image

    if path.stat().st_size <= max_kb * 1024:
        return
    im = Image.open(path).convert("RGB")
    for colors in (256, 128, 64, 32):
        im.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(path, optimize=True)
        if path.stat().st_size <= max_kb * 1024:
            return


def capture(url: str, out_dir: Path, width: int, height: int, max_kb: int) -> list[Path]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
        for name, path in PAGES.items():
            page.goto(f"{url}/{path}", wait_until="networkidle")
            page.wait_for_selector(".js-plotly-plot", timeout=60_000)
            page.wait_for_timeout(2500)  # let Plotly finish drawing and fonts settle
            target = out_dir / f"{name}.png"
            page.screenshot(path=str(target))
            shrink(target, max_kb)
            print(f"{target.relative_to(ROOT)}  {target.stat().st_size / 1024:.0f} KB")
            written.append(target)
        browser.close()
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", help="existing app URL; if omitted, a headless streamlit is started on a free port")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "img")
    ap.add_argument("--width", type=int, default=1400)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--max-kb", type=int, default=400)
    a = ap.parse_args(argv)
    if a.url:
        capture(a.url.rstrip("/"), a.out, a.width, a.height, a.max_kb)
        return 0
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true", "--server.port", str(port),
           "--browser.gatherUsageStats", "false"]
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_healthy(url)
        capture(url, a.out, a.width, a.height, a.max_kb)
    finally:
        proc.terminate()
        proc.wait(timeout=15)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
