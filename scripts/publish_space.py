#!/usr/bin/env python
"""Mirror the dashboard to a Hugging Face Space (Docker SDK running Streamlit) through the Hub API only.

Hugging Face no longer accepts ``sdk: streamlit`` for new Spaces (the create API allows only gradio,
docker and static), so the Space is a Docker Space: a generated Dockerfile on python:3.11-slim installs
the pinned requirements and runs ``streamlit run app.py`` on port 7860.

    python scripts/publish_space.py --dry-run          # assemble the tree in a temp folder and list it
    python scripts/publish_space.py                    # create DavutKursun/demand-forecast (Docker Space, public, cpu-basic), upload, wait
    python scripts/publish_space.py --logs build       # print the Space's build (or run) logs and exit

The Space gets exactly what the three pages need and nothing else: app.py, dashboard/, the two
demand_forecast modules the app imports (config, recommend), .streamlit/config.toml, the eight
artifacts, requirements-app.txt as requirements.txt, the Dockerfile, and a Space README with the YAML header.
One commit via upload_folder. Needs `hf auth login`; huggingface_hub is dev-only tooling and is
not in the requirements files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/DavutKursun/demand-forecast"

# the eight artifacts the app reads (kept in sync with dashboard/common.py REQUIRED by a test)
ARTIFACTS = (
    "weekly.parquet", "forecasts.parquet", "backtest_summary.csv", "backtest_by_fold.csv",
    "backtest_by_series.csv", "worst_items.csv", "feature_importance.csv", "metrics.json",
)
CODE = (
    "app.py",
    ".streamlit/config.toml",
    "demand_forecast/__init__.py",
    "demand_forecast/config.py",
    "demand_forecast/recommend.py",
)


def pinned_version(requirements: Path, package: str) -> str:
    m = re.search(rf"^{package}==([\w.]+)", requirements.read_text(), flags=re.M)
    if not m:
        raise SystemExit(f"{package} is not pinned in {requirements}")
    return m.group(1)


PYTHON = "3.11"
PORT = 7860  # Hugging Face's default app_port for Docker Spaces


def dockerfile(streamlit_version: str) -> str:
    return f"""# Streamlit {streamlit_version} on Python {PYTHON}; the Space runs this container on Hugging Face (Docker SDK).
FROM python:{PYTHON}-slim

# Hugging Face runs the container as uid 1000; give it a writable home for Streamlit's own files.
RUN useradd -m -u 1000 user
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . /app
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

EXPOSE {PORT}
CMD ["streamlit", "run", "app.py", "--server.port={PORT}", "--server.address=0.0.0.0", \
     "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", \
     "--browser.gatherUsageStats=false"]
"""


def space_readme(repo_id: str, streamlit_version: str, metrics: dict) -> str:
    lgb, sn, ma = (100 * metrics[k] for k in ("wape_lightgbm", "wape_seasonal_naive", "wape_moving_average_8"))
    return f"""---
title: demand-forecast
emoji: 📦
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: {PORT}
pinned: false
license: mit
short_description: Weekly demand forecasts with P10–P90 bands and order advice
---

Weekly sales forecasts for {metrics['n_series']} store × product series, with P10–P90 uncertainty bands and a plain-language order recommendation from stock on hand, lead time and service level.
LightGBM beats both rules of thumb a retailer would use in a {metrics['folds']}-fold rolling-origin backtest: WAPE {lgb:.1f}% vs {sn:.1f}% (same week last year) and {ma:.1f}% (8-week average), {metrics['folds_won']}/{metrics['folds']} folds won, P10–P90 coverage {100 * metrics['coverage_80']:.0f}%.
The app reads precomputed artifacts only; nothing is trained at runtime, so it starts in seconds. Docker Space: Python {PYTHON}, Streamlit {streamlit_version}, pinned requirements, cpu-basic.
Pages: Forecast (history, forecast and band, order box, CSV download), Model quality (backtest tables and charts, feature importance), How it works.
Source, training code, tests and design notes: {GITHUB} — this Space is a mirror of the dashboard there.
"""


def assemble(dst: Path, repo_id: str) -> list[tuple[str, int]]:
    """Copy the runtime tree into ``dst``; return (relative path, bytes) for every file."""
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for rel in CODE:
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst / rel)
    (dst / "dashboard").mkdir()
    for py in sorted((ROOT / "dashboard").glob("*.py")):
        shutil.copy2(py, dst / "dashboard" / py.name)
    (dst / "artifacts").mkdir()
    for name in ARTIFACTS:
        shutil.copy2(ROOT / "artifacts" / name, dst / "artifacts" / name)
    shutil.copy2(ROOT / "requirements-app.txt", dst / "requirements.txt")
    metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
    streamlit_version = pinned_version(ROOT / "requirements-app.txt", "streamlit")
    (dst / "Dockerfile").write_text(dockerfile(streamlit_version))
    (dst / "README.md").write_text(space_readme(repo_id, streamlit_version, metrics))
    return sorted((str(p.relative_to(dst)), p.stat().st_size) for p in dst.rglob("*") if p.is_file())


def direct_url(repo_id: str) -> str:
    """https://<owner>-<name>.hf.space, lowercased with underscores and dots hyphenated in the subdomain."""
    owner, name = repo_id.split("/")
    sub = f"{owner}-{name}".lower().replace("_", "-").replace(".", "-")
    return f"https://{sub}.hf.space"


def print_logs(repo_id: str, kind: str, seconds: float = 40.0) -> None:
    """Stream the Space's build or run logs (SSE) for a bounded time."""
    import requests
    from huggingface_hub import get_token

    url = f"https://huggingface.co/api/spaces/{repo_id}/logs/{kind}"
    with requests.get(url, headers={"Authorization": f"Bearer {get_token()}"}, stream=True, timeout=(10, seconds)) as r:
        print(f"[{kind} logs] http {r.status_code} (streaming for up to {seconds:.0f}s)")
        if r.status_code != 200:
            print(r.text[:500])
            return
        deadline = time.monotonic() + seconds
        try:
            for raw in r.iter_lines():
                if time.monotonic() > deadline:
                    break
                if not raw or not raw.startswith(b"data:"):
                    continue
                try:
                    evt = json.loads(raw[5:].decode())
                    print(evt.get("data", "").rstrip())
                except json.JSONDecodeError:
                    print(raw.decode(errors="replace"))
        except requests.exceptions.RequestException:  # idle SSE stream: read timeout surfaces as ConnectionError too
            pass


def wait_running(api, repo_id: str, timeout: float) -> tuple[str, float]:
    t0, last = time.monotonic(), None
    while time.monotonic() - t0 < timeout:
        rt = api.get_space_runtime(repo_id)
        stage = str(rt.stage)
        if stage != last:
            print(f"  {time.monotonic() - t0:5.0f}s  stage={stage}  hardware={rt.hardware}")
            last = stage
        if stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "NO_APP_FILE", "PAUSED"):
            return stage, time.monotonic() - t0
        time.sleep(10)
    return last or "UNKNOWN", time.monotonic() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="DavutKursun/demand-forecast")
    ap.add_argument("--hardware", default="cpu-basic")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="assemble and list only")
    ap.add_argument("--force", action="store_true", help="upload even if the Space already has files")
    ap.add_argument("--wait", type=float, default=900, help="seconds to wait for RUNNING")
    ap.add_argument("--out", type=Path, help="where to assemble (default: a temp folder)")
    ap.add_argument("--logs", choices=["build", "run"], help="print these logs and exit")
    a = ap.parse_args(argv)

    if a.logs:
        print_logs(a.repo_id, a.logs)
        return 0

    dst = a.out or Path(tempfile.mkdtemp(prefix="hf-space-"))
    files = assemble(dst, a.repo_id)
    total = sum(n for _, n in files)
    print(f"assembled {len(files)} files, {total / 1024:.0f} KB total, in {dst}")
    for rel, n in files:
        print(f"  {n:>8,}  {rel}")
    if a.dry_run:
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    if api.repo_exists(a.repo_id, repo_type="space"):
        existing = api.list_repo_files(a.repo_id, repo_type="space")
        print(f"Space {a.repo_id} already exists with {len(existing)} files: {existing}")
        if existing and not a.force:
            print("refusing to overwrite without --force", file=sys.stderr)
            return 2
    else:
        url = api.create_repo(a.repo_id, repo_type="space", space_sdk="docker", space_hardware=a.hardware,
                              private=a.private, exist_ok=True)
        print(f"created Space {url}")

    t_upload = time.monotonic()
    info = api.upload_folder(folder_path=str(dst), repo_id=a.repo_id, repo_type="space",
                             commit_message="Mirror the demand-forecast dashboard: app, artifacts, slim requirements")
    print(f"uploaded in one commit: {info.commit_url}")
    print("waiting for the Space to build and start…")
    stage, seconds = wait_running(api, a.repo_id, a.wait)
    print(f"final stage {stage} after {seconds:.0f}s (from upload: {time.monotonic() - t_upload:.0f}s)")
    print(f"Space page : https://huggingface.co/spaces/{a.repo_id}")
    print(f"direct URL : {direct_url(a.repo_id)}")
    return 0 if stage == "RUNNING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
