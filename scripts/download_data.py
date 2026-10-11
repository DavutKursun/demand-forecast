#!/usr/bin/env python
"""Download the Kaggle "Store Item Demand Forecasting Challenge" train.csv into data/raw/.

Needs the Kaggle API: `pip install kaggle` and ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY in the environment), and you must have accepted the competition rules on
kaggle.com once. The raw file is not committed (competition data terms).

    python scripts/download_data.py
"""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from demand_forecast.config import RAW_DIR  # noqa: E402

COMPETITION = "demand-forecasting-kernels-only"


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / "train.csv"
    if target.exists():
        print(f"{target} already exists")
        return 0
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", "train.csv", "-p", str(RAW_DIR)]
    print(" ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("download failed; check the Kaggle API setup and that you accepted the competition rules", file=sys.stderr)
        return r.returncode
    for z in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW_DIR)
        z.unlink()
    print(f"ok: {target} ({target.stat().st_size / 1e6:.1f} MB)" if target.exists() else "train.csv not found after download")
    return 0 if target.exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
