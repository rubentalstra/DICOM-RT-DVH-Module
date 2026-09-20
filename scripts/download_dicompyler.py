"""Download the dicompyler-core bundled test dataset.

Source: https://github.com/dicompyler/dicompyler-core/tree/master/tests/testdata/example_data

Downloads the standard test case (RT Dose with embedded DVH, RT Struct,
RT Plan) used by dicompyler-core. The RT Dose file contains an embedded
DVH Sequence from the original TPS, which serves as a commercial
reference for cross-validation.

Published expected values (Heart, ROI 5):
    volume  = 437.46 cc
    max     = 3.11 Gy
    min     = 0.01 Gy
    mean    = 0.643 Gy
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DICOMPYLER_DIR = PROJECT_ROOT / "data" / "dicompyler"

BASE_URL = (
    "https://raw.githubusercontent.com/dicompyler/dicompyler-core"
    "/master/tests/testdata/example_data"
)

FILES = ["rtdose.dcm", "rtss.dcm", "rtplan.dcm"]


def download_dicompyler(target_dir: Path = DICOMPYLER_DIR) -> Path:
    """Download the dicompyler-core test dataset.

    Parameters
    ----------
    target_dir : Path
        Directory to save the files into.

    Returns
    -------
    Path
        Path to the download directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        dest = target_dir / filename
        if dest.exists():
            print(f"  Already exists: {filename}")
            continue

        url = f"{BASE_URL}/{filename}"
        print(f"  Downloading {filename} ...")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print(f"  Saved ({len(response.content)} bytes)")

    return target_dir


if __name__ == "__main__":
    try:
        path = download_dicompyler()
        print(f"\nDataset ready at: {path}")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
