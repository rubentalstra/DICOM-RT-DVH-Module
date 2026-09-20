"""Download SlicerRtData Eclipse phantom datasets.

Source: https://github.com/SlicerRt/SlicerRtData

Target datasets:
    - eclipse-8.1.20-phantom-prostate (6 structures)
    - eclipse-8.1.20-phantom-breast   (1 structure with DVH)
    - eclipse-8.1.20-phantom-ent      (16 structures)

Each contains Eclipse-exported RT Dose, RT Structure Set, and RT Plan
files, plus DVH text export files for cross-validation.

Only DICOM RT files (RD, RS, RP) and DVH exports are downloaded — CT
slices and DRR images are skipped to save bandwidth.
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SLICERRT_DIR = PROJECT_ROOT / "data" / "slicerrt"

BASE_URL = (
    "https://raw.githubusercontent.com/SlicerRt/SlicerRtData/master"
)

# File manifests per phantom (RT files + DVH exports only, no CT/DRR)
PHANTOMS = {
    "prostate": {
        "subdir": "eclipse-8.1.20-phantom-prostate",
        "files": [
            "Original/RD.1.2.246.352.71.7.2088656855.452083.20110920153746.dcm",
            "Original/RS.1.2.246.352.71.4.2088656855.2404649.20110920153449.dcm",
            "Original/RP.1.2.246.352.71.5.2088656855.377401.20110920153647.dcm",
        ],
    },
    "breast": {
        "subdir": "eclipse-8.1.20-phantom-breast",
        "files": [
            "Original/RD.1.2.246.352.71.7.2088656855.452097.20110920152341.dcm",
            "Original/RS.1.2.246.352.71.4.2088656855.2402030.20110920095607.dcm",
            "Original/RP.1.2.246.352.71.5.2088656855.377051.20110920152006.dcm",
        ],
    },
    "ent": {
        "subdir": "eclipse-8.1.20-phantom-ent",
        "files": [
            "Original/RD.1.2.246.352.71.7.2088656855.452404.20110921073629.dcm",
            "Original/RS.1.2.246.352.71.4.2088656855.2401823.20110920093221.dcm",
            "Original/RP.1.2.246.352.71.5.2088656855.377514.20110921073559.dcm",
        ],
    },
}


def download_slicerrt(target_dir: Path = SLICERRT_DIR) -> Path:
    """Download SlicerRtData Eclipse phantom DICOM files.

    Parameters
    ----------
    target_dir : Path
        Root directory for SlicerRt data.

    Returns
    -------
    Path
        Path to the download directory.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    for phantom_name, info in PHANTOMS.items():
        phantom_dir = target_dir / phantom_name
        phantom_dir.mkdir(parents=True, exist_ok=True)
        subdir = info["subdir"]

        print(f"\n=== {phantom_name} ({subdir}) ===")

        for rel_path in info["files"]:
            filename = Path(rel_path).name
            dest = phantom_dir / filename
            if dest.exists():
                print(f"  Already exists: {filename}")
                continue

            url = f"{BASE_URL}/{subdir}/{rel_path}"
            print(f"  Downloading {filename} ...")
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with (
                    open(dest, "wb") as f,
                    tqdm(
                        total=total, unit="B", unit_scale=True,
                        desc=f"  {filename}", leave=False,
                    ) as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                print(f"  Saved ({dest.stat().st_size} bytes)")
            except requests.RequestException as e:
                print(f"  FAILED: {e}")
                if dest.exists():
                    dest.unlink()

    return target_dir


if __name__ == "__main__":
    try:
        path = download_slicerrt()
        print(f"\nDatasets ready at: {path}")
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
