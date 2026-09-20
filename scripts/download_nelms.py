"""Download the Nelms analytical phantom dataset.

Primary source (offline as of 2026):
    http://canislupusllc.com/CurveCompare/DVH-Analysis-Data-Etc.zip

Wayback Machine archive:
    https://web.archive.org/web/20190404173644/http://canislupusllc.com/CurveCompare/DVH-Analysis-Data-Etc.zip

The dataset contains DICOM RT Structure Set files with sphere, cylinder,
and cone contours at various axial spacings, synthetic DICOM RT Dose files
with linear dose gradients at grid resolutions from 0.4-3 mm, and
analytical reference values in Excel format.

Reference:
    Nelms BE, Tome WA, Robinson G, Wheeler J. (2015).
    "Variations in the contouring of organs at risk: test case from
    a patient with oropharyngeal cancer."
    International Journal of Radiation Oncology, Biology, Physics.
"""

from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# Resolve paths relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NELMS_DIR = PROJECT_ROOT / "data" / "nelms"

URLS = [
    "https://web.archive.org/web/20190404173644/http://canislupusllc.com/CurveCompare/DVH-Analysis-Data-Etc.zip",
    "http://canislupusllc.com/CurveCompare/DVH-Analysis-Data-Etc.zip",
]

EXPECTED_SHA256 = "db0a9607537d852aa67e6c14358d9973364af074e7fe942a635810f9a7912369"


def download_nelms(target_dir: Path = NELMS_DIR) -> Path:
    """Download and extract the Nelms analytical phantom dataset.

    Parameters
    ----------
    target_dir : Path
        Directory to extract the dataset into.

    Returns
    -------
    Path
        Path to the extracted dataset root directory.

    Raises
    ------
    ConnectionError
        If download fails from all URLs.
    RuntimeError
        If extraction produces no DICOM files or SHA-256 mismatch.
    """
    extracted_dir = target_dir / "DVH-Analysis-Data-Etc"
    if extracted_dir.exists() and any(extracted_dir.rglob("*.dcm")):
        print(f"Dataset already extracted at {extracted_dir}")
        return extracted_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / "DVH-Analysis-Data-Etc.zip"

    # Download
    if not zip_path.exists():
        for url in URLS:
            print(f"Attempting download from:\n  {url}")
            try:
                response = requests.get(url, stream=True, timeout=60)
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with (
                    open(zip_path, "wb") as f,
                    tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
                print("Download complete.")
                break
            except (requests.RequestException, OSError) as e:
                print(f"  Failed: {e}")
                if zip_path.exists():
                    zip_path.unlink()
                continue
        else:
            raise ConnectionError(
                "Failed to download Nelms dataset from all URLs. "
                "The dataset may need to be obtained manually."
            )

    # Verify checksum
    sha256 = hashlib.sha256()
    with open(zip_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    digest = sha256.hexdigest()
    print(f"SHA-256: {digest}")

    if digest != EXPECTED_SHA256:
        raise RuntimeError(
            f"SHA-256 mismatch.\n"
            f"  Expected: {EXPECTED_SHA256}\n"
            f"  Got:      {digest}\n"
            "The downloaded file may be corrupted."
        )

    # Extract
    print(f"Extracting to {target_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    dcm_count = len(list(extracted_dir.rglob("*.dcm")))
    if dcm_count == 0:
        raise RuntimeError("Extraction produced no DICOM files.")

    print(f"Extracted {dcm_count} DICOM files to {extracted_dir}")
    return extracted_dir


if __name__ == "__main__":
    try:
        path = download_nelms()
        print(f"\nDataset ready at: {path}")
    except (ConnectionError, RuntimeError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
