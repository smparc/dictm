"""
fetch_data.py
-------------
Download the SCDB (Supreme Court Database) case-centered dataset.

The dataset is not vendored in the repository — it is redistributed by
Washington University Law under their own terms. This script fetches it so a
clean clone is reproducible:

    python scripts/fetch_data.py                # latest known release (2024_01)
    python scripts/fetch_data.py --release 2023_01
    python scripts/fetch_data.py --force        # re-download even if present
"""

import argparse
import io
import os
import sys
import urllib.request
import zipfile

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DEFAULT_RELEASE = "2024_01"
URL_TEMPLATE = (
    "http://scdb.wustl.edu/_brickFiles/{release}/"
    "SCDB_{release}_caseCentered_Citation.csv.zip"
)


def target_path(release: str) -> str:
    return os.path.join(DATA_DIR, f"SCDB_{release}_caseCentered_Citation.csv")


def fetch(release: str = DEFAULT_RELEASE, force: bool = False) -> str:
    """Download and extract the SCDB CSV. Returns the path to the extracted file."""
    dest = target_path(release)
    if os.path.exists(dest) and not force:
        print(f"Already present: {dest}")
        return dest

    os.makedirs(DATA_DIR, exist_ok=True)
    url = URL_TEMPLATE.format(release=release)
    print(f"Downloading {url} ...")

    request = urllib.request.Request(url, headers={"User-Agent": "dictm/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    print(f"  {len(payload):,} bytes")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV found inside {url}")
        with archive.open(names[0]) as src, open(dest, "wb") as out:
            out.write(src.read())

    print(f"Extracted to {dest}")
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the SCDB case-centered dataset")
    parser.add_argument("--release", default=DEFAULT_RELEASE, help="SCDB release, e.g. 2024_01")
    parser.add_argument("--force", action="store_true", help="Re-download even if present")
    args = parser.parse_args()

    try:
        fetch(args.release, args.force)
    except Exception as exc:  # network / archive problems are the expected failure here
        print(f"Failed to fetch SCDB data: {exc}", file=sys.stderr)
        print("Download manually from http://scdb.wustl.edu/data.php", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
