"""Fetch the latest MRMS composite reflectivity frame over Florida and save a PNG.

First coding task. Usage:
    python scripts/fetch_mrms_frame.py --out data/latest.png
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/latest.png"))
    args = parser.parse_args()

    # 1. stormlattice.data.mrms.latest_key()
    # 2. stormlattice.data.mrms.download(key, data_dir())
    # 3. stormlattice.data.mrms.load_florida(path)
    # 4. plot with matplotlib (fixed color scale, 0-70 dBZ) and save to args.out
    raise NotImplementedError(f"Implement in the next session (would write {args.out})")


if __name__ == "__main__":
    main()
