"""Build the static geography features and print a sanity check.

    python scripts/build_geography.py

Reads the Florida box from settings.yaml, writes artifacts/geography.npz (geography.build) and
prints the features at a few places whose answers are known from a map. The first run downloads
two Natural Earth layers through cartopy (a few MB).
"""

from __future__ import annotations

from pathlib import Path

from stormlattice import geography, grid
from stormlattice.config import settings

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = REPO_ROOT / "artifacts" / "geography.npz"

# name, latitude, longitude, what a map says
PLACES = [
    ("Orlando", 28.538, -81.379, "inland; Atlantic about 60 km, Gulf about 120 km"),
    ("Tampa", 27.951, -82.457, "on Tampa Bay: Gulf 0-10 km"),
    ("Jacksonville", 30.332, -81.656, "Atlantic about 25 km"),
    ("Tallahassee", 30.438, -84.281, "Gulf about 35 km"),
    ("Naples", 26.142, -81.795, "on the Gulf"),
    ("Flamingo", 25.142, -80.925, "mainland tip on Florida Bay: Gulf 0, Atlantic = the Keys"),
    ("Key West", 24.555, -81.780, "both coasts 0, mostly water"),
    ("Lake Okeechobee", 26.950, -80.830, "lake: not land, all water within 5 km"),
]


def cell_of(lat: float, lon: float, lat_max: float, lon_min: float) -> tuple[int, int]:
    """Row and column of the cell that contains a point."""
    return int((lat_max - lat) / grid.CELL_DEG), int((lon - lon_min) / grid.CELL_DEG)


def main() -> None:
    box = settings()["grid"]
    features = geography.build(CACHE_PATH, box["lat_min"], box["lat_max"], box["lon_min"], box["lon_max"])

    land = features["land_mask"]
    print(f"wrote {CACHE_PATH.relative_to(REPO_ROOT)}  ({CACHE_PATH.stat().st_size / 1024:.0f} kB)")
    print(f"grid {land.shape[0]} x {land.shape[1]} cells, {land.mean():.1%} land")
    print()
    print(f"{'place':16s} {'land':5s} {'water 5km':>9s} {'water 20km':>10s} {'Atlantic km':>11s} {'Gulf km':>8s}   expected")

    route = settings()["route"]
    places = PLACES + [("SunTrax", route["suntrax_lonlat"][1], route["suntrax_lonlat"][0], "corridor start"),
                       ("MCO", route["mco_lonlat"][1], route["mco_lonlat"][0], "corridor end")]
    for name, lat, lon, expected in places:
        r, c = cell_of(lat, lon, box["lat_max"], box["lon_min"])
        print(f"{name:16s} {'yes' if land[r, c] else 'no':5s} {features['water_frac_5km'][r, c]:9.2f} "
              f"{features['water_frac_20km'][r, c]:10.2f} {features['dist_atlantic_km'][r, c]:11.1f} "
              f"{features['dist_gulf_km'][r, c]:8.1f}   {expected}")


if __name__ == "__main__":
    main()
