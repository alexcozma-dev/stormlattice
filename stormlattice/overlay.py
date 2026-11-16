"""
Map overlays: turn a grid array into an image a web map can place by its bounds.

Web maps (Leaflet, MapLibre) draw in Web Mercator. Mercator x is linear in longitude, so the
grid's columns already line up. Mercator y is not linear in latitude: it stretches towards the
pole. An image whose rows are evenly spaced in latitude, pinned to the map by its bounds, is
drawn up to 5.4 km too far north in the middle of the Florida box (worst near 27.8 N).

The fix is a 1-D resample (docs/structure-review.md, "Grid decision"): build output rows evenly
spaced in Mercator y between the box edges and take the nearest source row for each. Values are
copied, never blended, so dBZ stays exact and NaN stays NaN. The result is placed with an
ordinary bounds overlay (L.imageOverlay in Leaflet) and is then exact to half a cell.

Latitudes are in degrees. Mercator y is in radians on the unit sphere, north positive.
Arrays are (ny, nx), row 0 north. Images are (ny, nx, 4) uint8 RGBA.
"""

from __future__ import annotations

import numpy as np

# Reflectivity colour table: the familiar US weather-radar colours, one per 5 dBZ step.
# DBZ_LEVELS holds the lower edge of each step. Below the first edge the image is transparent.
# Storms in this project start at 35 dBZ, which is where the table turns yellow.
DBZ_LEVELS = np.arange(15.0, 80.0, 5.0)     # 15, 20, ..., 75
LEVEL_RGB = np.array([
    (3, 0, 244),        # 15  blue
    (2, 253, 2),        # 20  greens
    (1, 197, 1),        # 25
    (0, 142, 0),        # 30
    (253, 248, 2),      # 35  yellows
    (229, 188, 0),      # 40
    (253, 149, 0),      # 45  orange
    (253, 0, 0),        # 50  reds
    (212, 0, 0),        # 55
    (188, 0, 0),        # 60
    (248, 0, 253),      # 65  magenta
    (152, 84, 198),     # 70  purple
    (253, 253, 253),    # 75  white
], dtype=np.uint8)


def mercator_y(lat_deg):
    """Web Mercator y of a latitude: radians on the unit sphere, 0 at the equator. Works on arrays."""
    return np.arcsinh(np.tan(np.radians(lat_deg)))


def mercator_lat(y):
    """Latitude in degrees of a Web Mercator y: the inverse of mercator_y. Works on arrays."""
    return np.degrees(np.arctan(np.sinh(y)))


def mercator_row_count(lat_min: float, lat_max: float, ny: int) -> int:
    """
    The fewest output rows, evenly spaced in Mercator y, that still use every one of ny source rows.

    With fewer rows, an output row is taller than the thinnest source row and steps over it, so a
    storm one cell thick could vanish from the image.
    """
    y_edges = mercator_y(np.linspace(lat_max, lat_min, ny + 1))     # edges of the source rows
    heights = -np.diff(y_edges)                                     # Mercator height of each source row
    return int(np.ceil((y_edges[0] - y_edges[-1]) / heights.min()))


def mercator_rows(lat_min: float, lat_max: float, ny: int, ny_out: int | None = None) -> np.ndarray:
    """
    The source row to copy into each output row of a Mercator-spaced image: (ny_out,) int array.

    The ny source rows are evenly spaced in latitude between lat_max (row 0) and lat_min. The
    output rows are evenly spaced in Mercator y between the same two edges, row 0 north. Each
    output row gets the source row that contains its centre, which is also the nearest one.
    ny_out defaults to mercator_row_count(...), so no source row is skipped.
    """
    if ny_out is None:
        ny_out = mercator_row_count(lat_min, lat_max, ny)
    y_north, y_south = mercator_y(lat_max), mercator_y(lat_min)
    y_centres = y_north - (np.arange(ny_out) + 0.5) / ny_out * (y_north - y_south)
    lat_centres = mercator_lat(y_centres)
    # Centres lie strictly inside the box, so the index is always 0 .. ny - 1.
    return np.floor((lat_max - lat_centres) / (lat_max - lat_min) * ny).astype(int)


def to_mercator_rows(a: np.ndarray, lat_min: float, lat_max: float, ny_out: int | None = None) -> np.ndarray:
    """`a` (ny, nx) with its rows re-spaced evenly in Mercator y: (ny_out, nx), values copied."""
    return a[mercator_rows(lat_min, lat_max, a.shape[0], ny_out)]


def reflectivity_rgba(dbz: np.ndarray) -> np.ndarray:
    """
    Reflectivity (ny, nx) in dBZ as an image: (ny, nx, 4) uint8 RGBA.

    Cells below DBZ_LEVELS[0] (15 dBZ) and NaN cells (no radar coverage) are fully transparent.
    Every other cell is opaque, in the colour of its 5 dBZ step. Values above the last edge keep
    the last colour.
    """
    # A comparison with NaN is False, so NaN cells drop out here. That matters: np.digitize
    # sorts NaN above every edge and would otherwise give it the top colour.
    shown = dbz >= DBZ_LEVELS[0]
    step = np.digitize(dbz[shown], DBZ_LEVELS) - 1      # 0 for 15-20 dBZ, ..., 12 for 75 and up
    rgba = np.zeros(dbz.shape + (4,), dtype=np.uint8)
    rgba[shown, :3] = LEVEL_RGB[step]
    rgba[shown, 3] = 255
    return rgba
