"""
Static geography features on the 0.02 deg cell grid: computed once, cached as one .npz file.

Sea-breeze physics: land heats faster than water, onshore flow develops, fronts from the
Atlantic and Gulf coasts move inland and collide over the peninsula, lifting moist air. Lakes
make their own breezes. These features tell the birth model where a block sits in that picture.

Every feature is an (ny, nx) array, row 0 north, column 0 west:

    land_mask          bool      True where the cell centre is on land (not sea, not a lake)
    water_frac_5km     float32   share (0..1) of the area within 5 km that is water, sea or lake
    water_frac_20km    float32   the same within 20 km
    dist_atlantic_km   float32   km to the nearest Atlantic coast cell
    dist_gulf_km       float32   km to the nearest Gulf coast cell

Source: Natural Earth 1:10m "land" and "lakes" shapes (in the box: Okeechobee, George, Kissimmee,
Apopka, Istokpoga, Seminole). A cell takes the value at its centre. The land shape has no holes
for lakes, so "sea" is everything outside it and a lake shore is never a coast.

Coast distances
    A coast cell is a cell that is not sea and has a sea cell as one of its 4 neighbours. It is
    an Atlantic coast cell if one of those neighbours is Atlantic water, a Gulf coast cell if one
    is Gulf water, and both if it touches both. The distance is 0 on a coast cell and grows in
    both directions, inland and out to sea; it is never negative (land_mask says which side).
    It is inf if the grid has no coast of that kind.

The Atlantic / Gulf split
    This is our own rule, "which sea a coast faces", not an official boundary. Only the water is
    split; the coast inherits its name from the water it touches. Sea cells inside the polygon
    GULF_SIDE_LONLAT are Gulf water, all other sea cells are Atlantic water. The polygon was
    drawn by hand. North of the Everglades the peninsula itself separates the two seas, so the
    polygon's edge just runs down the middle of the land. The edge that matters approximately
    follows the Keys: from the mainland at the east end of Florida Bay (where US-1 leaves it)
    to Key West, then west past the Dry Tortugas to 83 W, then south. So:
        Florida Bay and everything north of the Keys   Gulf
        Biscayne Bay, Card and Barnes Sounds, the
        Straits of Florida south of the Keys           Atlantic
        the Keys themselves                            both: 0 km from each coast
        Flamingo, Cape Sable (the mainland's tip)      on the Gulf coast; their Atlantic distance
                                                       is measured across Florida Bay to the Keys
    Bays and lagoons (Tampa Bay, the Indian River) are sea in Natural Earth and take their name
    from where they lie, so the mainland shore of a lagoon is a coast.

Limits
    Only coasts and water inside the box are seen. Checked against a box padded by 2 deg: no
    land cell inside the evaluation region (72 km from the edge) changes. Open-water cells far
    offshore would be nearer to Cuba or Louisiana than to anything in the box. The western
    Bahamas are inside the box and count as Atlantic coast.
    Distances use one east-west spacing (grid.DX_KM), at most 3.4% off at the box edges.
    Near the box edge a water fraction is the share of the part of the disk inside the box.

shapely and cartopy are needed only to build the cache, so they are imported inside the functions
that use them. Loading the cache needs NumPy alone (the VM does not install shapely or cartopy).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from stormlattice import grid

FEATURE_NAMES = [
    "dist_atlantic_km",
    "dist_gulf_km",
    "water_frac_5km",
    "water_frac_20km",
    "land_mask",
]

# Sea cells inside this polygon are Gulf water (see "The Atlantic / Gulf split" above).
# (lon, lat) in degrees, drawn by hand. Only the Keys and 83 W edges cross water inside the Florida box.
GULF_SIDE_LONLAT = [
    (-82.50, 32.00),    # inland Georgia, north of the box
    (-81.70, 29.00),    # down the middle of the peninsula: Ocala forest,
    (-81.20, 27.00),    #   west of Lake Okeechobee,
    (-80.80, 25.60),    #   the Everglades
    (-80.45, 25.24),    # mainland shore between Florida Bay and Barnes Sound
    (-80.39, 25.185),   # Jewfish Creek, where US-1 reaches Key Largo
    (-80.45, 25.09),    # Key Largo
    (-80.63, 24.92),    # Islamorada
    (-80.82, 24.82),    # Long Key
    (-81.09, 24.71),    # Marathon
    (-81.36, 24.66),    # Big Pine Key
    (-81.80, 24.555),   # Key West
    (-83.00, 24.58),    # past the Marquesas and the Dry Tortugas to 83 W
    (-83.00, 22.00),    # south along 83 W, beyond the box
    (-90.00, 22.00),    # around the Gulf, outside the box
    (-90.00, 32.00),
]


def cell_centres(lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """Latitude of each row of cells (ny,), north to south, and longitude of each column (nx,), west to east."""
    ny = round((lat_max - lat_min) / grid.CELL_DEG)
    nx = round((lon_max - lon_min) / grid.CELL_DEG)
    lons = lon_min + (np.arange(nx) + 0.5) * grid.CELL_DEG      # TODO: move to grid.py
    return grid.row_lats(lat_max, ny), lons


def inside(shape, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """True where the cell centre lies inside `shape`, a shapely geometry in lon/lat degrees: (ny, nx) bool."""
    import shapely

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    shapely.prepare(shape)      # builds an index on the shape, so testing every cell takes well under a second
    return shapely.contains_xy(shape, lon_grid, lat_grid)


def coast_distances_km(sea: np.ndarray, gulf_side: np.ndarray):
    """
    Distance from every cell to the nearest Atlantic coast cell and to the nearest Gulf coast cell.

    `sea` and `gulf_side` are (ny, nx) bool. Returns two (ny, nx) arrays in km: 0 on a coast cell,
    inf everywhere if the grid has no coast of that kind.
    """
    distances = []
    for water in (sea & ~gulf_side, sea & gulf_side):       # Atlantic water, then Gulf water
        # binary_dilation grows the water by one cell up, down, left and right (not diagonally).
        coast = ~sea & ndimage.binary_dilation(water)
        if coast.any():
            # The transform measures to the nearest False cell; sampling gives the size of one
            # step along each axis, so the answer comes out in km.
            distances.append(ndimage.distance_transform_edt(~coast, sampling=(grid.DY_KM, grid.DX_KM)))
        else:
            distances.append(np.full(sea.shape, np.inf))
    return distances[0], distances[1]


# TODO: move to grid.py
def disk(radius_km: float) -> np.ndarray:
    """Cells whose centre lies within radius_km of the middle cell: a small bool array, odd on both sides."""
    ry = int(radius_km // grid.DY_KM)
    rx = int(radius_km // grid.DX_KM)
    dy_km = np.arange(-ry, ry + 1)[:, None] * grid.DY_KM
    dx_km = np.arange(-rx, rx + 1)[None, :] * grid.DX_KM
    return dy_km ** 2 + dx_km ** 2 <= radius_km ** 2


def water_fraction(land: np.ndarray, radius_km: float, lats: np.ndarray) -> np.ndarray:
    """
    Share of the area within radius_km of each cell that is water: (ny, nx), 0 = all land, 1 = all water.

    `land` is (ny, nx) bool and `lats` the latitude of each row. Areas are sums of cell areas
    (grid.cell_area_km2), not cell counts. Cells outside the grid are left out of both sums.
    """
    area_km2 = np.broadcast_to(grid.cell_area_km2(lats)[:, None], land.shape)
    kernel = disk(radius_km).astype(float)
    # mode="constant" pads with zeros: outside the grid there is no water area and no area at all.
    water_km2 = ndimage.convolve(np.where(land, 0.0, area_km2), kernel, mode="constant")
    total_km2 = ndimage.convolve(area_km2, kernel, mode="constant")
    return water_km2 / total_km2


def compute(land_shape, lake_shape, gulf_shape,
            lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> dict[str, np.ndarray]:
    """
    Every static feature for the box, from shapes in lon/lat degrees: FEATURE_NAMES -> (ny, nx) array.

    land_shape covers the land with its lakes filled in, lake_shape the lakes, and gulf_shape
    the side of the map whose sea is Gulf water.
    """
    lats, lons = cell_centres(lat_min, lat_max, lon_min, lon_max)
    sea = ~inside(land_shape, lats, lons)
    land = ~sea & ~inside(lake_shape, lats, lons)
    dist_atlantic_km, dist_gulf_km = coast_distances_km(sea, inside(gulf_shape, lats, lons))
    return {
        "dist_atlantic_km": dist_atlantic_km.astype(np.float32),
        "dist_gulf_km": dist_gulf_km.astype(np.float32),
        "water_frac_5km": water_fraction(land, 5, lats).astype(np.float32),
        "water_frac_20km": water_fraction(land, 20, lats).astype(np.float32),
        "land_mask": land,
    }


def natural_earth_shape(name: str, lat_min: float, lat_max: float, lon_min: float, lon_max: float):
    """
    One Natural Earth 1:10m physical layer ("land" or "lakes") cut to the box: a shapely geometry.

    cartopy downloads the layer on first use (a few MB, into its own data folder) and reads the
    local copy after that.
    """
    import shapely
    from cartopy.io import shapereader

    path = shapereader.natural_earth(resolution="10m", category="physical", name=name)
    shapes = list(shapereader.Reader(path).geometries())
    # Cut first, then merge: the layers cover the whole world and only a few pieces touch the box.
    return shapely.union_all(shapely.clip_by_rect(shapes, lon_min, lat_min, lon_max, lat_max))


def build(cache_path: Path, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> dict[str, np.ndarray]:
    """
    Compute every static feature for the box from Natural Earth and cache it at cache_path.

    Returns FEATURE_NAMES -> (ny, nx) array, the same dict load(cache_path) gives. The file also
    stores `box` = [lat_min, lat_max, lon_min, lon_max], so a cache built for another box can be told apart.
    """
    import shapely

    land_shape = natural_earth_shape("land", lat_min, lat_max, lon_min, lon_max)
    lake_shape = natural_earth_shape("lakes", lat_min, lat_max, lon_min, lon_max)
    features = compute(land_shape, lake_shape, shapely.Polygon(GULF_SIDE_LONLAT),
                       lat_min, lat_max, lon_min, lon_max)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, box=np.array([lat_min, lat_max, lon_min, lon_max]), **features)
    return features


def load(cache_path: Path) -> dict[str, np.ndarray]:
    """Load cached static features."""
    with np.load(cache_path) as data:
        return {name: data[name] for name in FEATURE_NAMES}
