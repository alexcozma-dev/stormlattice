"""
Features at issue time t0 for the birth model: one row per block, per issue time, per lead.

    static   distance to the Atlantic and Gulf coasts, water fraction, land mask (geography.py)
    clock    local solar time (from UTC and longitude, not a time zone), day of year
    lead     lead time L
    radar    neighbourhood maximum dBZ and its 10-minute trend, distance to the nearest storm,
             storm area within 20 km and 50 km

Sub-35 dBZ echoes are the strongest cheap predictor, so the radar features use the full
reflectivity range, not the storm mask alone.

Tables are written one file per storm day (data/tables/<YYYYMMDD>.npz), so a split is a list
of files. Validation and test days are kept complete; how training days are thinned is decided
in scripts/build_tables.py.

Distances and areas come from grid.py.

To write: the radar-state features, the clock features, and the table for one storm day.
"""
