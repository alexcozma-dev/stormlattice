import numpy as np

from stormlattice import archive
from stormlattice.config import data_dir


# %% Checkpoint A: slot time sfor one storm day

t = archive.slot_times("20240715", 10, 10)
print("slots:", len(t))
print("first:", t[0].isoformat())
print("last:", t[-1].isoformat())
print("UTC dates touched:", sorted({ x.strftime("%Y%m%d") for x in t }))



# %% Checkpoint B: real keys for one storm day

keys = archive.frame_keys("20240715", 10)
print("frames:", len(keys))
print("first:", keys[0].rsplit("_", 1)[-1])
print("last:", keys[-1].rsplit("_", 1)[-1])



# %% Checkpoint C: nearest-frame matching on a hand-made case, then a real day

F = "noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50/20240715/"
fake = [ F + "MRMS_MergedReflectivityQCComposite_00.50_20240715-" + t + ".grib2.gz" for t in ("100038", "101838", "103538") ]
slots = archive.slot_times("20240715", 10, 10)[:4]
for s, k in zip(slots, archive.match_slots(slots, fake, 4)):
    print(s.strftime("%H%M"), "->", k.rsplit("-", 1)[-1][:6] if k else None)

matched = archive.match_slots(archive.slot_times("20240715", 10, 10), keys, 4)
print("real day: empty slots =", sum(k is None for k in matched), "of", len(matched))



# %% Checkpoint D: one real frame through the whole chain

grib_dir = data_dir() / "grib"
grib_dir.mkdir(parents=True, exist_ok=True)
cells = archive.load_cells(matched[48], grib_dir)       # slot 48 = 18:00 UTC = 2 pm EDT
print("shape:", cells.shape, cells.dtype)
print("max dBZ:", np.nanmax(cells), " no-coverage cells:", int(np.isnan(cells).sum()))
print("storm cells (>= 35 dBZ):", int((cells >= 35).sum()))
print("GRIBs left:", list(grib_dir.iterdir()))
