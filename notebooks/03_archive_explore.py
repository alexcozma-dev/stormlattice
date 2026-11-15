# %% Checkpoint A: slot time sfor one storm day
from stormlattice import archive

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
