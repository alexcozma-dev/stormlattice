# 01_mrms_explore.py - exploring NOAA MRMS radar data on AWS
# each block run separate in console with alt+shift+e

# %% checkpoint A: browsing the MRMS bucket
import s3fs

fs = s3fs.S3FileSystem(anon=True) #anonymous connection to Amazon S3 (NOAA buckets are public)
print(fs.ls("noaa-mrms-pds")) #top level of the bucket: one folder per region

#all products for continental US
products = fs.ls("noaa-mrms-pds/CONUS")
print(f"{len(products)} CONUS products")
print([p for p in products if "Reflectivity" in p]) #keeps only the reflectivity products

#one folder per day (UTC dates) for composite reflectivity
days = fs.ls("noaa-mrms-pds/CONUS/MergedReflectivityQCComposite_00.50")
print("oldest days:", days[:3]) #oldest 3
print("newest days:", days[-3:]) #newest 3

#files in the newest day (one every ~2 min, 24*60=720 per full day)
files = fs.ls(days[-1])
print(f"{len(files)} files today; newest: {files[-3:]}")

# %% checkpoint B: download and decompress the newest frame
from datetime import datetime, timezone
import gzip
import shutil
from pathlib import Path

#newest file (names sort by time because of the YYYYMMDD-HHMMSS format)
latest = files[-1]
name = latest.split("/")[-1] #just the filename, no folders
print("newest file:", name)

#parse the timestamp from the filename and label it as UTC
stamp = name.split("_")[-1].replace(".grib2.gz", "")
t = datetime.strptime(stamp, "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)

#frame time in UTC, in local time, and how old the data is
print("frame time UTC: ", t)
print("frame time local:", t.astimezone())

age = datetime.now(timezone.utc) - t
print(f"data age: {age.total_seconds() / 60:.1f} min") #age in minutes

#local folder for downloads (inside gitignored data/)
print("working dir:", Path.cwd()) #should be the project root
out_dir = Path("data/mrms")
out_dir.mkdir(parents=True, exist_ok=True)

#download the compressed file from S3
gz_path = out_dir / name
if not gz_path.exists():
    fs.get(latest, str(gz_path))

#decompress .grib2.gz -> .grib2 (streamed, files autoclosed by `with`)
grib_path = gz_path.with_suffix("")
if not grib_path.exists():
    with gzip.open(gz_path, "rb") as src, open(grib_path, "wb") as dst:
        shutil.copyfileobj(src, dst)

#file sizes in MB: compressed vs. decompressed
print(f"sizes: {gz_path.stat().st_size / 1e6:.2f} MB gz, "
      f"{grib_path.stat().st_size / 1e6:.2f} MB grib2")

# %% checkpoint C: open the grib2 file and inspect the grid
import numpy as np
import xarray as xr

#open with the cfgrib engine; indexpath="" stops it writing a .idx file next to the data
ds = xr.open_dataset(grib_path, engine="cfgrib", backend_kwargs={"indexpath": ""})
print(ds)

#the dataset holds one variable; grab it by position (MRMS names are often unhelpful)
var_name = list(ds.data_vars)[0]
da = ds[var_name]
print(f"variable name: {var_name}, dims: {da.dims}, shape: {da.shape}, dtype: {da.dtype}")
print(f"memory: {da.nbytes / 1e6:.0f} MB")

#coordinate conventions (watch the direction and the range)
print("first/last latitudes: ", da.latitude.values[:2], da.latitude.values[-2:])
print("first/last longitudes: ", da.longitude.values[:2], da.longitude.values[-2:])

#pull the numbers into a plain Numpy array and look for special negative codes
vals = da.values
print("negative codes:", np.unique(vals[vals < 0])[:10])
print(f"max reflectivity: {vals.max():.1f} dBZ")
print(f"share of US with >= 35 dBZ: {(vals >= 35).mean() * 100:.3f} %")

# %% checkpoint D: crop to Florida and mask missing data
#florida box in normal (-180..180) longitudes
lat_min, lat_max = 24.5, 31.0
lon_min, lon_max = -87.5, -80.0

#latitude runs north->south, so the slice goes max -> min;
# % 360 converts negative longitudes to the file's 0..360 convention
fl = da.sel(latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min % 360, lon_max % 360))
print("Florida shape:", fl.shape)

#two different flags: -999 = no radar coverage (unknown)
#                     -99  = covered but no echo (clean air)
missing = fl == -999
no_echo = fl == -99
print(f"missing (no coverage): {float(missing.mean()) * 100:.1f} %")
print(f"no echo (clean air): {float(no_echo.mean()) * 100:.1f} %")
fl = fl.where(~missing)
fl = fl.where(~no_echo, -32.0)

#switch longitudes back to -180..180 so maps and labels read normally
fl = fl.assign_coords(longitude=fl.longitude - 360)
print("longitude range:", float(fl.longitude.min()), float(fl.longitude.max()))

print(f"Florida max: {float(fl.max()):.1f} dBZ")
print(f"Florida pixels >= 35 dBZ: {int((fl >= 35).sum())}")
print("min after masking:", float(fl.min())) # should be -32.0, not -999

# %% checkpoint E: plot Florida reflectivity and save a PNG
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 7))

#hide echoes under 5dBZ (drizzle, clutter) so storms stand out
show = fl.where(fl >= 5)
im = ax.pcolormesh(fl.longitude, fl.latitude, show,
                   cmap="turbo", vmin=0, vmax=70, shading="auto")
fig.colorbar(im, ax=ax, label="Composite reflectivity (dBZ)")

#grey wherever there's no radar coverage, so "unknown" never looks like "clear"
ax.pcolormesh(fl.longitude, fl.latitude, fl.isnull().where(fl.isnull()),
              cmap="Greys", vmin=0, vmax=2, shading="auto")

ax.set_title(f"MRMS composite reflectivity, {t:%Y-%m-%d %H:%M} UTC")
ax.set_xlabel("Longitude (°)")
ax.set_ylabel("Latitude (°)")
#a degree of longitude is shorter than a degree of latitude (by cos(lat)), so correct the shape
ax.set_aspect(1 / np.cos(np.deg2rad(27.75)))

out_png = Path("data/latest.png")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print("saved:", out_png.resolve())
plt.show()