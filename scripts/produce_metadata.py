# pip install pillow pillow-heif
import os, csv
from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS
import pillow_heif

pillow_heif.register_heif_opener()

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "raw_photos")
DST = os.path.join(HERE, "dataset")

os.makedirs(DST, exist_ok=True)

# rough bounds of the test rectangle — tweak after you plot your points
LAT_MIN, LAT_MAX = 39.950, 39.955
LON_MIN, LON_MAX = -75.195, -75.188


def to_deg(val, ref):
    d, m, s = [float(x) for x in val]
    deg = d + m / 60 + s / 3600
    return -deg if ref in ("S", "W") else deg


def get_gps(img):
    gps = img.getexif().get_ifd(0x8825)
    if not gps:
        return None
    g = {GPSTAGS.get(k, k): v for k, v in gps.items()}
    try:
        return to_deg(g["GPSLatitude"], g["GPSLatitudeRef"]), to_deg(g["GPSLongitude"], g["GPSLongitudeRef"])
    except KeyError:
        return None


rows, no_gps, out_of_bounds, failed = [], [], [], []

for f in sorted(os.listdir(SRC)):
    if not f.lower().endswith((".heic", ".heif", ".jpg", ".jpeg", ".png")):
        continue
    src = os.path.join(SRC, f)
    try:
        img = Image.open(src)
        gps = get_gps(img)                       # read EXIF before any conversion
        if gps is None:
            no_gps.append(f); continue
        lat, lon = gps
        if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
            out_of_bounds.append((f, lat, lon)); continue

        img = ImageOps.exif_transpose(img).convert("RGB")   # apply rotation flag, drop alpha
        name = os.path.splitext(f)[0] + ".jpg"
        img.save(os.path.join(DST, name), "JPEG", quality=95)  # EXIF (incl. GPS) stripped here
        rows.append({"image_path": name, "Latitude": lat, "Longitude": lon})
    except Exception as e:
        failed.append((f, repr(e)))

with open(os.path.join(DST, "metadata.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["image_path", "Latitude", "Longitude"])
    w.writeheader(); w.writerows(rows)

print(f"kept {len(rows)} | no GPS {len(no_gps)} | out of bounds {len(out_of_bounds)} | failed {len(failed)}")
for lst, label in [(no_gps, "NO GPS"), (out_of_bounds, "OUT OF BOUNDS"), (failed, "FAILED")]:
    for item in lst[:10]:
        print(f"  [{label}] {item}")