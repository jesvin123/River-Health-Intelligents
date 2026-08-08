"""
Extracts GPS coordinates from a folder of photos' EXIF metadata (works for
phone photos and most drones that geotag automatically) and builds a base
CSV you can then merge with your Module 1 YOLO detection counts.

Usage:
  python extract_gps_from_photos.py /path/to/photos/

Output: photo_gps.csv with columns: filename, latitude, longitude

If your camera/drone does NOT geotag photos automatically, this will find
zero coordinates — in that case, use a GPS logging app while shooting and
manually build a CSV with the same columns instead.
"""

import os
import sys
import piexif
import pandas as pd


def dms_to_decimal(dms, ref):
    """Convert EXIF's degrees/minutes/seconds format to decimal degrees."""
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    decimal = degrees + minutes / 60 + seconds / 3600
    if ref in (b"S", b"W", "S", "W"):
        decimal = -decimal
    return decimal


def get_gps_from_image(image_path):
    try:
        exif_dict = piexif.load(image_path)
    except Exception:
        return None

    gps = exif_dict.get("GPS")
    if not gps or piexif.GPSIFD.GPSLatitude not in gps:
        return None

    try:
        lat = dms_to_decimal(gps[piexif.GPSIFD.GPSLatitude], gps[piexif.GPSIFD.GPSLatitudeRef])
        lon = dms_to_decimal(gps[piexif.GPSIFD.GPSLongitude], gps[piexif.GPSIFD.GPSLongitudeRef])
        return lat, lon
    except (KeyError, ZeroDivisionError):
        return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_gps_from_photos.py /path/to/photos/")
        sys.exit(1)

    photo_dir = sys.argv[1]
    rows = []
    no_gps = []

    for fname in sorted(os.listdir(photo_dir)):
        if not fname.lower().endswith((".jpg", ".jpeg")):
            continue  # EXIF GPS reliably only works with JPEG, not PNG

        full_path = os.path.join(photo_dir, fname)
        result = get_gps_from_image(full_path)

        if result is None:
            no_gps.append(fname)
        else:
            lat, lon = result
            rows.append({"filename": fname, "latitude": lat, "longitude": lon})

    df = pd.DataFrame(rows)
    df.to_csv("photo_gps.csv", index=False)

    print(f"Found GPS data for {len(rows)} photos.")
    if no_gps:
        print(f"{len(no_gps)} photos had NO GPS data (location services were "
              f"probably off when taken): {no_gps[:5]}{'...' if len(no_gps) > 5 else ''}")
    print("Saved: photo_gps.csv")
    print("\nNext: merge this with your Module 1 detection counts (per-image "
          "plastic count from YOLO) to build detections.csv, then run "
          "cluster_hotspots.py on it.")


if __name__ == "__main__":
    main()
