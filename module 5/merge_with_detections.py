"""
Merges GPS coordinates (from extract_gps_from_photos.py) with your Module 1
YOLO model's detection counts, producing detections.csv — the exact input
cluster_hotspots.py expects.

Usage:
  python merge_with_detections.py /path/to/photos/ photo_gps.csv

Requires your trained Module 1 model at runs/detect/train*/weights/best.pt
(update MODEL_PATH below to match your actual path).
"""

import os
import sys
import pandas as pd
from ultralytics import YOLO

MODEL_PATH = "best.pt"  # update to your actual trained weights path


def main():
    if len(sys.argv) < 3:
        print("Usage: python merge_with_detections.py /path/to/photos/ photo_gps.csv")
        sys.exit(1)

    photo_dir = sys.argv[1]
    gps_csv = sys.argv[2]

    gps_df = pd.read_csv(gps_csv)
    model = YOLO(MODEL_PATH)

    rows = []
    for _, row in gps_df.iterrows():
        img_path = os.path.join(photo_dir, row["filename"])
        if not os.path.exists(img_path):
            continue

        results = model.predict(img_path, conf=0.35, verbose=False)
        # Count only "Plastic" class detections (class id 0, per your data.yaml)
        plastic_count = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)

        rows.append({
            "filename": row["filename"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "plastic_count": plastic_count,
        })

    df = pd.DataFrame(rows)
    df.to_csv("detections.csv", index=False)
    print(f"Processed {len(df)} geotagged photos.")
    print(f"Total plastic detections across all photos: {df['plastic_count'].sum()}")
    print("Saved: detections.csv")
    print("\nNext: python cluster_hotspots.py detections.csv")


if __name__ == "__main__":
    main()
