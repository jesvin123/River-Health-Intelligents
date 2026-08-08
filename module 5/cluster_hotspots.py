"""
Clusters geo-tagged plastic detections into pollution hotspots using DBSCAN
with true haversine (great-circle) distance in meters — not raw lat/lon
degrees, which distort distance depending on latitude.

Input:  detections.csv with columns: latitude, longitude, plastic_count
        (one row per detection event — e.g. one drone photo's result,
        or one manual observation)
Output: hotspots.csv — cluster centers, total detections, severity rank
        detections_labeled.csv — original data with a cluster_id column added

Usage:
  python cluster_hotspots.py detections.csv
"""

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

EARTH_RADIUS_M = 6_371_000

# How close two detections must be (in meters) to count as the same hotspot.
# Tune this to your river's scale — 100-150m is reasonable for a river
# stretch; increase for a sparser sensor network, decrease for a dense one.
EPS_METERS = 120
MIN_SAMPLES = 2  # minimum detections to form a hotspot (not just noise)


def cluster_hotspots(df, eps_meters=EPS_METERS, min_samples=MIN_SAMPLES):
    """df must have 'latitude' and 'longitude' columns (decimal degrees)."""
    coords_rad = np.radians(df[["latitude", "longitude"]].values)

    # DBSCAN's haversine metric expects eps in radians
    eps_radians = eps_meters / EARTH_RADIUS_M

    db = DBSCAN(eps=eps_radians, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    df = df.copy()
    df["cluster_id"] = labels  # -1 means "noise", not part of any hotspot

    hotspot_rows = []
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster_points = df[df["cluster_id"] == cluster_id]
        hotspot_rows.append({
            "cluster_id": cluster_id,
            "center_lat": cluster_points["latitude"].mean(),
            "center_lon": cluster_points["longitude"].mean(),
            "num_detection_events": len(cluster_points),
            "total_plastic_count": cluster_points["plastic_count"].sum()
                if "plastic_count" in cluster_points.columns else len(cluster_points),
        })

    hotspots = pd.DataFrame(hotspot_rows)
    if len(hotspots) > 0:
        hotspots = hotspots.sort_values("total_plastic_count", ascending=False).reset_index(drop=True)
        hotspots["severity_rank"] = hotspots.index + 1

    return df, hotspots


def main():
    if len(sys.argv) < 2:
        print("Usage: python cluster_hotspots.py detections.csv")
        sys.exit(1)

    input_path = sys.argv[1]
    df = pd.read_csv(input_path)

    required_cols = {"latitude", "longitude"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Input CSV must have columns: {required_cols}. "
                          f"Found: {list(df.columns)}")

    labeled_df, hotspots = cluster_hotspots(df)

    n_noise = (labeled_df["cluster_id"] == -1).sum()
    n_clusters = len(hotspots)

    print(f"Found {n_clusters} hotspot(s) from {len(df)} detection events "
          f"({n_noise} events were isolated/noise, not part of any hotspot).")

    if n_clusters > 0:
        print("\nTop hotspots by severity:")
        print(hotspots.to_string(index=False))
    else:
        print("\nNo hotspots found — try increasing EPS_METERS if detections "
              "are spread further apart than expected, or check min_samples "
              "isn't higher than your densest cluster.")

    labeled_df.to_csv("detections_labeled.csv", index=False)
    hotspots.to_csv("hotspots.csv", index=False)
    print("\nSaved: detections_labeled.csv, hotspots.csv")


if __name__ == "__main__":
    main()
