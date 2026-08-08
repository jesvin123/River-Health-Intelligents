"""
Quick static plot of detections + hotspots — a fast sanity check before
building the full interactive GIS dashboard in Module 6.

Usage:
  python quick_plot.py
(run after cluster_hotspots.py has produced detections_labeled.csv and hotspots.csv)
"""

import pandas as pd
import matplotlib.pyplot as plt

labeled = pd.read_csv("detections_labeled.csv")
hotspots = pd.read_csv("hotspots.csv")

fig, ax = plt.subplots(figsize=(8, 8))

# Plot noise points in gray
noise = labeled[labeled["cluster_id"] == -1]
ax.scatter(noise["longitude"], noise["latitude"], c="gray", marker="x",
           s=60, label="Isolated (noise)")

# Plot clustered points, colored by cluster
clustered = labeled[labeled["cluster_id"] != -1]
scatter = ax.scatter(clustered["longitude"], clustered["latitude"],
                      c=clustered["cluster_id"], cmap="tab10", s=80,
                      edgecolors="black", label="Detections")

# Mark hotspot centers
if len(hotspots) > 0:
    ax.scatter(hotspots["center_lon"], hotspots["center_lat"],
               c="red", marker="*", s=400, edgecolors="black",
               linewidths=1.5, label="Hotspot center", zorder=5)
    for _, h in hotspots.iterrows():
        ax.annotate(f"#{int(h['severity_rank'])}",
                     (h["center_lon"], h["center_lat"]),
                     textcoords="offset points", xytext=(8, 8), fontsize=10, fontweight="bold")

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Pollution Detections & Hotspots")
ax.legend()
plt.tight_layout()
plt.savefig("hotspot_preview.png", dpi=150)
print("Saved: hotspot_preview.png")
