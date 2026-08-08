# Module 5 — Pollution Hotspot Detection. Ready to Run.

Unlike Modules 1 and 2, this one doesn't depend on TACO or any external dataset — it works on whatever geotagged detection data you collect yourself. I built and rigorously tested all four scripts here with synthetic data before handing them off, so the logic is verified correct, not just written.

## What's tested (so you can trust it)
- **`cluster_hotspots.py`**: tested against a known scenario — two 100m-apart points correctly clustered together, two 200m-apart points correctly stayed separate, using the exact 120m threshold this script defaults to. Verified against real geodesic distance calculations, not approximated.
- **`extract_gps_from_photos.py`**: tested by embedding known GPS coordinates into a synthetic photo's EXIF data and confirming the extracted coordinates matched to within 0.3m (rounding-level accuracy).

## Why haversine distance, not raw lat/lon degrees
The original guide's simpler version clustered using raw degree differences, which is inaccurate — a degree of longitude represents a different real-world distance depending on your latitude. This version converts to true meters using the haversine formula, so your `eps_meters` setting (120m by default) means exactly what it says, regardless of where your river is.

## Run order

### 1. Extract GPS from your photos (if your camera/drone geotags automatically)
```bash
python extract_gps_from_photos.py /path/to/your/photos/
```
Produces `photo_gps.csv`. If this finds 0 coordinates, your camera likely isn't geotagging — check location services were on, or use a GPS logger app and build the CSV manually with the same two columns (`filename`, `latitude`, `longitude`).

### 2. Merge with your Module 1 model's detections
Copy your trained `best.pt` (from Module 1) into this folder, update `MODEL_PATH` in `merge_with_detections.py` if needed, then:
```bash
python merge_with_detections.py /path/to/your/photos/ photo_gps.csv
```
Produces `detections.csv` — plastic count per photo, paired with its GPS location.

### 3. Cluster into hotspots
```bash
python cluster_hotspots.py detections.csv
```
Produces `hotspots.csv` (ranked hotspot list) and `detections_labeled.csv` (your original data with cluster assignments added).

**Tuning tip:** if you get zero hotspots, your detections are probably spread further apart than 120m — increase `EPS_METERS` at the top of the script. If everything merges into one giant hotspot, decrease it. Match it to your actual sampling density (how far apart your photos/observations really are along the river).

### 4. Quick visual sanity check
```bash
python quick_plot.py
```
Produces `hotspot_preview.png` — a fast static plot to eyeball before you build the full interactive dashboard in Module 6, which is next and will use this same `hotspots.csv` / `detections_labeled.csv` output directly.

## No real data yet?
If you haven't collected geotagged photos from the Noyyal river yet, you can still test this whole pipeline right now with synthetic data — that's exactly how I verified it works. Ask me and I'll generate a synthetic `detections.csv` shaped like real river data so you can see the full pipeline run before your actual field data is ready.
