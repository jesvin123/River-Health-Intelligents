# Module 1 — Ready-to-Run Package

I verified the real TACO `annotations.json` (1500 images, 4784 annotations, 60 categories) so every script here uses the actual category names and structure — not guesses.

## Why you must run this in Colab, not locally/here
TACO's images are hosted on Flickr (`flickr_url` field in the annotations), not bundled in the GitHub repo. `download.py` fetches them at runtime. Colab has full internet access; this prep sandbox does not — so step 1 below must run in Colab.

## Exact order to run things in Colab

### Step 1 — Clone TACO and download images
```python
!git clone https://github.com/pedropro/TACO.git
%cd TACO
!pip install -r requirements.txt
!python download.py
```
This takes 15–30 minutes depending on Flickr's response times. If some images fail to download (a few always do — old Flickr links occasionally break), that's normal; the scripts below just skip missing files.

### Step 2 — Upload the 4 files from this package into the TACO/data folder in Colab
- `01_merge_categories.py`
- `02_convert_to_yolo.py`
- `03_train_val_split.py`
- `data.yaml`

(Drag-and-drop into Colab's file browser on the left, or use `files.upload()`.)

### Step 3 — Run in this order
```python
!python 01_merge_categories.py
!pip install pylabel
!python 02_convert_to_yolo.py
!python 03_train_val_split.py
```

### Step 4 — Train
```python
!pip install ultralytics
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
results = model.train(data='data.yaml', epochs=100, imgsz=640, batch=16, patience=20, device=0)
```
Make sure Runtime → Change runtime type → GPU is set before this step.

### Step 5 — Add your own aerial-angle images (recommended)
Before training, if you've got any bridge/drone/raised-angle photos of plastic on water, drop them into `yolo_dataset/images/train/` and hand-label them quickly using Roboflow (free), exporting YOLO-format labels into `yolo_dataset/labels/train/`. Even 30–50 of these meaningfully reduces the domain-mismatch problem flagged in the main guide.
