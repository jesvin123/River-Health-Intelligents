"""
Splits the YOLO-format images/labels (from 02_convert_to_yolo.py) into
train/ and val/ folders (80/20), matching the structure data.yaml expects:

yolo_dataset/
├── images/train, images/val
└── labels/train, labels/val

Run this AFTER 02_convert_to_yolo.py.
"""

import os
import shutil
import random

# pylabel writes images into yolo_dataset/images/batch_* and labels into
# yolo_dataset/labels_temp/batch_* — each nested by its source batch folder.
IMAGES_ROOT = r"D:\bootcamp\yolo_dataset\images"
LABELS_ROOT = r"D:\bootcamp\yolo_dataset\labels_temp"
DEST_DIR = r"D:\bootcamp\yolo_dataset"
SPLIT_RATIO = 0.8
SEED = 42


def gather_files(root):
    """Return a list of (batch_name, filename) for every file under root/batch_*."""
    items = []
    if not os.path.isdir(root):
        return items
    for batch in sorted(os.listdir(root)):
        batch_dir = os.path.join(root, batch)
        if not os.path.isdir(batch_dir):
            continue
        for fname in sorted(os.listdir(batch_dir)):
            items.append((batch, fname))
    return items


def main():
    random.seed(SEED)

    image_items = gather_files(IMAGES_ROOT)
    if len(image_items) == 0:
        raise RuntimeError(
            f"No images found under {IMAGES_ROOT}. "
            "This usually means download.py didn't finish, or images failed "
            "to fetch from Flickr. Check TACO/data/batch_* folders."
        )

    random.shuffle(image_items)
    split_idx = int(SPLIT_RATIO * len(image_items))
    train_items, val_items = image_items[:split_idx], image_items[split_idx:]

    total_labels = 0
    for split_name, item_list in [("train", train_items), ("val", val_items)]:
        img_out = os.path.join(DEST_DIR, "images", split_name)
        lbl_out = os.path.join(DEST_DIR, "labels", split_name)
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        missing_labels = 0
        for batch, fname in item_list:
            # Prefix with the batch name so flattened filenames stay unique
            # (each batch reuses 000000.jpg, 000001.jpg, ...).
            stem, ext = os.path.splitext(fname)
            unique_name = f"{batch}_{stem}{ext.lower()}"

            src_img = os.path.join(IMAGES_ROOT, batch, fname)
            shutil.copy(src_img, os.path.join(img_out, unique_name))

            label_src = os.path.join(LABELS_ROOT, batch, f"{stem}.txt")
            if os.path.exists(label_src):
                shutil.copy(label_src, os.path.join(lbl_out, f"{batch}_{stem}.txt"))
                total_labels += 1
            else:
                missing_labels += 1

        print(f"{split_name}: {len(item_list)} images, "
              f"{missing_labels} without labels (skipped)")

    print(f"\nDone. Total: {len(train_items)} train / {len(val_items)} val images. "
          f"({total_labels} label files copied)")
    print(f"Dataset ready at: {DEST_DIR}/")
    print("Next: copy data.yaml into this folder (update the 'path' field to your "
          "Colab path), then start training (Step 4 in 00_START_HERE.md).")


if __name__ == "__main__":
    main()
