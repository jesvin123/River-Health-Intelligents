"""
Converts the merged COCO-format annotations (annotations_merged.json) into
YOLO-format .txt label files, using pylabel.

Run this AFTER 01_merge_categories.py, from the TACO/data folder.
Requires: pip install pylabel
"""

from pylabel import importer
import os

ANNOTATIONS_PATH = r"D:\bootcamp\TACO\data\annotations_merged.json"
OUTPUT_PATH = "yolo_dataset"


def main():
    if not os.path.exists(ANNOTATIONS_PATH):
        raise FileNotFoundError(
            f"{ANNOTATIONS_PATH} not found — run 01_merge_categories.py first."
        )

    print("Importing COCO annotations...")
    # The annotations' file_name already contains the batch subfolder
    # (e.g. "batch_1/000006.jpg"), so we don't pass path_to_images and let
    # pylabel resolve images relative to the annotations' parent folder.
    dataset = importer.ImportCoco(
        path=ANNOTATIONS_PATH,
        path_to_images="",
    )

    print(f"Loaded {len(dataset.df)} annotation rows across "
          f"{dataset.df['img_filename'].nunique()} images.")

    print("Exporting to YOLO format...")
    # pylabel nests files by the image's batch folder (e.g. batch_1/...).
    # Pre-create those subdirectories so the export doesn't fail.
    labels_dir = os.path.join(OUTPUT_PATH, "labels_temp")
    os.makedirs(labels_dir, exist_ok=True)
    images_dir = os.path.join(OUTPUT_PATH, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i in range(1, 16):
        os.makedirs(os.path.join(labels_dir, f"batch_{i}"), exist_ok=True)
        os.makedirs(os.path.join(images_dir, f"batch_{i}"), exist_ok=True)
    dataset.export.ExportToYoloV5(
        output_path=os.path.join(OUTPUT_PATH, "labels_temp"),
        copy_images=True,
        use_splits=False,  # we do our own split in 03_train_val_split.py
    )

    print(f"Done. YOLO-format labels + copied images are in {OUTPUT_PATH}/labels_temp")
    print("Next: run 03_train_val_split.py")


if __name__ == "__main__":
    main()
