"""
Merges TACO's 60 fine-grained categories into 6 supercategories suitable
for training with a limited dataset (1500 images).

Run this from inside the TACO/data folder, after download.py has completed.
Input:  annotations.json (original, 60 categories)
Output: annotations_merged.json (6 categories) — used by 02_convert_to_yolo.py
"""

import json

INPUT_PATH = r"D:\bootcamp\TACO\data\annotations.json"
OUTPUT_PATH = r"D:\bootcamp\TACO\data\annotations_merged.json"

# Verified against the real TACO category list (id -> name), all 60 accounted for.
CATEGORY_NAME_TO_SUPER = {
    "Aluminium foil": "Metal",
    "Battery": "Organic_Other",
    "Aluminium blister pack": "Metal",
    "Carded blister pack": "Plastic",
    "Other plastic bottle": "Plastic",
    "Clear plastic bottle": "Plastic",
    "Glass bottle": "Glass",
    "Plastic bottle cap": "Plastic",
    "Metal bottle cap": "Metal",
    "Broken glass": "Glass",
    "Food Can": "Metal",
    "Aerosol": "Metal",
    "Drink can": "Metal",
    "Toilet tube": "Paper_Cardboard",
    "Other carton": "Paper_Cardboard",
    "Egg carton": "Paper_Cardboard",
    "Drink carton": "Paper_Cardboard",
    "Corrugated carton": "Paper_Cardboard",
    "Meal carton": "Paper_Cardboard",
    "Pizza box": "Paper_Cardboard",
    "Paper cup": "Paper_Cardboard",
    "Disposable plastic cup": "Plastic",
    "Foam cup": "Plastic",
    "Glass cup": "Glass",
    "Other plastic cup": "Plastic",
    "Food waste": "Organic_Other",
    "Glass jar": "Glass",
    "Plastic lid": "Plastic",
    "Metal lid": "Metal",
    "Other plastic": "Plastic",
    "Magazine paper": "Paper_Cardboard",
    "Tissues": "Paper_Cardboard",
    "Wrapping paper": "Paper_Cardboard",
    "Normal paper": "Paper_Cardboard",
    "Paper bag": "Paper_Cardboard",
    "Plastified paper bag": "Plastic",
    "Plastic film": "Plastic",
    "Six pack rings": "Plastic",
    "Garbage bag": "Plastic",
    "Other plastic wrapper": "Plastic",
    "Single-use carrier bag": "Plastic",
    "Polypropylene bag": "Plastic",
    "Crisp packet": "Plastic",
    "Spread tub": "Plastic",
    "Tupperware": "Plastic",
    "Disposable food container": "Plastic",
    "Foam food container": "Plastic",
    "Other plastic container": "Plastic",
    "Plastic glooves": "Plastic",
    "Plastic utensils": "Plastic",
    "Pop tab": "Metal",
    "Rope & strings": "Organic_Other",
    "Scrap metal": "Metal",
    "Shoe": "Organic_Other",
    "Squeezable tube": "Plastic",
    "Plastic straw": "Plastic",
    "Paper straw": "Paper_Cardboard",
    "Styrofoam piece": "Plastic",
    "Unlabeled litter": "Organic_Other",
    "Cigarette": "Cigarette",
}

# Fixed order matters — this becomes your YOLO class id order (0-5)
NEW_CATEGORIES = ["Plastic", "Metal", "Paper_Cardboard", "Glass", "Organic_Other", "Cigarette"]
NEW_CAT_NAME_TO_ID = {name: i for i, name in enumerate(NEW_CATEGORIES)}


def main():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    old_id_to_new_id = {}
    unmapped = []
    for cat in data["categories"]:
        super_name = CATEGORY_NAME_TO_SUPER.get(cat["name"])
        if super_name is None:
            unmapped.append(cat["name"])
            super_name = "Organic_Other"  # safety fallback
        old_id_to_new_id[cat["id"]] = NEW_CAT_NAME_TO_ID[super_name]

    if unmapped:
        print(f"WARNING: {len(unmapped)} categories had no explicit mapping and were "
              f"put in Organic_Other: {unmapped}")

    kept_annotations = 0
    for ann in data["annotations"]:
        ann["category_id"] = old_id_to_new_id[ann["category_id"]]
        kept_annotations += 1

    data["categories"] = [
        {"id": i, "name": name, "supercategory": name}
        for i, name in enumerate(NEW_CATEGORIES)
    ]

    with open(OUTPUT_PATH, "w") as f:
        json.dump(data, f)

    print(f"Done. {kept_annotations} annotations remapped into {len(NEW_CATEGORIES)} classes.")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Class order (this is your YOLO class id order): {NEW_CATEGORIES}")


if __name__ == "__main__":
    main()
