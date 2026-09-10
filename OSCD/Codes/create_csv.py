import os
import glob
import csv

# ============================================================
# CONFIGURATION
# ============================================================

root = os.getcwd()
PROJECT_ROOT = os.path.dirname(root)

PATCH_ROOT = os.path.join(
    PROJECT_ROOT,
    "Data",
    "patches"
)

IMAGE_ROOT = os.path.join(
    PATCH_ROOT,
    "images"
)

LABEL_ROOT = os.path.join(
    PATCH_ROOT,
    "labels"
)

CSV_ROOT = os.path.join(
    PROJECT_ROOT,
    "Data",
    "csv"
)

os.makedirs(CSV_ROOT, exist_ok=True)


# ============================================================
# DATASET SPLITS
# ============================================================

SPLITS = [
    "train",
    "val",
    "test"
]


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("CREATE CSV FILES FOR OSCD PATCH DATASET")
print("=" * 80)

print()
print("Project root:")
print(PROJECT_ROOT)

print()
print("Patch root:")
print(PATCH_ROOT)

print()
print("Image root:")
print(IMAGE_ROOT)

print()
print("Label root:")
print(LABEL_ROOT)

print()
print("CSV output:")
print(CSV_ROOT)

print("=" * 80)


# ============================================================
# FIND PATCHES
# ============================================================

def get_files(folder, pattern):
    """
    Return sorted TIFF files from a folder.
    """

    files = glob.glob(
        os.path.join(folder, pattern)
    )

    return sorted(files)


# ============================================================
# EXTRACT PATCH ID
# ============================================================

def get_patch_id(filename):
    """
    Extract city + patch number.

    Examples:

        aguasclaras_0_s1.tif
        aguasclaras_0_s2.tif
        aguasclaras_0_cm.tif

    return:

        aguasclaras_0
    """

    name = os.path.basename(filename)

    # Remove extension
    name = os.path.splitext(name)[0]

    if name.endswith("_s1"):
        return name[:-3]

    if name.endswith("_s2"):
        return name[:-3]

    if name.endswith("_cm"):
        return name[:-3]

    return None


# ============================================================
# CREATE CSV FOR ONE SPLIT
# ============================================================

def create_csv(split):

    print()
    print()
    print("#" * 80)
    print(f"PROCESSING {split.upper()}")
    print("#" * 80)

    image_dir = os.path.join(
        IMAGE_ROOT,
        split
    )

    label_dir = os.path.join(
        LABEL_ROOT,
        split
    )

    # --------------------------------------------------------
    # Check directories
    # --------------------------------------------------------

    if not os.path.exists(image_dir):

        print()
        print(f"ERROR: Image directory does not exist:")
        print(image_dir)

        return

    if not os.path.exists(label_dir):

        print()
        print(f"ERROR: Label directory does not exist:")
        print(label_dir)

        return

    # --------------------------------------------------------
    # Get image files
    # --------------------------------------------------------

    s1_files = get_files(
        image_dir,
        "*_s1.tif"
    )

    s2_files = get_files(
        image_dir,
        "*_s2.tif"
    )

    label_files = get_files(
        label_dir,
        "*_cm.tif"
    )

    print()
    print(f"S1 patches    : {len(s1_files)}")
    print(f"S2 patches    : {len(s2_files)}")
    print(f"Label patches : {len(label_files)}")

    # --------------------------------------------------------
    # Create dictionaries
    # --------------------------------------------------------

    s1_dict = {}

    for path in s1_files:

        patch_id = get_patch_id(path)

        if patch_id is not None:
            s1_dict[patch_id] = path


    s2_dict = {}

    for path in s2_files:

        patch_id = get_patch_id(path)

        if patch_id is not None:
            s2_dict[patch_id] = path


    label_dict = {}

    for path in label_files:

        patch_id = get_patch_id(path)

        if patch_id is not None:
            label_dict[patch_id] = path


    # --------------------------------------------------------
    # Find common patches
    # --------------------------------------------------------

    common_ids = sorted(
        set(s1_dict.keys())
        & set(s2_dict.keys())
        & set(label_dict.keys())
    )

    print()
    print(f"Matched patches: {len(common_ids)}")

    # --------------------------------------------------------
    # Report missing patches
    # --------------------------------------------------------

    missing_s2 = sorted(
        set(s1_dict.keys()) - set(s2_dict.keys())
    )

    missing_s1 = sorted(
        set(s2_dict.keys()) - set(s1_dict.keys())
    )

    missing_label = sorted(
        (set(s1_dict.keys()) & set(s2_dict.keys()))
        - set(label_dict.keys())
    )

    if missing_s2:

        print()
        print(
            f"Missing S2 patches: {len(missing_s2)}"
        )

        for patch_id in missing_s2[:10]:
            print("  ", patch_id)

    if missing_s1:

        print()
        print(
            f"Missing S1 patches: {len(missing_s1)}"
        )

        for patch_id in missing_s1[:10]:
            print("  ", patch_id)

    if missing_label:

        print()
        print(
            f"Missing labels: {len(missing_label)}"
        )

        for patch_id in missing_label[:10]:
            print("  ", patch_id)


    # ========================================================
    # CREATE CSV
    # ========================================================

    csv_path = os.path.join(
        CSV_ROOT,
        f"{split}.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        writer.writerow([
            "patch_id",
            "split",
            "sentinel1",
            "sentinel2",
            "label"
        ])

        # ----------------------------------------------------
        # Rows
        # ----------------------------------------------------

        for patch_id in common_ids:

            writer.writerow([
                patch_id,
                split,
                os.path.abspath(
                    s1_dict[patch_id]
                ),
                os.path.abspath(
                    s2_dict[patch_id]
                ),
                os.path.abspath(
                    label_dict[patch_id]
                )
            ])


    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("-" * 80)
    print(f"{split.upper()} CSV CREATED")
    print("-" * 80)

    print()
    print("CSV:")
    print(csv_path)

    print()
    print(f"Rows: {len(common_ids)}")


    # --------------------------------------------------------
    # Show first few rows
    # --------------------------------------------------------

    print()
    print("First patches:")

    for patch_id in common_ids[:5]:

        print()
        print(f"Patch: {patch_id}")
        print(
            "  S1:",
            os.path.basename(
                s1_dict[patch_id]
            )
        )
        print(
            "  S2:",
            os.path.basename(
                s2_dict[patch_id]
            )
        )
        print(
            "  Label:",
            os.path.basename(
                label_dict[patch_id]
            )
        )


# ============================================================
# PROCESS ALL SPLITS
# ============================================================

for split in SPLITS:

    create_csv(split)


# ============================================================
# FINISHED
# ============================================================

print()
print()
print("=" * 80)
print("ALL CSV FILES CREATED")
print("=" * 80)

print()
print("Output directory:")
print(CSV_ROOT)

print()
print("Files:")

for split in SPLITS:

    csv_path = os.path.join(
        CSV_ROOT,
        f"{split}.csv"
    )

    if os.path.exists(csv_path):

        print(
            f"  {split}.csv"
        )

print()
print("=" * 80)