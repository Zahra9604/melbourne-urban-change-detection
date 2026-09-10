
import os
import glob
import csv
import random


# ============================================================
# CONFIGURATION
# ============================================================

root = os.getcwd()
PROJECT_ROOT = os.path.dirname(root)

PATCH_ROOT = os.path.join(
    PROJECT_ROOT,
    "MelbourneData"
)

IMAGE_ROOT = os.path.join(
    PATCH_ROOT,
    "images"
)

LABEL_ROOT = os.path.join(
    PATCH_ROOT,
    "change_labels_10m"
)

CSV_ROOT = os.path.join(
    PATCH_ROOT,
    "csv"
)

os.makedirs(CSV_ROOT, exist_ok=True)


# ============================================================
# DATASET SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# CHECK RATIOS
# ============================================================

assert abs(
    TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0
) < 1e-6, "Split ratios must sum to 1.0"


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("CREATE TRAIN / VAL / TEST CSV FILES")
print("MELBOURNE CHANGE DETECTION DATASET")
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

print()
print("Split ratios:")
print(f"  Train: {TRAIN_RATIO * 100:.0f}%")
print(f"  Val  : {VAL_RATIO * 100:.0f}%")
print(f"  Test : {TEST_RATIO * 100:.0f}%")

print()
print(f"Random seed: {RANDOM_SEED}")

print("=" * 80)


# ============================================================
# GET FILES
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
    Extract patch ID from Melbourne filenames.

    Examples
    --------
    melbourne_0000_s1_2023.tif
        -> melbourne_0000

    melbourne_0000_s2_2023.tif
        -> melbourne_0000

    melbourne_0000_s1_2025.tif
        -> melbourne_0000

    melbourne_0000_s2_2025.tif
        -> melbourne_0000

    melbourne_0000_change.tif
        -> melbourne_0000
    """

    name = os.path.basename(filename)

    # Remove extension
    name = os.path.splitext(name)[0]

    suffixes = [
        "_s1_2023",
        "_s2_2023",
        "_s1_2025",
        "_s2_2025",
        "_change_label_10m"
    ]

    for suffix in suffixes:

        if name.endswith(suffix):

            return name[:-len(suffix)]

    return None


# ============================================================
# BUILD FILE DICTIONARIES
# ============================================================

def build_dictionary(files):
    """
    Convert file list into:

        patch_id -> file path
    """

    dictionary = {}

    for path in files:

        patch_id = get_patch_id(path)

        if patch_id is not None:

            dictionary[patch_id] = path

    return dictionary


# ============================================================
# CREATE DATASET DICTIONARIES
# ============================================================

def load_dataset():

    print()
    print("#" * 80)
    print("LOADING DATASET")
    print("#" * 80)


    # --------------------------------------------------------
    # DIRECTORIES
    # --------------------------------------------------------

    s1_2023_dir = os.path.join(
        IMAGE_ROOT,
        "2023",
        "sentinel1"
    )

    s2_2023_dir = os.path.join(
        IMAGE_ROOT,
        "2023",
        "sentinel2"
    )

    s1_2025_dir = os.path.join(
        IMAGE_ROOT,
        "2025",
        "sentinel1"
    )

    s2_2025_dir = os.path.join(
        IMAGE_ROOT,
        "2025",
        "sentinel2"
    )

    label_dir = LABEL_ROOT


    # --------------------------------------------------------
    # CHECK DIRECTORIES
    # --------------------------------------------------------

    directories = {
        "S1 2023": s1_2023_dir,
        "S2 2023": s2_2023_dir,
        "S1 2025": s1_2025_dir,
        "S2 2025": s2_2025_dir,
        "Labels": label_dir
    }

    for name, directory in directories.items():

        if not os.path.exists(directory):

            raise FileNotFoundError(
                f"{name} directory does not exist:\n"
                f"{directory}"
            )


    # --------------------------------------------------------
    # GET FILES
    # --------------------------------------------------------

    s1_2023_files = get_files(
        s1_2023_dir,
        "*_s1_2023.tif"
    )

    s2_2023_files = get_files(
        s2_2023_dir,
        "*_s2_2023.tif"
    )

    s1_2025_files = get_files(
        s1_2025_dir,
        "*_s1_2025.tif"
    )

    s2_2025_files = get_files(
        s2_2025_dir,
        "*_s2_2025.tif"
    )

    label_files = get_files(
        label_dir,
        "*change_label_10m.tif"
    )


    # --------------------------------------------------------
    # PRINT COUNTS
    # --------------------------------------------------------

    print()
    print(f"S1 2023 patches : {len(s1_2023_files)}")
    print(f"S2 2023 patches : {len(s2_2023_files)}")
    print(f"S1 2025 patches : {len(s1_2025_files)}")
    print(f"S2 2025 patches : {len(s2_2025_files)}")
    print(f"Label patches   : {len(label_files)}")


    # --------------------------------------------------------
    # CREATE DICTIONARIES
    # --------------------------------------------------------

    s1_2023_dict = build_dictionary(
        s1_2023_files
    )

    s2_2023_dict = build_dictionary(
        s2_2023_files
    )

    s1_2025_dict = build_dictionary(
        s1_2025_files
    )

    s2_2025_dict = build_dictionary(
        s2_2025_files
    )

    label_dict = build_dictionary(
        label_files
    )


    # --------------------------------------------------------
    # FIND COMPLETE PATCHES
    # --------------------------------------------------------

    common_ids = sorted(
        set(s1_2023_dict.keys())
        & set(s2_2023_dict.keys())
        & set(s1_2025_dict.keys())
        & set(s2_2025_dict.keys())
        & set(label_dict.keys())
    )


    print()
    print("-" * 80)
    print(f"COMPLETE PATCHES: {len(common_ids)}")
    print("-" * 80)


    return (
        s1_2023_dict,
        s2_2023_dict,
        s1_2025_dict,
        s2_2025_dict,
        label_dict,
        common_ids
    )


# ============================================================
# SPLIT PATCH IDS
# ============================================================

def split_dataset(patch_ids):

    print()
    print("#" * 80)
    print("SPLITTING DATASET")
    print("#" * 80)


    # --------------------------------------------------------
    # COPY IDS
    # --------------------------------------------------------

    patch_ids = list(patch_ids)


    # --------------------------------------------------------
    # SHUFFLE
    # --------------------------------------------------------

    random.seed(RANDOM_SEED)

    random.shuffle(patch_ids)


    # --------------------------------------------------------
    # TOTAL NUMBER
    # --------------------------------------------------------

    total = len(patch_ids)


    # --------------------------------------------------------
    # CALCULATE SPLIT SIZES
    # --------------------------------------------------------

    train_size = int(
        total * TRAIN_RATIO
    )

    val_size = int(
        total * VAL_RATIO
    )


    # --------------------------------------------------------
    # SPLIT
    # --------------------------------------------------------

    train_ids = patch_ids[
        :train_size
    ]

    val_ids = patch_ids[
        train_size:
        train_size + val_size
    ]

    test_ids = patch_ids[
        train_size + val_size:
    ]


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print(f"Total patches : {total}")
    print()
    print(f"Train patches : {len(train_ids)}")
    print(f"Val patches   : {len(val_ids)}")
    print(f"Test patches  : {len(test_ids)}")
    print()
    print(
        f"Total after split: "
        f"{len(train_ids) + len(val_ids) + len(test_ids)}"
    )


    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids)
    }


# ============================================================
# CREATE CSV
# ============================================================

def create_csv(
    split,
    patch_ids,
    s1_2023_dict,
    s2_2023_dict,
    s1_2025_dict,
    s2_2025_dict,
    label_dict
):

    csv_path = os.path.join(
        CSV_ROOT,
        f"{split}.csv"
    )


    # --------------------------------------------------------
    # WRITE CSV
    # --------------------------------------------------------

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)


        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        writer.writerow([
            "patch_id",
            "split",
            "sentinel1_2023",
            "sentinel2_2023",
            "sentinel1_2025",
            "sentinel2_2025",
            "label"
        ])


        # ----------------------------------------------------
        # ROWS
        # ----------------------------------------------------

        for patch_id in patch_ids:

            writer.writerow([

                patch_id,

                split,

                os.path.abspath(
                    s1_2023_dict[patch_id]
                ),

                os.path.abspath(
                    s2_2023_dict[patch_id]
                ),

                os.path.abspath(
                    s1_2025_dict[patch_id]
                ),

                os.path.abspath(
                    s2_2025_dict[patch_id]
                ),

                os.path.abspath(
                    label_dict[patch_id]
                )
            ])


    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print()
    print("-" * 80)
    print(f"{split.upper()} CSV CREATED")
    print("-" * 80)

    print()
    print("CSV path:")
    print(csv_path)

    print()
    print(f"Rows: {len(patch_ids)}")


    # --------------------------------------------------------
    # SHOW FIRST 3
    # --------------------------------------------------------

    print()
    print("First patches:")

    for patch_id in patch_ids[:3]:

        print()
        print(f"Patch: {patch_id}")

        print(
            "  S1 2023:",
            os.path.basename(
                s1_2023_dict[patch_id]
            )
        )

        print(
            "  S2 2023:",
            os.path.basename(
                s2_2023_dict[patch_id]
            )
        )

        print(
            "  S1 2025:",
            os.path.basename(
                s1_2025_dict[patch_id]
            )
        )

        print(
            "  S2 2025:",
            os.path.basename(
                s2_2025_dict[patch_id]
            )
        )

        print(
            "  Label:",
            os.path.basename(
                label_dict[patch_id]
            )
        )


    return csv_path


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # LOAD ALL COMPLETE PATCHES
    # --------------------------------------------------------

    (
        s1_2023_dict,
        s2_2023_dict,
        s1_2025_dict,
        s2_2025_dict,
        label_dict,
        common_ids
    ) = load_dataset()


    # --------------------------------------------------------
    # SPLIT DATASET
    # --------------------------------------------------------

    splits = split_dataset(
        common_ids
    )


    # --------------------------------------------------------
    # CREATE CSV FILES
    # --------------------------------------------------------

    csv_paths = {}


    for split in [
        "train",
        "val",
        "test"
    ]:

        csv_paths[split] = create_csv(

            split,

            splits[split],

            s1_2023_dict,
            s2_2023_dict,
            s1_2025_dict,
            s2_2025_dict,
            label_dict
        )


    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print()
    print("=" * 80)
    print("DATASET SPLITTING COMPLETE")
    print("=" * 80)

    print()
    print("Dataset:")
    print(f"  Total : {len(common_ids)}")
    print(f"  Train : {len(splits['train'])}")
    print(f"  Val   : {len(splits['val'])}")
    print(f"  Test  : {len(splits['test'])}")

    print()
    print("CSV files:")

    for split in [
        "train",
        "val",
        "test"
    ]:

        print(
            f"  {split}: {csv_paths[split]}"
        )

    print()
    print("=" * 80)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

