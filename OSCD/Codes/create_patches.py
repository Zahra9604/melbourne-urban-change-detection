# ============================================================
# OSCD PATCH CREATION
#
# 1. Find cities automatically using os + glob
# 2. Read:
#       S1 T1
#       S1 T2
#       S2 T1
#       S2 T2
#       Label
#
# 3. Stack:
#       S1 T1 + S1 T2 -> 4 bands
#       S2 T1 + S2 T2 -> 8 bands
#
# 4. Convert OSCD labels:
#       original 1 -> 0 (NO CHANGE)
#       original 2 -> 1 (CHANGE)
#
# 5. Reproject/resample label to S2 grid
# 6. Create 256 x 256 patches
# 7. Save images and labels in:
#
#       Data/patches/images/train
#       Data/patches/images/val
#       Data/patches/images/test
#
#       Data/patches/labels/train
#       Data/patches/labels/val
#       Data/patches/labels/test
#
# Image names:
#       brasilia_0_s1.tif
#       brasilia_0_s2.tif
#
# Label:
#       brasilia_0_cm.tif
#
# ============================================================

import os
import glob
import numpy as np
import rasterio

from rasterio.windows import Window
from rasterio.warp import reproject
from rasterio.enums import Resampling


# ============================================================
# PROJECT PATHS
# ============================================================
root = os.getcwd()
PROJECT_ROOT = os.path.dirname(root)

DATA_DIR = os.path.join(PROJECT_ROOT, "Data")

IMAGES_DIR = os.path.join(DATA_DIR, "images")
LABELS_DIR = os.path.join(DATA_DIR, "labels")

PATCHES_DIR = os.path.join(DATA_DIR, "patches")

# Output folders
OUTPUT_IMAGES = os.path.join(PATCHES_DIR, "images")
OUTPUT_LABELS = os.path.join(PATCHES_DIR, "labels")


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 256

# 128 means 50% overlap
STRIDE = 128

# Keep patches containing only 0 or only 1.
# We are NOT skipping no-change patches.
SAVE_EMPTY_LABEL_PATCHES = True

# If True:
#   S1 T1 + S1 T2 = 4 bands
#   S2 T1 + S2 T2 = 8 bands
STACK_IMAGES = True


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "val", "test"]:

    os.makedirs(
        os.path.join(OUTPUT_IMAGES, split),
        exist_ok=True
    )

    os.makedirs(
        os.path.join(OUTPUT_LABELS, split),
        exist_ok=True
    )


# ============================================================
# PRINT HEADER
# ============================================================

print()
print("=" * 80)
print("OSCD PATCH CREATION")
print("=" * 80)

print(f"Project root : {PROJECT_ROOT}")
print(f"Data         : {DATA_DIR}")
print(f"Patch size   : {PATCH_SIZE} x {PATCH_SIZE}")
print(f"Stride       : {STRIDE}")
print()
print("Image stacking:")
print("  S1 T1 + S1 T2 -> 4 bands")
print("  S2 T1 + S2 T2 -> 8 bands")
print()
print("Label conversion:")
print("  Original label 1 -> 0 = NO CHANGE")
print("  Original label 2 -> 1 = CHANGE")
print()
print("Output:")
print(OUTPUT_IMAGES)
print(OUTPUT_LABELS)
print("=" * 80)


# ============================================================
# FIND CITY NAMES
# ============================================================

def find_cities(split):

    split_image_dir = os.path.join(IMAGES_DIR, split)
    split_label_dir = os.path.join(LABELS_DIR, split)

    pattern = os.path.join(
        split_image_dir,
        "sentinel1_*_t1.tif"
    )

    files = glob.glob(pattern)

    cities = []

    for path in files:

        filename = os.path.basename(path)

        # sentinel1_brasilia_t1.tif
        name = filename.replace(
            "sentinel1_",
            ""
        ).replace(
            "_t1.tif",
            ""
        )

        cities.append(name)

    cities = sorted(set(cities))

    return cities


# ============================================================
# FIND FILES FOR ONE CITY
# ============================================================

def get_city_files(split, city):

    image_dir = os.path.join(IMAGES_DIR, split)
    label_dir = os.path.join(LABELS_DIR, split)

    files = {

        "s1_t1": os.path.join(
            image_dir,
            f"sentinel1_{city}_t1.tif"
        ),

        "s1_t2": os.path.join(
            image_dir,
            f"sentinel1_{city}_t2.tif"
        ),

        "s2_t1": os.path.join(
            image_dir,
            f"sentinel2_{city}_t1.tif"
        ),

        "s2_t2": os.path.join(
            image_dir,
            f"sentinel2_{city}_t2.tif"
        ),

        "label": os.path.join(
            label_dir,
            f"{city}_cm.tif"
        )
    }

    return files


# ============================================================
# CHECK FILES
# ============================================================

def check_files(files):

    all_exist = True

    for key, path in files.items():

        if os.path.exists(path):

            print(f"  {key:<8}: OK")

        else:

            print(f"  {key:<8}: MISSING")
            print(f"             {path}")

            all_exist = False

    return all_exist


# ============================================================
# PRINT RASTER INFORMATION
# ============================================================

def print_raster_info(name, src):

    print()
    print(name)

    print(f"  CRS       : {src.crs}")
    print(f"  Size      : {src.width} x {src.height}")
    print(f"  Bands     : {src.count}")
    print(f"  Resolution: {src.res}")
    print(f"  Nodata    : {src.nodata}")


# ============================================================
# CHECK IMAGE ALIGNMENT
# ============================================================

def check_image_alignment(s2_t1, s2_t2, s1_t1, s1_t2):

    reference = s2_t1

    datasets = {
        "S2 T2": s2_t2,
        "S1 T1": s1_t1,
        "S1 T2": s1_t2,
    }

    for name, src in datasets.items():

        same_crs = src.crs == reference.crs
        same_width = src.width == reference.width
        same_height = src.height == reference.height
        same_transform = src.transform == reference.transform

        if (
            same_crs
            and same_width
            and same_height
            and same_transform
        ):

            print(f"  {name}: aligned")

        else:

            print(f"  {name}: NOT aligned")

            print(
                f"      CRS: "
                f"{src.crs == reference.crs}"
            )

            print(
                f"      Size: "
                f"{src.width == reference.width and src.height == reference.height}"
            )

            print(
                f"      Transform: "
                f"{src.transform == reference.transform}"
            )

            return False

    return True


# ============================================================
# CONVERT LABEL TO BINARY
# ============================================================
def create_binary_label(label_src, reference_src):
    """
    Create a binary OSCD label in memory.

    Original OSCD:
        1 = no change
        2 = change

    Binary:
        0 = no change
        1 = change

    If the label has a CRS, it is reprojected to the
    Sentinel-2 reference grid.

    If the label has NO CRS but has exactly the same
    width/height as the Sentinel-2 image, we assume that
    it is already pixel-aligned and directly use its
    pixel grid.

    The binary label is NOT saved as a separate file.
    """

    # ========================================================
    # READ ORIGINAL LABEL
    # ========================================================

    original_label = label_src.read(1)

    print()
    print("  Original label values:")

    unique, counts = np.unique(
        original_label,
        return_counts=True
    )

    for value, count in zip(unique, counts):

        print(
            f"      {value}: "
            f"{count:,} pixels"
        )

    # ========================================================
    # CONVERT TO BINARY
    #
    # OSCD:
    #   1 -> no change
    #   2 -> change
    #
    # Binary:
    #   0 -> no change
    #   1 -> change
    # ========================================================

    binary_original = np.zeros(
        original_label.shape,
        dtype=np.uint8
    )

    binary_original[original_label == 2] = 1

    # ========================================================
    # CASE 1:
    # LABEL HAS NO CRS
    # ========================================================

    if label_src.crs is None:

        print()
        print(
            "  WARNING: Label has no CRS."
        )

        print(
            "  Checking whether label is "
            "pixel-aligned with S2..."
        )

        # ----------------------------------------------------
        # Check dimensions
        # ----------------------------------------------------

        same_size = (
            label_src.width == reference_src.width
            and
            label_src.height == reference_src.height
        )

        if not same_size:

            raise ValueError(
                "\nLabel has no CRS and dimensions "
                "do not match Sentinel-2.\n"
                f"Label: "
                f"{label_src.width} x "
                f"{label_src.height}\n"
                f"S2: "
                f"{reference_src.width} x "
                f"{reference_src.height}\n"
                "Cannot safely align the label."
            )

        # ----------------------------------------------------
        # IMPORTANT
        #
        # Since dimensions match and the OSCD label is a
        # pixel-based change mask, use it directly on the
        # Sentinel grid.
        # ----------------------------------------------------

        binary_label = binary_original.copy()

        print(
            "  Label dimensions match S2."
        )

        print(
            "  Using label directly on S2 pixel grid."
        )

        return binary_label

    # ========================================================
    # CASE 2:
    # LABEL HAS CRS
    # ========================================================

    print()
    print(
        "  Label has CRS:"
        f" {label_src.crs}"
    )

    # --------------------------------------------------------
    # Destination array
    # --------------------------------------------------------

    binary_label = np.zeros(
        (
            reference_src.height,
            reference_src.width
        ),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # Reproject to exact S2 grid
    # --------------------------------------------------------

    reproject(
        source=binary_original,

        destination=binary_label,

        src_transform=label_src.transform,
        src_crs=label_src.crs,

        dst_transform=reference_src.transform,
        dst_crs=reference_src.crs,

        resampling=Resampling.nearest
    )

    print(
        "  Label successfully aligned "
        "to S2 grid."
    )

    return binary_label


# ============================================================
# CREATE STACKED IMAGE
# ============================================================

def create_stack(
    src_t1,
    src_t2
):
    """
    Stack T1 and T2.

    S1:
        T1 VV
        T1 VH
        T2 VV
        T2 VH

        -> 4 bands

    S2:
        T1 B2
        T1 B3
        T1 B4
        T1 B8
        T2 B2
        T2 B3
        T2 B4
        T2 B8

        -> 8 bands
    """

    data_t1 = src_t1.read()
    data_t2 = src_t2.read()

    # --------------------------------------------------------
    # Check dimensions
    # --------------------------------------------------------

    if data_t1.shape[1:] != data_t2.shape[1:]:

        raise ValueError(
            "T1 and T2 dimensions do not match:\n"
            f"T1: {data_t1.shape}\n"
            f"T2: {data_t2.shape}"
        )

    # --------------------------------------------------------
    # Stack along band dimension
    # --------------------------------------------------------

    stacked = np.concatenate(
        [
            data_t1,
            data_t2
        ],
        axis=0
    )

    return stacked


# ============================================================
# SAVE PATCH
# ============================================================

def save_patch(
    output_path,
    data,
    reference_src,
    window,
    count,
    dtype=None
):

    # --------------------------------------------------------
    # Calculate transform for this patch
    # --------------------------------------------------------

    transform = rasterio.windows.transform(
        window,
        reference_src.transform
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    profile = reference_src.profile.copy()

    if dtype is None:

        dtype = data.dtype

    profile.update(

        driver="GTiff",

        height=data.shape[1],

        width=data.shape[2],

        count=count,

        dtype=dtype,

        transform=transform,

        compress="LZW",

        tiled=False,

        nodata=None
    )

    # --------------------------------------------------------
    # Write
    # --------------------------------------------------------

    with rasterio.open(
        output_path,
        "w",
        **profile
    ) as dst:

        dst.write(data.astype(dtype))


# ============================================================
# PROCESS ONE CITY
# ============================================================

def process_city(
    split,
    city,
    patch_counter
):

    print()
    print("-" * 80)
    print(f"{split.upper()} | {city.upper()}")
    print("-" * 80)

    files = get_city_files(
        split,
        city
    )

    print()
    print("Checking files:")

    if not check_files(files):

        print()
        print(
            f"Skipping {city}: "
            "one or more files are missing."
        )

        return patch_counter, 0, 0

    # ========================================================
    # OPEN DATASETS
    # ========================================================

    with rasterio.open(files["s2_t1"]) as s2_t1, \
         rasterio.open(files["s2_t2"]) as s2_t2, \
         rasterio.open(files["s1_t1"]) as s1_t1, \
         rasterio.open(files["s1_t2"]) as s1_t2, \
         rasterio.open(files["label"]) as label_src:

        # ----------------------------------------------------
        # Print information
        # ----------------------------------------------------

        print_raster_info(
            "SENTINEL-2 T1",
            s2_t1
        )

        print_raster_info(
            "SENTINEL-2 T2",
            s2_t2
        )

        print_raster_info(
            "SENTINEL-1 T1",
            s1_t1
        )

        print_raster_info(
            "SENTINEL-1 T2",
            s1_t2
        )

        print_raster_info(
            "LABEL",
            label_src
        )

        # ====================================================
        # CHECK IMAGE ALIGNMENT
        # ====================================================

        print()
        print("Checking image alignment...")

        if not check_image_alignment(
            s2_t1,
            s2_t2,
            s1_t1,
            s1_t2
        ):

            print(
                f"\nERROR: Images are not aligned "
                f"for {city}."
            )

            return patch_counter, 0, 1

        # ====================================================
        # CHECK LABEL
        # ====================================================

        print()
        print("Checking label CRS...")

        print(
            f"  Label CRS: {label_src.crs}"
        )

        print(
            f"  S2 CRS   : {s2_t1.crs}"
        )

        # ====================================================
        # CREATE BINARY LABEL
        # ====================================================

        print()
        print("Creating binary label in memory...")

        binary_label = create_binary_label(
            label_src,
            s2_t1
        )

        # ====================================================
        # CHECK BINARY LABEL
        # ====================================================

        label_values, label_counts = np.unique(
            binary_label,
            return_counts=True
        )

        print()
        print("Binary label values:")

        for value, count in zip(
            label_values,
            label_counts
        ):

            percentage = (
                count /
                binary_label.size *
                100
            )

            if value == 0:

                meaning = "NO CHANGE"

            elif value == 1:

                meaning = "CHANGE"

            else:

                meaning = "UNKNOWN"

            print(
                f"  {value} = {meaning:<10} "
                f"{count:,} pixels "
                f"({percentage:.2f}%)"
            )

        # ====================================================
        # CREATE STACKS
        # ====================================================

        print()
        print("Creating image stacks...")

        # ----------------------------------------------------
        # S1 stack
        # ----------------------------------------------------

        s1_stack = create_stack(
            s1_t1,
            s1_t2
        )

        # ----------------------------------------------------
        # S2 stack
        # ----------------------------------------------------

        s2_stack = create_stack(
            s2_t1,
            s2_t2
        )

        print(
            f"  S1 stack shape: "
            f"{s1_stack.shape}"
        )

        print(
            f"  S2 stack shape: "
            f"{s2_stack.shape}"
        )

        # ====================================================
        # VERIFY STACK SIZE
        # ====================================================

        if s1_stack.shape[1:] != (
            s2_t1.height,
            s2_t1.width
        ):

            raise ValueError(
                "S1 stack dimensions do not match S2."
            )

        if s2_stack.shape[1:] != (
            s2_t1.height,
            s2_t1.width
        ):

            raise ValueError(
                "S2 stack dimensions are incorrect."
            )

        # ====================================================
        # PATCH LOOP
        # ====================================================

        height = s2_t1.height
        width = s2_t1.width

        candidate_patches = 0
        empty_patches = 0
        saved_patches = 0

        print()
        print(
            f"Creating {PATCH_SIZE}x{PATCH_SIZE} "
            f"patches..."
        )

        # ----------------------------------------------------
        # Only complete patches
        #
        # This avoids writing partially filled patches.
        # ----------------------------------------------------

        for row in range(
            0,
            height - PATCH_SIZE + 1,
            STRIDE
        ):

            for col in range(
                0,
                width - PATCH_SIZE + 1,
                STRIDE
            ):

                candidate_patches += 1

                window = Window(
                    col,
                    row,
                    PATCH_SIZE,
                    PATCH_SIZE
                )

                # ============================================
                # Extract label patch
                # ============================================

                label_patch = binary_label[
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]

                # Safety check
                if label_patch.shape != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):

                    continue

                # ============================================
                # CHECK LABEL
                # ============================================

                unique_values = np.unique(
                    label_patch
                )

                # A completely empty/invalid label means
                # there are no valid 0/1 pixels.
                #
                # We still allow a patch containing only 0.
                # We only reject patches that somehow contain
                # no valid binary values.
                # ============================================

                valid_pixels = np.isin(
                    label_patch,
                    [0, 1]
                )

                if not np.any(valid_pixels):

                    empty_patches += 1

                    continue

                # ============================================
                # Extract image patches
                # ============================================

                s1_patch = s1_stack[
                    :,
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]

                s2_patch = s2_stack[
                    :,
                    row:row + PATCH_SIZE,
                    col:col + PATCH_SIZE
                ]

                # ============================================
                # Safety check
                # ============================================

                if s1_patch.shape[1:] != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):

                    continue

                if s2_patch.shape[1:] != (
                    PATCH_SIZE,
                    PATCH_SIZE
                ):

                    continue

                # ============================================
                # PATCH NAME
                # ============================================

                patch_id = patch_counter

                s1_name = (
                    f"{city}_{patch_id}_s1.tif"
                )

                s2_name = (
                    f"{city}_{patch_id}_s2.tif"
                )

                label_name = (
                    f"{city}_{patch_id}_cm.tif"
                )

                # ============================================
                # OUTPUT PATHS
                # ============================================

                s1_output = os.path.join(
                    OUTPUT_IMAGES,
                    split,
                    s1_name
                )

                s2_output = os.path.join(
                    OUTPUT_IMAGES,
                    split,
                    s2_name
                )

                label_output = os.path.join(
                    OUTPUT_LABELS,
                    split,
                    label_name
                )

                # ============================================
                # SAVE S1
                # ============================================

                save_patch(
                    s1_output,
                    s1_patch,
                    s1_t1,
                    window,
                    count=s1_patch.shape[0],
                    dtype=s1_patch.dtype
                )

                # ============================================
                # SAVE S2
                # ============================================

                save_patch(
                    s2_output,
                    s2_patch,
                    s2_t1,
                    window,
                    count=s2_patch.shape[0],
                    dtype=s2_patch.dtype
                )

                # ============================================
                # SAVE BINARY LABEL
                # ============================================

                label_profile = s2_t1.profile.copy()

                label_transform = (
                    rasterio.windows.transform(
                        window,
                        s2_t1.transform
                    )
                )

                label_profile.update(

                    driver="GTiff",

                    height=PATCH_SIZE,

                    width=PATCH_SIZE,

                    count=1,

                    dtype="uint8",

                    transform=label_transform,

                    compress="LZW",

                    tiled=False,

                    nodata=None
                )

                with rasterio.open(
                    label_output,
                    "w",
                    **label_profile
                ) as dst:

                    dst.write(
                        label_patch.astype(
                            np.uint8
                        ),
                        1
                    )

                # ============================================
                # STATISTICS FOR FIRST FEW PATCHES
                # ============================================

                if saved_patches < 5:

                    patch_values = np.unique(
                        label_patch
                    )

                    change_pixels = np.sum(
                        label_patch == 1
                    )

                    no_change_pixels = np.sum(
                        label_patch == 0
                    )

                    print()
                    print(
                        f"  Patch {patch_id}:"
                    )

                    print(
                        f"    S1 shape: "
                        f"{s1_patch.shape}"
                    )

                    print(
                        f"    S2 shape: "
                        f"{s2_patch.shape}"
                    )

                    print(
                        f"    Label shape: "
                        f"{label_patch.shape}"
                    )

                    print(
                        f"    Label values: "
                        f"{patch_values}"
                    )

                    print(
                        f"    No-change: "
                        f"{no_change_pixels:,}"
                    )

                    print(
                        f"    Change: "
                        f"{change_pixels:,}"
                    )

                saved_patches += 1
                patch_counter += 1

        # ====================================================
        # CITY SUMMARY
        # ====================================================

        print()
        print(
            f"City {city.upper()} finished:"
        )

        print(
            f"  Candidate patches : "
            f"{candidate_patches}"
        )

        print(
            f"  Empty patches     : "
            f"{empty_patches}"
        )

        print(
            f"  Saved patches     : "
            f"{saved_patches}"
        )

    return patch_counter, saved_patches, 0


# ============================================================
# PROCESS ALL SPLITS
# ============================================================

total_saved = 0
total_failed = 0

global_patch_counter = 0


for split in [
    "train",
    "val",
    "test"
]:

    print()
    print()
    print("#" * 80)
    print(
        f"PROCESSING {split.upper()}"
    )
    print("#" * 80)

    cities = find_cities(split)

    print()
    print(
        f"Cities found: "
        f"{len(cities)}"
    )

    for city in cities:

        print(
            f"   {city}"
        )

    split_saved = 0
    split_failed = 0

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Patch numbering starts from 0 for EACH CITY.
    #
    # Therefore:
    #
    # brasilia_0_s1.tif
    # brasilia_1_s1.tif
    #
    # rather than continuing globally between cities.
    # --------------------------------------------------------

    for city in cities:

        # Find next available patch number for this city
        city_patch_counter = 0

        (
            city_counter,
            saved,
            failed
        ) = process_city(
            split,
            city,
            city_patch_counter
        )

        split_saved += saved
        split_failed += failed

    total_saved += split_saved
    total_failed += split_failed

    print()
    print("=" * 80)
    print(
        f"{split.upper()} SUMMARY"
    )
    print("=" * 80)

    print(
        f"Cities found : {len(cities)}"
    )

    print(
        f"Saved patches: {split_saved}"
    )

    print(
        f"Failed cities: {split_failed}"
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("PATCH CREATION FINISHED")
print("=" * 80)

print()
print(
    f"Total saved patches: "
    f"{total_saved}"
)

print(
    f"Total failed cities: "
    f"{total_failed}"
)

print()
print("Output folders:")
print()

print(
    os.path.join(
        PATCHES_DIR,
        "images",
        "train"
    )
)

print(
    os.path.join(
        PATCHES_DIR,
        "images",
        "val"
    )
)

print(
    os.path.join(
        PATCHES_DIR,
        "images",
        "test"
    )
)

print()

print(
    os.path.join(
        PATCHES_DIR,
        "labels",
        "train"
    )
)

print(
    os.path.join(
        PATCHES_DIR,
        "labels",
        "val"
    )
)

print(
    os.path.join(
        PATCHES_DIR,
        "labels",
        "test"
    )

)

print()
print("=" * 80)