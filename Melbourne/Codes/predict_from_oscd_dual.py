# ============================================================
# MELBOURNE URBAN CHANGE DETECTION
# PREDICTION USING TRAINED OSCD MODEL
#
# Input:
#   Melbourne 2023 + 2025 Sentinel-1
#   Melbourne 2023 + 2025 Sentinel-2
#
# S1:
#   2023 VV
#   2023 VH
#   2025 VV
#   2025 VH
#   -> 4 channels
#
# S2:
#   2023 B2 B3 B4 B8
#   2025 B2 B3 B4 B8
#   -> 8 channels
#
# Output:
#   0 = NO CHANGE
#   1 = CHANGE
#
# No probability maps are saved.
# ============================================================

import os
import sys
import glob
import re

import numpy as np
import rasterio
import torch
from tqdm import tqdm


# ============================================================
# PROJECT ROOT
# ============================================================

root = os.getcwd()
PROJECT_ROOT = os.path.dirname(root)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT MODEL
# ============================================================

from models import CrossModalTemporalGateUNet


# ============================================================
# PATHS
# ============================================================

MELBOURNE_ROOT = (
    r"C:\melbourne-urban-change-detection"
    r"\Melbourne\MelbourneData"
)

S1_2023_DIR = os.path.join(
    MELBOURNE_ROOT,
    "images",
    "2023",
    "sentinel1"
)

S1_2025_DIR = os.path.join(
    MELBOURNE_ROOT,
    "images",
    "2025",
    "sentinel1"
)

S2_2023_DIR = os.path.join(
    MELBOURNE_ROOT,
    "images",
    "2023",
    "sentinel2"
)

S2_2025_DIR = os.path.join(
    MELBOURNE_ROOT,
    "images",
    "2025",
    "sentinel2"
)


# ============================================================
# OUTPUT
# ============================================================

OUTPUT_DIR = os.path.join(
    MELBOURNE_ROOT,
    "change_maps"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TRAINED MODEL
# ============================================================

CHECKPOINT_PATH = os.path.join(
    MELBOURNE_ROOT,
    "checkpoints",
    "best_dual_stream_oscd.pth")


# ============================================================
# SETTINGS
# ============================================================

THRESHOLD = 0.5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("MELBOURNE URBAN CHANGE DETECTION")
print("=" * 80)

print(
    f"Project root : {PROJECT_ROOT}"
)

print(
    f"Melbourne    : {MELBOURNE_ROOT}"
)

print(
    f"Checkpoint   : {CHECKPOINT_PATH}"
)

print(
    f"Output       : {OUTPUT_DIR}"
)

print(
    f"Device       : {DEVICE}"
)

print(
    f"Threshold    : {THRESHOLD}"
)

if torch.cuda.is_available():

    print(
        f"GPU          : "
        f"{torch.cuda.get_device_name(0)}"
    )

print("=" * 80)


# ============================================================
# FIND MELBOURNE PATCHES
# ============================================================

def find_sentinel2_2023():

    pattern = os.path.join(
        S2_2023_DIR,
        "melbourne_*_s2_2023.tif"
    )

    return sorted(
        glob.glob(pattern)
    )


# ============================================================
# EXTRACT PATCH NUMBER
# ============================================================

def get_patch_id(path):

    filename = os.path.basename(path)

    match = re.search(
        r"melbourne_(\d+)_",
        filename
    )

    if match is None:

        raise ValueError(
            f"Could not extract patch ID from:\n{path}"
        )

    return match.group(1)


# ============================================================
# BUILD FILE MATCHES
# ============================================================

s2_2023_files = find_sentinel2_2023()

print("\nFinding Melbourne patches...")

print(
    f"S2 2023 patches found: "
    f"{len(s2_2023_files)}"
)

if len(s2_2023_files) == 0:

    raise RuntimeError(
        "No Melbourne Sentinel-2 2023 files found."
    )


patches = []


for s2_2023_path in s2_2023_files:

    patch_id = get_patch_id(
        s2_2023_path
    )

    s1_2023_path = os.path.join(
        S1_2023_DIR,
        f"melbourne_{patch_id}_s1_2023.tif"
    )

    s2_2025_path = os.path.join(
        S2_2025_DIR,
        f"melbourne_{patch_id}_s2_2025.tif"
    )

    s1_2025_path = os.path.join(
        S1_2025_DIR,
        f"melbourne_{patch_id}_s1_2025.tif"
    )

    paths = [
        s1_2023_path,
        s1_2025_path,
        s2_2023_path,
        s2_2025_path
    ]

    missing = [
        p for p in paths
        if not os.path.exists(p)
    ]

    if missing:

        print(
            f"\nWARNING: Missing files for patch "
            f"{patch_id}"
        )

        for p in missing:
            print(
                f"  Missing: {p}"
            )

        continue

    patches.append(
        {
            "id": patch_id,
            "s1_2023": s1_2023_path,
            "s1_2025": s1_2025_path,
            "s2_2023": s2_2023_path,
            "s2_2025": s2_2025_path,
        }
    )


print(
    f"\nComplete patch pairs: "
    f"{len(patches)}"
)

if len(patches) == 0:

    raise RuntimeError(
        "No complete Melbourne patch pairs found."
    )


# ============================================================
# READ RASTER
# ============================================================

def read_raster(path):

    with rasterio.open(path) as src:

        data = src.read().astype(
            np.float32
        )

        profile = src.profile.copy()

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height

        nodata = src.nodata

    return (
        data,
        profile,
        transform,
        crs,
        width,
        height,
        nodata
    )


# ============================================================
# CHECK INPUT RASTER
# ============================================================

def print_raster_info(
    name,
    path,
    data
):

    print(
        f"\n{name}"
    )

    print(
        f"  File   : {path}"
    )

    print(
        f"  Shape  : {data.shape}"
    )

    print(
        f"  Min    : "
        f"{np.nanmin(data):.4f}"
    )

    print(
        f"  Max    : "
        f"{np.nanmax(data):.4f}"
    )

    print(
        f"  Mean   : "
        f"{np.nanmean(data):.4f}"
    )


# ============================================================
# NORMALIZATION
# ============================================================
#
# IMPORTANT:
#
# This must match the preprocessing used during training.
#
# If your OSCD training dataset uses another normalization,
# replace this function with the exact same preprocessing.
#
# ============================================================

def normalize_channels(data):

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    normalized = np.zeros_like(
        data,
        dtype=np.float32
    )

    for b in range(data.shape[0]):

        band = data[b]

        mean = band.mean()

        std = band.std()

        if std < 1e-6:

            normalized[b] = (
                band - mean
            )

        else:

            normalized[b] = (
                (band - mean)
                / (std + 1e-6)
            )

    return normalized


# ============================================================
# LOAD MODEL
# ============================================================

print("\n")
print("=" * 80)
print("LOADING TRAINED MODEL")
print("=" * 80)

if not os.path.exists(
    CHECKPOINT_PATH
):

    raise FileNotFoundError(
        f"\nModel checkpoint not found:\n"
        f"{CHECKPOINT_PATH}"
    )


model = CrossModalTemporalGateUNet(
    s1_in_channels=4,
    s2_in_channels=8
).to(DEVICE)


checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Loaded model_state_dict."
    )

    if "epoch" in checkpoint:

        print(
            f"Training epoch: "
            f"{checkpoint['epoch']}"
        )

    if "best_val_iou" in checkpoint:

        print(
            f"Best validation score: "
            f"{checkpoint['best_val_iou']}"
        )

else:

    model.load_state_dict(
        checkpoint
    )

    print(
        "Loaded raw model state_dict."
    )


model.eval()

print(
    "Model loaded successfully."
)


# ============================================================
# PREDICT ONE PATCH
# ============================================================

def predict_patch(patch):

    patch_id = patch["id"]

    print("\n")
    print("-" * 80)
    print(
        f"PROCESSING MELBOURNE PATCH "
        f"{patch_id}"
    )
    print("-" * 80)


    # --------------------------------------------------------
    # READ S1
    # --------------------------------------------------------

    s1_2023, profile, transform, crs, width, height, nodata = (
        read_raster(
            patch["s1_2023"]
        )
    )

    s1_2025, _, _, _, _, _, _ = (
        read_raster(
            patch["s1_2025"]
        )
    )


    # --------------------------------------------------------
    # READ S2
    # --------------------------------------------------------

    s2_2023, _, _, _, _, _, _ = (
        read_raster(
            patch["s2_2023"]
        )
    )

    s2_2025, _, _, _, _, _, _ = (
        read_raster(
            patch["s2_2025"]
        )
    )


    # --------------------------------------------------------
    # PRINT INFORMATION
    # --------------------------------------------------------

    print_raster_info(
        "S1 2023",
        patch["s1_2023"],
        s1_2023
    )

    print_raster_info(
        "S1 2025",
        patch["s1_2025"],
        s1_2025
    )

    print_raster_info(
        "S2 2023",
        patch["s2_2023"],
        s2_2023
    )

    print_raster_info(
        "S2 2025",
        patch["s2_2025"],
        s2_2025
    )


    # ========================================================
    # CHECK BAND COUNTS
    # ========================================================

    if s1_2023.shape[0] != 2:

        raise ValueError(
            f"S1 2023 should have 2 bands, "
            f"got {s1_2023.shape[0]}"
        )

    if s1_2025.shape[0] != 2:

        raise ValueError(
            f"S1 2025 should have 2 bands, "
            f"got {s1_2025.shape[0]}"
        )

    if s2_2023.shape[0] != 4:

        raise ValueError(
            f"S2 2023 should have 4 bands, "
            f"got {s2_2023.shape[0]}"
        )

    if s2_2025.shape[0] != 4:

        raise ValueError(
            f"S2 2025 should have 4 bands, "
            f"got {s2_2025.shape[0]}"
        )


    # ========================================================
    # CHECK SPATIAL DIMENSIONS
    # ========================================================

    shapes = [
        s1_2023.shape[1:],
        s1_2025.shape[1:],
        s2_2023.shape[1:],
        s2_2025.shape[1:]
    ]

    if not all(
        shape == shapes[0]
        for shape in shapes
    ):

        raise ValueError(
            f"Spatial dimensions do not match:\n"
            f"{shapes}"
        )


    # ========================================================
    # CHECK GEOREFERENCING
    # ========================================================

    with rasterio.open(
        patch["s1_2025"]
    ) as src:

        s1_2025_transform = src.transform
        s1_2025_crs = src.crs

    with rasterio.open(
        patch["s2_2023"]
    ) as src:

        s2_2023_transform = src.transform
        s2_2023_crs = src.crs

    with rasterio.open(
        patch["s2_2025"]
    ) as src:

        s2_2025_transform = src.transform
        s2_2025_crs = src.crs


    if not np.allclose(
        transform,
        s1_2025_transform
    ):

        raise ValueError(
            f"S1 2023 and S1 2025 "
            f"are not aligned for patch "
            f"{patch_id}"
        )


    if not np.allclose(
        transform,
        s2_2023_transform
    ):

        raise ValueError(
            f"S1 and S2 2023 are not "
            f"aligned for patch {patch_id}"
        )


    if not np.allclose(
        transform,
        s2_2025_transform
    ):

        raise ValueError(
            f"S1 2023 and S2 2025 are "
            f"not aligned for patch "
            f"{patch_id}"
        )


    if crs != s1_2025_crs:

        raise ValueError(
            f"CRS mismatch for patch "
            f"{patch_id}"
        )


    if crs != s2_2023_crs:

        raise ValueError(
            f"CRS mismatch between S1 "
            f"and S2 for patch {patch_id}"
        )


    if crs != s2_2025_crs:

        raise ValueError(
            f"CRS mismatch for patch "
            f"{patch_id}"
        )


    # ========================================================
    # STACK TEMPORAL IMAGES
    # ========================================================
    #
    # S1:
    #
    # [S1_2023_VV,
    #  S1_2023_VH,
    #  S1_2025_VV,
    #  S1_2025_VH]
    #
    # S2:
    #
    # [S2_2023_B2,
    #  S2_2023_B3,
    #  S2_2023_B4,
    #  S2_2023_B8,
    #  S2_2025_B2,
    #  S2_2025_B3,
    #  S2_2025_B4,
    #  S2_2025_B8]
    #
    # ========================================================

    s1 = np.concatenate(
        [
            s1_2023,
            s1_2025
        ],
        axis=0
    )

    s2 = np.concatenate(
        [
            s2_2023,
            s2_2025
        ],
        axis=0
    )


    print(
        "\nStacked input shapes:"
    )

    print(
        f"  S1: {s1.shape}"
    )

    print(
        f"  S2: {s2.shape}"
    )


    # ========================================================
    # NORMALIZE
    # ========================================================

    s1 = normalize_channels(
        s1
    )

    s2 = normalize_channels(
        s2
    )


    # ========================================================
    # CONVERT TO TORCH
    # ========================================================

    s1_tensor = torch.from_numpy(
        s1
    ).float().unsqueeze(0).to(
        DEVICE
    )

    s2_tensor = torch.from_numpy(
        s2
    ).float().unsqueeze(0).to(
        DEVICE
    )


    print(
        "\nModel input:"
    )

    print(
        f"  S1 tensor: "
        f"{tuple(s1_tensor.shape)}"
    )

    print(
        f"  S2 tensor: "
        f"{tuple(s2_tensor.shape)}"
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    with torch.no_grad():

        logits = model(
            s1_tensor,
            s2_tensor
        )

        probabilities = torch.sigmoid(
            logits
        )

        prediction = (
            probabilities >= THRESHOLD
        ).to(torch.uint8)


    # ========================================================
    # REMOVE DIMENSIONS
    # ========================================================

    prediction = (
        prediction
        .squeeze()
        .cpu()
        .numpy()
    )


    probability_stats = (
        probabilities
        .squeeze()
        .cpu()
        .numpy()
    )


    # ========================================================
    # PRINT PREDICTION STATISTICS
    # ========================================================

    unique, counts = np.unique(
        prediction,
        return_counts=True
    )

    print(
        "\nPrediction statistics:"
    )

    print(
        f"  Probability min : "
        f"{probability_stats.min():.6f}"
    )

    print(
        f"  Probability max : "
        f"{probability_stats.max():.6f}"
    )

    print(
        f"  Probability mean: "
        f"{probability_stats.mean():.6f}"
    )

    print(
        f"  Unique classes  : "
        f"{unique}"
    )

    for value, count in zip(
        unique,
        counts
    ):

        percentage = (
            count /
            prediction.size *
            100
        )

        if value == 0:

            name = "NO CHANGE"

        else:

            name = "CHANGE"

        print(
            f"  {value} ({name}): "
            f"{count:,} pixels "
            f"({percentage:.2f}%)"
        )


    # ========================================================
    # SAVE BINARY CHANGE MAP
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        f"melbourne_{patch_id}_change.tif"
    )


    output_profile = profile.copy()

    output_profile.update(
        {
            "driver": "GTiff",
            "height": prediction.shape[0],
            "width": prediction.shape[1],
            "count": 1,
            "dtype": "uint8",
            "compress": "LZW",
            "nodata": None
        }
    )


    with rasterio.open(
        output_path,
        "w",
        **output_profile
    ) as dst:

        dst.write(
            prediction,
            1
        )


    print(
        f"\nSaved binary change map:"
    )

    print(
        f"  {output_path}"
    )


    return prediction


# ============================================================
# PROCESS ALL PATCHES
# ============================================================

all_predictions = []


for patch in tqdm(
    patches,
    desc="Melbourne patches"
):

    prediction = predict_patch(
        patch
    )

    all_predictions.append(
        prediction
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("MELBOURNE CHANGE DETECTION COMPLETE")
print("=" * 80)

print(
    f"Patches processed: "
    f"{len(all_predictions)}"
)

print(
    f"Output directory:"
)

print(
    OUTPUT_DIR
)

print("\nOutput files:")

for patch in patches:

    print(
        f"  melbourne_{patch['id']}_change.tif"
    )

print("\nEach output contains:")

print("  0 = NO CHANGE")
print("  1 = CHANGE")

print("=" * 80)