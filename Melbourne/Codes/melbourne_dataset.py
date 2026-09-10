
"""
Melbourne Sentinel-1 / Sentinel-2 Change Detection Dataset

Prepares data for:

1. Early-Fusion U-Net
   Input:
       [S1_T1, S2_T1, S1_T2, S2_T2]

   Shape:
       [B, 12, H, W]

   Channels:
       S1_T1 = 2
       S2_T1 = 4
       S1_T2 = 2
       S2_T2 = 4

2. Dual-Stream U-Net
   Sentinel-1:
       [S1_T1, S1_T2]
       Shape = [B, 4, H, W]

   Sentinel-2:
       [S2_T1, S2_T2]
       Shape = [B, 8, H, W]

Labels:
    [B, 1, H, W]

CSV structure:

patch_id
split
sentinel1_2023
sentinel2_2023
sentinel1_2025
sentinel2_2025
label
"""

import os
import numpy as np
import pandas as pd
import rasterio

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MELBOURNE_DATA = os.path.join(
    ROOT,
    "MelbourneData"
)

CSV_ROOT = os.path.join(
    MELBOURNE_DATA,
    "csv"
)


# ============================================================
# NORMALIZATION CONFIGURATION
# ============================================================

# Set this to True if your TIFF values are not already normalized.
NORMALIZE = True


# ------------------------------------------------------------
# Recommended percentile normalization
#
# Each image is normalized independently.
#
# This is robust to outliers in Sentinel-1 and Sentinel-2.
# ------------------------------------------------------------

LOW_PERCENTILE = 2
HIGH_PERCENTILE = 98


# ============================================================
# RASTER READING
# ============================================================

def read_raster(path):
    """
    Read a multi-band GeoTIFF.

    Returns
    -------
    image : np.ndarray
        Shape:
            [C, H, W]
    """

    with rasterio.open(path) as src:

        image = src.read()

    image = image.astype(
        np.float32
    )

    return image


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_image(image):
    """
    Percentile normalization to [0, 1].

    Parameters
    ----------
    image : np.ndarray
        Shape [C, H, W]

    Returns
    -------
    normalized : np.ndarray
        Shape [C, H, W]
    """

    image = image.copy()

    for channel in range(
        image.shape[0]
    ):

        band = image[channel]

        valid = np.isfinite(band)

        if not np.any(valid):
            image[channel] = 0.0
            continue

        values = band[valid]

        low = np.percentile(
            values,
            LOW_PERCENTILE
        )

        high = np.percentile(
            values,
            HIGH_PERCENTILE
        )

        if high > low:

            band = (
                band - low
            ) / (
                high - low
            )

        else:

            band = np.zeros_like(
                band,
                dtype=np.float32
            )

        band = np.clip(
            band,
            0.0,
            1.0
        )

        band[
            ~np.isfinite(band)
        ] = 0.0

        image[channel] = band

    return image


# ============================================================
# LABEL READING
# ============================================================

def read_label(path):
    """
    Read binary change label.

    Expected:
        0 = unchanged
        1 = change

    Also handles labels stored as:
        0 / 255
    """

    with rasterio.open(path) as src:

        label = src.read(1)

    label = label.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Convert 255 -> 1
    # --------------------------------------------------------

    if label.max() > 1:

        label = (
            label > 0
        ).astype(
            np.float32
        )

    # --------------------------------------------------------
    # Add channel dimension
    # --------------------------------------------------------

    label = np.expand_dims(
        label,
        axis=0
    )

    return label


# ============================================================
# CHECK IMAGE SHAPES
# ============================================================

def check_shapes(
    s1_2023,
    s2_2023,
    s1_2025,
    s2_2025,
    label,
    patch_id
):
    """
    Verify spatial dimensions and channel counts.
    """

    # --------------------------------------------------------
    # Channel checks
    # --------------------------------------------------------

    if s1_2023.shape[0] != 2:

        raise ValueError(
            f"{patch_id}: "
            f"S1 2023 must have 2 channels, "
            f"got {s1_2023.shape[0]}"
        )

    if s1_2025.shape[0] != 2:

        raise ValueError(
            f"{patch_id}: "
            f"S1 2025 must have 2 channels, "
            f"got {s1_2025.shape[0]}"
        )

    if s2_2023.shape[0] != 4:

        raise ValueError(
            f"{patch_id}: "
            f"S2 2023 must have 4 channels, "
            f"got {s2_2023.shape[0]}"
        )

    if s2_2025.shape[0] != 4:

        raise ValueError(
            f"{patch_id}: "
            f"S2 2025 must have 4 channels, "
            f"got {s2_2025.shape[0]}"
        )

    # --------------------------------------------------------
    # Spatial shape
    # --------------------------------------------------------

    spatial_shapes = [
        s1_2023.shape[-2:],
        s2_2023.shape[-2:],
        s1_2025.shape[-2:],
        s2_2025.shape[-2:],
        label.shape[-2:]
    ]

    if len(
        set(spatial_shapes)
    ) != 1:

        raise ValueError(
            f"{patch_id}: "
            f"Spatial dimensions do not match:\n"
            f"{spatial_shapes}"
        )


# ============================================================
# BASE DATASET
# ============================================================

class MelbourneChangeDataset(Dataset):

    def __init__(
        self,
        csv_path,
        normalize=True
    ):

        self.csv_path = csv_path

        self.normalize = normalize

        # ----------------------------------------------------
        # Read CSV
        # ----------------------------------------------------

        self.df = pd.read_csv(
            csv_path
        )

        print()
        print("=" * 80)
        print("DATASET")
        print("=" * 80)

        print()
        print("CSV:")
        print(csv_path)

        print()
        print(
            f"Number of samples: "
            f"{len(self.df)}"
        )


    def __len__(self):

        return len(
            self.df
        )


    def load_sample(self, index):

        row = self.df.iloc[index]

        patch_id = row[
            "patch_id"
        ]

        # ----------------------------------------------------
        # Read Sentinel-1
        # ----------------------------------------------------

        s1_2023 = read_raster(
            row["sentinel1_2023"]
        )

        s1_2025 = read_raster(
            row["sentinel1_2025"]
        )

        # ----------------------------------------------------
        # Read Sentinel-2
        # ----------------------------------------------------

        s2_2023 = read_raster(
            row["sentinel2_2023"]
        )

        s2_2025 = read_raster(
            row["sentinel2_2025"]
        )

        # ----------------------------------------------------
        # Read label
        # ----------------------------------------------------

        label = read_label(
            row["label"]
        )

        # ----------------------------------------------------
        # Check shapes
        # ----------------------------------------------------

        check_shapes(
            s1_2023,
            s2_2023,
            s1_2025,
            s2_2025,
            label,
            patch_id
        )

        # ----------------------------------------------------
        # Normalize images
        # ----------------------------------------------------

        if self.normalize:

            s1_2023 = normalize_image(
                s1_2023
            )

            s1_2025 = normalize_image(
                s1_2025
            )

            s2_2023 = normalize_image(
                s2_2023
            )

            s2_2025 = normalize_image(
                s2_2025
            )

        return (
            s1_2023,
            s2_2023,
            s1_2025,
            s2_2025,
            label,
            patch_id
        )


# ============================================================
# EARLY-FUSION DATASET
# ============================================================

class EarlyFusionDataset(
    MelbourneChangeDataset
):

    """
    Dataset for:

        ContrastPyramidEarlyFusionUNet

    Final input:

        [S1_T1 | S2_T1 | S1_T2 | S2_T2]

    Channels:

        2 + 4 + 2 + 4 = 12

    Shape:

        [12, H, W]
    """

    def __getitem__(self, index):

        (
            s1_2023,
            s2_2023,
            s1_2025,
            s2_2025,
            label,
            patch_id
        ) = self.load_sample(index)


        # ----------------------------------------------------
        # Early fusion
        # ----------------------------------------------------
        #
        # IMPORTANT:
        #
        # TemporalContrastStem expects:
        #
        # first 6 channels = T1
        # last 6 channels  = T2
        #
        # Therefore:
        #
        # T1 = S1_2023 + S2_2023
        # T2 = S1_2025 + S2_2025
        #
        # ----------------------------------------------------

        t1 = np.concatenate(
            [
                s1_2023,
                s2_2023
            ],
            axis=0
        )

        t2 = np.concatenate(
            [
                s1_2025,
                s2_2025
            ],
            axis=0
        )

        image = np.concatenate(
            [
                t1,
                t2
            ],
            axis=0
        )


        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        image = torch.from_numpy(
            image
        ).float()

        label = torch.from_numpy(
            label
        ).float()


        return {
            "image": image,
            "label": label,
            "patch_id": patch_id
        }


# ============================================================
# DUAL-STREAM DATASET
# ============================================================

class DualStreamDataset(
    MelbourneChangeDataset
):

    """
    Dataset for:

        CrossModalTemporalGateUNet

    Sentinel-1 input:

        [S1_T1 | S1_T2]

        2 + 2 = 4 channels

    Sentinel-2 input:

        [S2_T1 | S2_T2]

        4 + 4 = 8 channels
    """

    def __getitem__(self, index):

        (
            s1_2023,
            s2_2023,
            s1_2025,
            s2_2025,
            label,
            patch_id
        ) = self.load_sample(index)


        # ----------------------------------------------------
        # Sentinel-1 temporal stack
        # ----------------------------------------------------

        sentinel1 = np.concatenate(
            [
                s1_2023,
                s1_2025
            ],
            axis=0
        )


        # ----------------------------------------------------
        # Sentinel-2 temporal stack
        # ----------------------------------------------------

        sentinel2 = np.concatenate(
            [
                s2_2023,
                s2_2025
            ],
            axis=0
        )


        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        sentinel1 = torch.from_numpy(
            sentinel1
        ).float()

        sentinel2 = torch.from_numpy(
            sentinel2
        ).float()

        label = torch.from_numpy(
            label
        ).float()


        return {
            "sentinel1": sentinel1,
            "sentinel2": sentinel2,
            "label": label,
            "patch_id": patch_id
        }


# ============================================================
# CREATE DATASETS
# ============================================================

def create_datasets(
    normalize=True
):

    train_csv = os.path.join(
        CSV_ROOT,
        "train.csv"
    )

    val_csv = os.path.join(
        CSV_ROOT,
        "val.csv"
    )

    test_csv = os.path.join(
        CSV_ROOT,
        "test.csv"
    )


    # ========================================================
    # EARLY FUSION
    # ========================================================

    early_train = EarlyFusionDataset(
        train_csv,
        normalize=normalize
    )

    early_val = EarlyFusionDataset(
        val_csv,
        normalize=normalize
    )

    early_test = EarlyFusionDataset(
        test_csv,
        normalize=normalize
    )


    # ========================================================
    # DUAL STREAM
    # ========================================================

    dual_train = DualStreamDataset(
        train_csv,
        normalize=normalize
    )

    dual_val = DualStreamDataset(
        val_csv,
        normalize=normalize
    )

    dual_test = DualStreamDataset(
        test_csv,
        normalize=normalize
    )


    return (
        early_train,
        early_val,
        early_test,
        dual_train,
        dual_val,
        dual_test
    )


# ============================================================
# CREATE DATALOADERS
# ============================================================

def create_dataloaders(
    batch_size=4,
    num_workers=0,
    normalize=True
):

    (
        early_train,
        early_val,
        early_test,
        dual_train,
        dual_val,
        dual_test
    ) = create_datasets(
        normalize=normalize
    )


    # ========================================================
    # EARLY-FUSION LOADERS
    # ========================================================

    early_train_loader = DataLoader(
        early_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    early_val_loader = DataLoader(
        early_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    early_test_loader = DataLoader(
        early_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    # ========================================================
    # DUAL-STREAM LOADERS
    # ========================================================

    dual_train_loader = DataLoader(
        dual_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    dual_val_loader = DataLoader(
        dual_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    dual_test_loader = DataLoader(
        dual_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )


    return (
        early_train_loader,
        early_val_loader,
        early_test_loader,
        dual_train_loader,
        dual_val_loader,
        dual_test_loader
    )


# ============================================================
# TEST DATA PREPARATION
# ============================================================

def test_datasets():

    print()
    print()
    print("=" * 80)
    print("TESTING DATASET PREPARATION")
    print("=" * 80)


    # --------------------------------------------------------
    # Create loaders
    # --------------------------------------------------------

    (
        early_train_loader,
        early_val_loader,
        early_test_loader,
        dual_train_loader,
        dual_val_loader,
        dual_test_loader
    ) = create_dataloaders(
        batch_size=2,
        num_workers=0,
        normalize=NORMALIZE
    )


    # ========================================================
    # TEST EARLY FUSION
    # ========================================================

    print()
    print("-" * 80)
    print("EARLY FUSION")
    print("-" * 80)

    batch = next(
        iter(early_train_loader)
    )

    image = batch["image"]
    label = batch["label"]

    print()
    print("Image shape:")
    print(image.shape)

    print()
    print("Label shape:")
    print(label.shape)

    print()
    print("Expected image:")
    print("(B, 12, 256, 256)")

    print()
    print("Expected label:")
    print("(B, 1, 256, 256)")


    # --------------------------------------------------------
    # Verify channel count
    # --------------------------------------------------------

    assert image.shape[1] == 12

    assert label.shape[1] == 1


    # ========================================================
    # TEST DUAL STREAM
    # ========================================================

    print()
    print()
    print("-" * 80)
    print("DUAL STREAM")
    print("-" * 80)

    batch = next(
        iter(dual_train_loader)
    )

    sentinel1 = batch[
        "sentinel1"
    ]

    sentinel2 = batch[
        "sentinel2"
    ]

    label = batch[
        "label"
    ]


    print()
    print("Sentinel-1 shape:")
    print(sentinel1.shape)

    print()
    print("Sentinel-2 shape:")
    print(sentinel2.shape)

    print()
    print("Label shape:")
    print(label.shape)

    print()
    print("Expected Sentinel-1:")
    print("(B, 4, 256, 256)")

    print()
    print("Expected Sentinel-2:")
    print("(B, 8, 256, 256)")

    print()
    print("Expected label:")
    print("(B, 1, 256, 256)")


    # --------------------------------------------------------
    # Verify channel counts
    # --------------------------------------------------------

    assert sentinel1.shape[1] == 4

    assert sentinel2.shape[1] == 8

    assert label.shape[1] == 1


    # ========================================================
    # PRINT VALUE RANGES
    # ========================================================

    print()
    print()
    print("-" * 80)
    print("VALUE RANGES")
    print("-" * 80)

    print()
    print(
        "Early-fusion image:"
    )

    print(
        "  Min:",
        image.min().item()
    )

    print(
        "  Max:",
        image.max().item()
    )

    print()
    print(
        "Sentinel-1:"
    )

    print(
        "  Min:",
        sentinel1.min().item()
    )

    print(
        "  Max:",
        sentinel1.max().item()
    )

    print()
    print(
        "Sentinel-2:"
    )

    print(
        "  Min:",
        sentinel2.min().item()
    )

    print(
        "  Max:",
        sentinel2.max().item()
    )

    print()
    print(
        "Label:"
    )

    print(
        "  Unique values:",
        torch.unique(label)
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("DATASET TEST PASSED")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    test_datasets()
