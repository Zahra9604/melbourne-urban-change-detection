# ============================================================
# OSCD DATASET
#
# CSV format:
# patch_id, split, sentinel1, sentinel2, label
#
# Each CSV row represents ONE already-created 256x256 patch.
#
# sentinel1: 4 bands
#   S1 T1 VV, S1 T1 VH, S1 T2 VV, S1 T2 VH
#
# sentinel2: 8 bands
#   S2 T1 B2, B3, B4, B8,
#   S2 T2 B2, B3, B4, B8
#
# label:
#   1 = change
#   0 = no change
# ============================================================

import os
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


class OSCDataset(Dataset):

    def __init__(
        self,
        dataframe,
        augment=False,
    ):
        """
        Parameters
        ----------
        dataframe : pandas.DataFrame
            CSV dataframe containing:
            patch_id, split, sentinel1, sentinel2, label

        augment : bool
            Apply random augmentation to training data.
        """

        self.df = dataframe.reset_index(drop=True)
        self.augment = augment

        required_columns = [
            "patch_id",
            "sentinel1",
            "sentinel2",
            "label",
        ]

        for col in required_columns:
            if col not in self.df.columns:
                raise ValueError(
                    f"Missing required CSV column: '{col}'\n"
                    f"Available columns: {list(self.df.columns)}"
                )

        print("\n" + "=" * 70)
        print("BUILDING OSCD PATCH DATASET")
        print("=" * 70)

        print(f"Number of patches : {len(self.df)}")
        print(f"Augmentation      : {self.augment}")

        # Check first patch
        if len(self.df) > 0:
            row = self.df.iloc[0]

            print("\nFirst patch:")
            print(f"  patch_id : {row['patch_id']}")
            print(f"  S1       : {row['sentinel1']}")
            print(f"  S2       : {row['sentinel2']}")
            print(f"  Label    : {row['label']}")

    def __len__(self):
        return len(self.df)

    # --------------------------------------------------------
    # READ GEOTIFF
    # --------------------------------------------------------

    @staticmethod
    def read_tif(path):

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"\nTIFF file does not exist:\n{path}"
            )

        with rasterio.open(path) as src:
            data = src.read()

        return data

    # --------------------------------------------------------
    # NORMALIZE IMAGE
    # --------------------------------------------------------

    @staticmethod
    def normalize_image(image):
        """
        Per-band robust normalization.

        Input:
            C x H x W

        Output:
            float32 C x H x W
        """

        image = image.astype(np.float32)

        for b in range(image.shape[0]):

            band = image[b]

            valid = np.isfinite(band)

            if not np.any(valid):
                image[b] = 0
                continue

            values = band[valid]

            # Robust percentiles
            p2 = np.percentile(values, 2)
            p98 = np.percentile(values, 98)

            if p98 > p2:

                band = (band - p2) / (p98 - p2)

            else:

                min_val = values.min()
                max_val = values.max()

                if max_val > min_val:
                    band = (band - min_val) / (
                        max_val - min_val
                    )
                else:
                    band = np.zeros_like(band)

            band = np.clip(band, 0.0, 1.0)

            band[~valid] = 0

            image[b] = band

        return image

    # --------------------------------------------------------
    # AUGMENTATION
    # --------------------------------------------------------

    def apply_augmentation(
        self,
        s1,
        s2,
        label,
    ):

        # Horizontal flip
        if np.random.random() < 0.5:

            s1 = np.flip(s1, axis=2).copy()
            s2 = np.flip(s2, axis=2).copy()
            label = np.flip(label, axis=1).copy()

        # Vertical flip
        if np.random.random() < 0.5:

            s1 = np.flip(s1, axis=1).copy()
            s2 = np.flip(s2, axis=1).copy()
            label = np.flip(label, axis=0).copy()

        # 90 degree rotation
        if np.random.random() < 0.5:

            k = np.random.randint(1, 4)

            s1 = np.rot90(
                s1,
                k=k,
                axes=(1, 2)
            ).copy()

            s2 = np.rot90(
                s2,
                k=k,
                axes=(1, 2)
            ).copy()

            label = np.rot90(
                label,
                k=k
            ).copy()

        return s1, s2, label

    # --------------------------------------------------------
    # GET ITEM
    # --------------------------------------------------------

    def __getitem__(self, index):

        row = self.df.iloc[index]

        patch_id = str(row["patch_id"])

        s1_path = str(row["sentinel1"])
        s2_path = str(row["sentinel2"])
        label_path = str(row["label"])

        # ----------------------------------------------------
        # Read data
        # ----------------------------------------------------

        s1 = self.read_tif(s1_path)
        s2 = self.read_tif(s2_path)
        label = self.read_tif(label_path)

        # ----------------------------------------------------
        # Remove extra label dimension
        # ----------------------------------------------------

        if label.ndim == 3:

            label = label[0]

        # ----------------------------------------------------
        # Check shapes
        # ----------------------------------------------------

        if s1.shape[0] != 4:

            raise ValueError(
                f"\nExpected 4 S1 bands but got "
                f"{s1.shape[0]} for {patch_id}\n"
                f"File: {s1_path}"
            )

        if s2.shape[0] != 8:

            raise ValueError(
                f"\nExpected 8 S2 bands but got "
                f"{s2.shape[0]} for {patch_id}\n"
                f"File: {s2_path}"
            )

        if label.shape != s1.shape[1:]:

            raise ValueError(
                f"\nShape mismatch for {patch_id}\n"
                f"S1     : {s1.shape}\n"
                f"S2     : {s2.shape}\n"
                f"Label  : {label.shape}\n"
            )

        # ----------------------------------------------------
        # Normalize images
        # ----------------------------------------------------

        s1 = self.normalize_image(s1)
        s2 = self.normalize_image(s2)

        # ----------------------------------------------------
        # Binary label
        #
        # Labels are already:
        # 0 = no change
        # 1 = change
        # ----------------------------------------------------

        label = label.astype(np.float32)

        # Force binary
        label = (label > 0).astype(np.float32)

        # ----------------------------------------------------
        # Augmentation
        # ----------------------------------------------------

        if self.augment:

            s1, s2, label = self.apply_augmentation(
                s1,
                s2,
                label
            )

        # ----------------------------------------------------
        # Convert to tensors
        # ----------------------------------------------------

        s1 = torch.from_numpy(
            s1.astype(np.float32)
        )

        s2 = torch.from_numpy(
            s2.astype(np.float32)
        )

        label = torch.from_numpy(
            label.astype(np.float32)
        )

        return {
            "sentinel1": s1,
            "sentinel2": s2,
            "label": label,
            "patch_id": patch_id,
        }