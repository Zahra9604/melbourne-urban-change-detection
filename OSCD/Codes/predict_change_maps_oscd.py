# ============================================================
# CREATE BINARY CHANGE MAPS FROM TRAINED OSCD MODEL
#
# Input:
#   CSV test patches
#   S1 = 4 bands (T1 VV,VH + T2 VV,VH)
#   S2 = 8 bands (T1 B2,B3,B4,B8 + T2 B2,B3,B4,B8)
#
# Output:
#   Binary GeoTIFF:
#       0 = NO CHANGE
#       1 = CHANGE
#
# NO probability maps are saved.
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import training_config as config

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# IMPORT MODEL
# ============================================================

from models import CrossModalTemporalGateUNet


# ============================================================
# SETTINGS
# ============================================================

PATCH_SIZE = 256
BATCH_SIZE = 4

THRESHOLD = 0.5

CHECKPOINT_NAME = config.BEST_CHECKPOINT_NAME

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "Data",
    "csv",
    "test.csv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "Data",
    "change_maps",
    "test"
)

MELBOURNE_ROOT = (
    r"C:\melbourne-urban-change-detection"
    r"\Melbourne\MelbourneData"
)
CHECKPOINT_PATH = os.path.join(
    MELBOURNE_ROOT,
    CHECKPOINT_NAME
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PRINT HEADER
# ============================================================

print("=" * 80)
print("OSCD CHANGE MAP GENERATION")
print("=" * 80)

print(f"Project root : {PROJECT_ROOT}")
print(f"Test CSV     : {CSV_PATH}")
print(f"Checkpoint   : {CHECKPOINT_PATH}")
print(f"Output       : {OUTPUT_DIR}")
print(f"Device       : {DEVICE}")
print(f"Threshold    : {THRESHOLD}")

if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

print("=" * 80)


# ============================================================
# DATASET
# ============================================================

class OSCDPredictionDataset(Dataset):

    def __init__(self, dataframe):

        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        s1_path = str(row["sentinel1"])
        s2_path = str(row["sentinel2"])
        label_path = str(row["label"])

        # ----------------------------------------------------
        # Read Sentinel-1
        # ----------------------------------------------------

        with rasterio.open(s1_path) as src:

            s1 = src.read().astype(np.float32)

            profile = src.profile.copy()
            transform = src.transform
            crs = src.crs
            width = src.width
            height = src.height

        # ----------------------------------------------------
        # Read Sentinel-2
        # ----------------------------------------------------

        with rasterio.open(s2_path) as src:

            s2 = src.read().astype(np.float32)

        # ----------------------------------------------------
        # Sanity check
        # ----------------------------------------------------

        if s1.shape[0] != 4:

            raise ValueError(
                f"Expected 4 S1 bands, got {s1.shape} "
                f"for {s1_path}"
            )

        if s2.shape[0] != 8:

            raise ValueError(
                f"Expected 8 S2 bands, got {s2.shape} "
                f"for {s2_path}"
            )

        # ----------------------------------------------------
        # Replace NaN / Inf
        # ----------------------------------------------------

        s1 = np.nan_to_num(
            s1,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        s2 = np.nan_to_num(
            s2,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Use the SAME preprocessing used during training.
        #
        # Here we perform per-patch standardization.
        #
        # If your training dataset uses a different
        # normalization, replace this section with EXACTLY
        # the same preprocessing.
        # ----------------------------------------------------

        s1_mean = s1.mean(axis=(1, 2), keepdims=True)
        s1_std = s1.std(axis=(1, 2), keepdims=True)

        s1 = (
            s1 - s1_mean
        ) / (s1_std + 1e-6)

        s2_mean = s2.mean(axis=(1, 2), keepdims=True)
        s2_std = s2.std(axis=(1, 2), keepdims=True)

        s2 = (
            s2 - s2_mean
        ) / (s2_std + 1e-6)

        # ----------------------------------------------------
        # Convert to Torch
        # ----------------------------------------------------

        s1 = torch.from_numpy(s1).float()
        s2 = torch.from_numpy(s2).float()

        return {
            "sentinel1": s1,
            "sentinel2": s2,
            "patch_id": str(row["patch_id"]),
            "label_path": label_path,
            "transform": transform,
            "crs": crs,
            "width": width,
            "height": height,
        }


# ============================================================
# CUSTOM COLLATE
# ============================================================

def collate_fn(batch):

    s1 = torch.stack(
        [item["sentinel1"] for item in batch]
    )

    s2 = torch.stack(
        [item["sentinel2"] for item in batch]
    )

    return {
        "sentinel1": s1,
        "sentinel2": s2,
        "patch_id": [
            item["patch_id"]
            for item in batch
        ],
        "label_path": [
            item["label_path"]
            for item in batch
        ],
        "transform": [
            item["transform"]
            for item in batch
        ],
        "crs": [
            item["crs"]
            for item in batch
        ],
        "width": [
            item["width"]
            for item in batch
        ],
        "height": [
            item["height"]
            for item in batch
        ],
    }


# ============================================================
# LOAD CSV
# ============================================================

print("\nLoading test CSV...")

df = pd.read_csv(CSV_PATH)

print(f"Samples: {len(df)}")
print(f"Columns: {list(df.columns)}")

required_columns = [
    "patch_id",
    "sentinel1",
    "sentinel2",
    "label"
]

for col in required_columns:

    if col not in df.columns:

        raise ValueError(
            f"Missing column '{col}' in test CSV."
        )


# ============================================================
# CHECK FILES
# ============================================================

print("\nChecking files...")

for i, row in df.iterrows():

    for col in ["sentinel1", "sentinel2", "label"]:

        path = str(row[col])

        if not os.path.exists(path):

            raise FileNotFoundError(
                f"\nFile does not exist:\n{path}"
            )

print("All files exist.")


# ============================================================
# DATASET
# ============================================================

dataset = OSCDPredictionDataset(df)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ============================================================
# CREATE MODEL
# ============================================================

print("\nCreating model...")

model = CrossModalTemporalGateUNet(
    s1_in_channels=4,
    s2_in_channels=8
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

if not os.path.exists(CHECKPOINT_PATH):

    raise FileNotFoundError(
        f"\nCheckpoint not found:\n{CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Handle both checkpoint formats
# ------------------------------------------------------------

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    print(
        f"Best validation Dice/IoU: "
        f"{checkpoint.get('best_val_iou', 'unknown')}"
    )

else:

    model.load_state_dict(checkpoint)


model.eval()

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

print("\n")
print("=" * 80)
print("GENERATING CHANGE MAPS")
print("=" * 80)


total_change = 0
total_no_change = 0


with torch.no_grad():

    for batch in tqdm(
        loader,
        desc="Predicting"
    ):

        s1 = batch["sentinel1"].to(DEVICE)
        s2 = batch["sentinel2"].to(DEVICE)

        # ----------------------------------------------------
        # Model prediction
        # ----------------------------------------------------

        logits = model(
            s1,
            s2
        )

        # ----------------------------------------------------
        # Convert logits to probability
        #
        # ONLY used internally.
        # We DO NOT save this probability map.
        # ----------------------------------------------------

        probabilities = torch.sigmoid(logits)

        # ----------------------------------------------------
        # Binary prediction
        #
        # 0 = no change
        # 1 = change
        # ----------------------------------------------------

        predictions = (
            probabilities >= THRESHOLD
        ).to(torch.uint8)

        # ----------------------------------------------------
        # Remove channel dimension
        # [B,1,H,W] -> [B,H,W]
        # ----------------------------------------------------

        predictions = predictions.squeeze(1).cpu().numpy()

        # ----------------------------------------------------
        # Save each patch
        # ----------------------------------------------------

        for i in range(len(predictions)):

            prediction = predictions[i]

            patch_id = batch["patch_id"][i]

            output_name = (
                f"{patch_id}_change.tif"
            )

            output_path = os.path.join(
                OUTPUT_DIR,
                output_name
            )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            unique, counts = np.unique(
                prediction,
                return_counts=True
            )

            stats = dict(
                zip(
                    unique.tolist(),
                    counts.tolist()
                )
            )

            no_change = stats.get(0, 0)
            change = stats.get(1, 0)

            total_no_change += no_change
            total_change += change

            print(
                f"\n{patch_id}"
            )

            print(
                f"  No change pixels : {no_change:,}"
            )

            print(
                f"  Change pixels    : {change:,}"
            )

            print(
                f"  Unique values    : "
                f"{np.unique(prediction)}"
            )

            # ------------------------------------------------
            # GeoTIFF profile
            # ------------------------------------------------

            profile = {
                "driver": "GTiff",
                "height": prediction.shape[0],
                "width": prediction.shape[1],
                "count": 1,
                "dtype": "uint8",
                "crs": batch["crs"][i],
                "transform": batch["transform"][i],
                "compress": "LZW",
                "nodata": None
            }

            # ------------------------------------------------
            # Save binary map
            # ------------------------------------------------

            with rasterio.open(
                output_path,
                "w",
                **profile
            ) as dst:

                dst.write(
                    prediction,
                    1
                )

            print(
                f"  Saved: {output_path}"
            )


# ============================================================
# FINAL STATISTICS
# ============================================================

total_pixels = (
    total_change +
    total_no_change
)

print("\n")
print("=" * 80)
print("CHANGE MAP GENERATION COMPLETE")
print("=" * 80)

print(
    f"Total pixels       : {total_pixels:,}"
)

print(
    f"No-change pixels   : {total_no_change:,}"
)

print(
    f"Change pixels      : {total_change:,}"
)

if total_pixels > 0:

    print(
        f"No-change %        : "
        f"{100 * total_no_change / total_pixels:.2f}%"
    )

    print(
        f"Change %           : "
        f"{100 * total_change / total_pixels:.2f}%"
    )

print("\nOutput directory:")
print(OUTPUT_DIR)

print("=" * 80)