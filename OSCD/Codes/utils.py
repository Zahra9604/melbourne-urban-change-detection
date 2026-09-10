import numpy as np 
import torch
from models import CrossModalTemporalGateUNet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
def _normalize_s2(image):

        image = image.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Sentinel-2 reflectance
        #
        # Usually stored as:
        # 0 - 10000
        # ----------------------------------------------------

        if np.nanmax(
            image
        ) > 2.0:

            image = image / 10000.0

        image = np.clip(
            image,
            0.0,
            1.0
        )

        return image



def _normalize_s1(image):

    image = image.astype(
        np.float32
    )

    # ----------------------------------------------------
    # Handle Sentinel-1 dB values
    #
    # Typical range approximately:
    # -30 to 5 dB
    #
    # Clip to a stable range.
    # ----------------------------------------------------

    image = np.clip(
        image,
        -30.0,
        5.0
    )

    image = (
        image + 30.0
    ) / 35.0

    return image



import torch


def collate_oscd_batch(batch):
    """
    Collate function for the patch-based CSV OSCD dataset.

    Expected dataset item:
        {
            "sentinel1": Tensor [4, H, W],
            "sentinel2": Tensor [8, H, W],
            "label":     Tensor [H, W],
            "patch_id":  str,
            "split":     str
        }

    Returns:
        {
            "sentinel1": Tensor [B, 4, H, W],
            "sentinel2": Tensor [B, 8, H, W],
            "label":     Tensor [B, H, W],
            "patch_id":  list[str],
            "split":     list[str]
        }
    """

    # Remove invalid samples if necessary
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    sentinel1 = torch.stack(
        [item["sentinel1"] for item in batch],
        dim=0
    )

    sentinel2 = torch.stack(
        [item["sentinel2"] for item in batch],
        dim=0
    )

    labels = torch.stack(
        [item["label"] for item in batch],
        dim=0
    )

    patch_ids = [
        item.get("patch_id", "")
        for item in batch
    ]

    splits = [
        item.get("split", "")
        for item in batch
    ]

    return {
        "sentinel1": sentinel1,
        "sentinel2": sentinel2,
        "label": labels,
        "patch_id": patch_ids,
        "split": splits,
    }


def load_trained_model(WEIGHTS_PATH):
    model = CrossModalTemporalGateUNet(s1_in_channels=4, s2_in_channels=8).to(
        DEVICE
    )
    if not os.path.exists(WEIGHTS_PATH):
        raise FileNotFoundError(f"Checkpoint not found at: {WEIGHTS_PATH}")

    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model