
# ============================================================
# MELBOURNE CHANGE DETECTION
#
# TRAINING / VALIDATION / TEST LOSS
# TRAINING SAMPLE VISUALIZATION
# VALIDATION SAMPLE VISUALIZATION
# TEST SAMPLE VISUALIZATION
#
# PREDICTION MAP EXPORT AS GEOTIFF
#
# Supports:
#   1. early_fusion
#   2. dual_stream
#
# GeoTIFF prediction maps:
#   0 = unchanged
#   1 = changed
#
# GeoTIFF probability maps:
#   0.0 - 1.0 = change probability
#
# Spatial reference is copied directly from the
# corresponding Sentinel-2 patch.
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
from glob import glob

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import rasterio

from models import (
    CrossModalTemporalGateUNet,
    ContrastPyramidEarlyFusionUNet
)

import training_config as config


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)

print(
    f"Device: {DEVICE}"
)

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )


# ============================================================
# BASE DIRECTORIES
# ============================================================

root = os.getcwd()

PROJECT_ROOT = os.path.dirname(root)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "MelbourneData"
)


# ============================================================
# CSV FILES
# ============================================================

TRAIN_CSV = os.path.join(
    BASE_DIR,
    "csv",
    "train.csv"
)

VAL_CSV = os.path.join(
    BASE_DIR,
    "csv",
    "val.csv"
)

TEST_CSV = os.path.join(
    BASE_DIR,
    "csv",
    "test.csv"
)


# ============================================================
# RESULTS DIRECTORY
# ============================================================

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)


# ============================================================
# PREDICTION MAP DIRECTORIES
# ============================================================

PREDICTION_MAPS_DIR = os.path.join(
    RESULTS_DIR,
    "prediction_maps"
)

TRAIN_PREDICTION_DIR = os.path.join(
    PREDICTION_MAPS_DIR,
    "train"
)

VAL_PREDICTION_DIR = os.path.join(
    PREDICTION_MAPS_DIR,
    "validation"
)

TEST_PREDICTION_DIR = os.path.join(
    PREDICTION_MAPS_DIR,
    "test"
)

for directory in [
    TRAIN_PREDICTION_DIR,
    VAL_PREDICTION_DIR,
    TEST_PREDICTION_DIR
]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# CHECKPOINT
# ============================================================
if config.STRATEGY not in ["early_fusion", "dual_stream"]:
    
    raise ValueError(
        f"Unknown strategy: "
        f"{config.STRATEGY}"
    )
if config.TRAINING_MODE not in ["scratch", "resume_melbourne", "pretrained_oscd"]:
    
    raise ValueError(
        f"Unknown training mode: "
        f"{config.TRAINING_MODE}"
    )
if config.TRAINING_MODE == "scratch":

    CHECKPOINT_PATH = os.path.join(
        BASE_DIR,
        "checkpoints",
        f"best_{config.STRATEGY}_melbourne_v1.pth"
    )
elif config.TRAINING_MODE == "resume_melbourne":
    
    CHECKPOINT_PATH = os.path.join(
        BASE_DIR,
        "checkpoints",
        f"best_{config.STRATEGY}_melbourne_v2.pth"
    )

elif config.TRAINING_MODE == "pretrained_oscd":

    CHECKPOINT_PATH = os.path.join(
        BASE_DIR,
        "checkpoints",
        f"best_{config.STRATEGY}_melbourne_oscd_finetune.pth"
    )
else:

    raise ValueError(
        f"Unknown training mode: "
        f"{config.TRAINING_MODE}"
    )


# ============================================================
# PARAMETERS
# ============================================================

# Number of training samples to visualize
N_TRAIN_SAMPLES = 3

# Number of validation samples to visualize
N_VAL_SAMPLES = 3

# Number of test samples to visualize
N_TEST_SAMPLES = 3

# Probability threshold
THRESHOLD = 0.25

# ------------------------------------------------------------
# IMPORTANT
#
# If True:
#   Save prediction map for EVERY patch.
#
# If False:
#   Save prediction maps only for the interactive samples.
# ------------------------------------------------------------

SAVE_ALL_PREDICTION_MAPS = True


# ============================================================
# LOAD MODEL
# ============================================================

def load_trained_model():

    print("\n" + "=" * 70)
    print("LOADING TRAINED MODEL")
    print("=" * 70)

    print(
        f"Strategy: {config.STRATEGY}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_PATH}"
    )

    if not os.path.exists(
        CHECKPOINT_PATH
    ):

        raise FileNotFoundError(
            "\nCheckpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )


    # ========================================================
    # EARLY FUSION
    # ========================================================

    if config.STRATEGY == "early_fusion":

        model = ContrastPyramidEarlyFusionUNet(
            in_channels=12,
            base_channels=32,
            out_channels=1
        )


    # ========================================================
    # DUAL STREAM
    # ========================================================

    elif config.STRATEGY == "dual_stream":

        model = CrossModalTemporalGateUNet(
            s1_in_channels=4,
            s2_in_channels=8,
            base_channels=32,
            out_channels=1
        )


    else:

        raise ValueError(
            f"Unknown strategy: "
            f"{config.STRATEGY}"
        )


    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(
        DEVICE
    )

    model.eval()

    print(
        "\nModel successfully loaded."
    )

    print(
        f"Best epoch: "
        f"{checkpoint.get('epoch', 'N/A')}"
    )

    return model


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(
    path,
    name
):

    print(
        f"\nLoading {name} CSV:"
    )

    print(
        path
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            f"\nCSV not found:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required_columns = [
        "patch_id",
        "split",
        "sentinel1_2023",
        "sentinel2_2023",
        "sentinel1_2025",
        "sentinel2_2025",
        "label"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing columns in {name} CSV:\n"
            f"{missing}"
        )

    print(
        f"{name} samples: {len(df)}"
    )

    return df


# ============================================================
# FIND HISTORY FILE
# ============================================================

def find_history_file():

    possible_files = [

        os.path.join(
            BASE_DIR,
            f"training_history_{config.STRATEGY}.csv"
        ),

        os.path.join(
            BASE_DIR,
            f"history_{config.STRATEGY}.csv"
        ),

        os.path.join(
            BASE_DIR,
            "training_history.csv"
        ),

        os.path.join(
            BASE_DIR,
            "history.csv"
        ),

        os.path.join(
            BASE_DIR,
            "checkpoints",
            f"training_history_{config.STRATEGY}.csv"
        ),

        os.path.join(
            BASE_DIR,
            "checkpoints",
            f"history_{config.STRATEGY}.csv"
        ),

        os.path.join(
            BASE_DIR,
            "results",
            f"training_history_{config.STRATEGY}.csv"
        ),

        os.path.join(
            BASE_DIR,
            "results",
            f"history_{config.STRATEGY}.csv"
        )
    ]

    for path in possible_files:

        if os.path.exists(
            path
        ):

            return path

    return None


# ============================================================
# LOAD LOSS HISTORY
# ============================================================

def load_loss_history():

    history_path = find_history_file()

    if history_path is None:

        print("\n" + "=" * 70)
        print("LOSS HISTORY NOT FOUND")
        print("=" * 70)

        print(
            "No training history CSV was found."
        )

        print(
            "\nExpected format:"
        )

        print(
            "epoch,train_loss,val_loss"
        )

        return None


    print("\n" + "=" * 70)
    print("LOADING LOSS HISTORY")
    print("=" * 70)

    print(
        f"History: {history_path}"
    )

    df = pd.read_csv(
        history_path
    )

    print(
        f"Columns: {list(df.columns)}"
    )


    # ========================================================
    # FIND COLUMNS
    # ========================================================

    train_column = None
    val_column = None

    for column in [
        "train_loss",
        "training_loss",
        "loss"
    ]:

        if column in df.columns:

            train_column = column
            break


    for column in [
        "val_loss",
        "validation_loss",
        "valid_loss",
        "val"
    ]:

        if column in df.columns:

            val_column = column
            break


    if train_column is None:

        raise ValueError(
            "Training loss column not found."
        )


    if val_column is None:

        raise ValueError(
            "Validation loss column not found."
        )


    # ========================================================
    # EPOCH
    # ========================================================

    if "epoch" in df.columns:

        epochs = df[
            "epoch"
        ].values

    else:

        epochs = np.arange(
            1,
            len(df) + 1
        )


    train_loss = df[
        train_column
    ].values

    val_loss = df[
        val_column
    ].values


    return (
        epochs,
        train_loss,
        val_loss
    )


# ============================================================
# PLOT LOSS
# ============================================================

def plot_loss():

    history = load_loss_history()

    if history is None:

        return

    epochs, train_loss, val_loss = history

    print("\n" + "=" * 70)
    print("PLOTTING TRAINING / VALIDATION LOSS")
    print("=" * 70)


    plt.figure(
        figsize=(10, 6)
    )


    plt.plot(
        epochs,
        train_loss,
        label="Training Loss",
        linewidth=2
    )


    plt.plot(
        epochs,
        val_loss,
        label="Validation Loss",
        linewidth=2
    )


    plt.xlabel(
        "Epoch",
        fontsize=12
    )

    plt.ylabel(
        "Loss",
        fontsize=12
    )


    plt.title(
        (
            f"Training and Validation Loss\n"
            f"{config.STRATEGY}"
        ),
        fontsize=14
    )


    plt.legend(
        fontsize=11
    )


    plt.grid(
        alpha=0.3
    )


    plt.tight_layout()


    output_path = os.path.join(
        RESULTS_DIR,
        f"loss_curve_{config.STRATEGY}.png"
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    plt.close()


    print(
        f"Saved:\n{output_path}"
    )


# ============================================================
# READ SENTINEL-2 RGB
# ============================================================

def load_rgb(
    path
):

    with rasterio.open(
        path
    ) as src:

        data = src.read()


    if data.shape[0] < 3:

        raise ValueError(
            f"Sentinel-2 image has fewer than "
            f"3 bands:\n{path}"
        )


    # ========================================================
    # ASSUMPTION
    #
    # Band 1 = B2
    # Band 2 = B3
    # Band 3 = B4
    #
    # Natural colour:
    #
    # RGB = B4, B3, B2
    # ========================================================

    blue = data[0]
    green = data[1]
    red = data[2]


    rgb = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )


    # ========================================================
    # PERCENTILE STRETCH
    # ========================================================

    rgb = rgb.astype(
        np.float32
    )

    result = np.zeros_like(
        rgb,
        dtype=np.float32
    )


    for band in range(3):

        image = rgb[
            :, :, band
        ]

        valid = image[
            np.isfinite(image)
        ]

        if len(valid) == 0:

            continue


        low = np.percentile(
            valid,
            2
        )

        high = np.percentile(
            valid,
            98
        )


        if high > low:

            result[
                :, :, band
            ] = (
                image - low
            ) / (
                high - low
            )


    result = np.clip(
        result,
        0,
        1
    )


    return result


# ============================================================
# READ LABEL
# ============================================================

def load_label(
    path
):

    with rasterio.open(
        path
    ) as src:

        label = src.read(
            1
        )


    # --------------------------------------------------------
    # Convert to binary
    # --------------------------------------------------------

    label = (
        label > 0
    ).astype(
        np.uint8
    )


    return label


# ============================================================
# NORMALIZE ARRAY RGB
# ============================================================

def normalize_array_rgb(
    rgb
):

    rgb = rgb.astype(
        np.float32
    )

    result = np.zeros_like(
        rgb
    )


    for band in range(3):

        image = rgb[
            :, :, band
        ]

        valid = image[
            np.isfinite(image)
        ]


        if len(valid) == 0:

            continue


        low = np.percentile(
            valid,
            2
        )

        high = np.percentile(
            valid,
            98
        )


        if high > low:

            result[
                :, :, band
            ] = (
                image - low
            ) / (
                high - low
            )


    return np.clip(
        result,
        0,
        1
    )


# ============================================================
# GET RGB FROM DATASET SAMPLE
# ============================================================

def get_rgb_from_sample(
    sample,
    date_index
):

    if config.STRATEGY == "early_fusion":

        image = sample[
            "image"
        ].detach().cpu().numpy()


        if date_index == 0:

            s2 = image[
                2:6
            ]

        else:

            s2 = image[
                8:12
            ]


    else:

        s2 = sample[
            "sentinel2"
        ].detach().cpu().numpy()


        if date_index == 0:

            s2 = s2[
                0:4
            ]

        else:

            s2 = s2[
                4:8
            ]


    # --------------------------------------------------------
    # B2, B3, B4 -> RGB
    # --------------------------------------------------------

    rgb = np.stack(
        [
            s2[2],
            s2[1],
            s2[0]
        ],
        axis=-1
    )


    rgb = normalize_array_rgb(
        rgb
    )


    return rgb


# ============================================================
# PREDICT ONE SAMPLE
# ============================================================

@torch.no_grad()
def predict_sample(
    model,
    sample
):

    # ========================================================
    # EARLY FUSION
    # ========================================================

    if config.STRATEGY == "early_fusion":

        image = sample[
            "image"
        ]


        if image.ndim == 3:

            image = image.unsqueeze(
                0
            )


        image = image.to(
            DEVICE,
            non_blocking=True
        )


        logits = model(
            image
        )


    # ========================================================
    # DUAL STREAM
    # ========================================================

    elif config.STRATEGY == "dual_stream":

        sentinel1 = sample[
            "sentinel1"
        ]

        sentinel2 = sample[
            "sentinel2"
        ]


        if sentinel1.ndim == 3:

            sentinel1 = sentinel1.unsqueeze(
                0
            )


        if sentinel2.ndim == 3:

            sentinel2 = sentinel2.unsqueeze(
                0
            )


        sentinel1 = sentinel1.to(
            DEVICE,
            non_blocking=True
        )

        sentinel2 = sentinel2.to(
            DEVICE,
            non_blocking=True
        )


        logits = model(
            sentinel1,
            sentinel2
        )


    else:

        raise ValueError(
            f"Unknown strategy: "
            f"{config.STRATEGY}"
        )


    # ========================================================
    # SIGMOID
    # ========================================================

    probability = torch.sigmoid(
        logits
    )


    probability = (
        probability
        .squeeze()
        .cpu()
        .numpy()
    )


    # ========================================================
    # BINARY MAP
    # ========================================================

    prediction = (
        probability >= THRESHOLD
    ).astype(
        np.uint8
    )


    return (
        probability,
        prediction
    )


# ============================================================
# SAVE PREDICTION MAP AS GEOTIFF
# ============================================================

def save_prediction_geotiff(
    prediction,
    probability,
    reference_path,
    output_prediction_path,
    output_probability_path
):

    # ========================================================
    # OPEN REFERENCE SENTINEL-2 PATCH
    # ========================================================

    with rasterio.open(
        reference_path
    ) as src:

        reference_profile = src.profile.copy()

        reference_width = src.width
        reference_height = src.height

        reference_transform = src.transform
        reference_crs = src.crs

        reference_bounds = src.bounds


    # ========================================================
    # CHECK DIMENSIONS
    # ========================================================

    prediction_height, prediction_width = (
        prediction.shape
    )

    probability_height, probability_width = (
        probability.shape
    )


    if (
        prediction_width != reference_width
        or
        prediction_height != reference_height
    ):

        raise ValueError(
            "\nPrediction dimensions do not match "
            "the Sentinel-2 reference patch.\n"
            f"Prediction: "
            f"{prediction_width} x {prediction_height}\n"
            f"Reference: "
            f"{reference_width} x {reference_height}\n"
            f"Reference: {reference_path}"
        )


    if (
        probability_width != reference_width
        or
        probability_height != reference_height
    ):

        raise ValueError(
            "\nProbability dimensions do not match "
            "the Sentinel-2 reference patch.\n"
            f"Probability: "
            f"{probability_width} x {probability_height}\n"
            f"Reference: "
            f"{reference_width} x {reference_height}\n"
            f"Reference: {reference_path}"
        )


    # ========================================================
    # PREDICTION PROFILE
    # ========================================================

    prediction_profile = reference_profile.copy()

    prediction_profile.update(
        {
            "driver": "GTiff",
            "height": reference_height,
            "width": reference_width,
            "count": 1,
            "dtype": "uint8",
            "crs": reference_crs,
            "transform": reference_transform,
            "compress": "lzw",
            "nodata": None
        }
    )


    # ========================================================
    # SAVE BINARY PREDICTION
    #
    # 0 = unchanged
    # 1 = changed
    # ========================================================

    with rasterio.open(
        output_prediction_path,
        "w",
        **prediction_profile
    ) as dst:

        dst.write(
            prediction.astype(
                np.uint8
            ),
            1
        )


    # ========================================================
    # PROBABILITY PROFILE
    # ========================================================

    probability_profile = reference_profile.copy()

    probability_profile.update(
        {
            "driver": "GTiff",
            "height": reference_height,
            "width": reference_width,
            "count": 1,
            "dtype": "float32",
            "crs": reference_crs,
            "transform": reference_transform,
            "compress": "lzw",
            "nodata": None
        }
    )


    # ========================================================
    # SAVE PROBABILITY MAP
    # ========================================================

    with rasterio.open(
        output_probability_path,
        "w",
        **probability_profile
    ) as dst:

        dst.write(
            probability.astype(
                np.float32
            ),
            1
        )


# ============================================================
# SAVE ONE SAMPLE PREDICTION
# ============================================================

def save_sample_prediction(
    model,
    dataset,
    csv_df,
    dataset_index,
    split,
    output_directory
):

    # ========================================================
    # CSV INFORMATION
    # ========================================================

    row = csv_df.iloc[
        dataset_index
    ]


    patch_id = str(
        row["patch_id"]
    )


    s2_2023_path = str(
        row["sentinel2_2023"]
    )


    # ========================================================
    # CHECK REFERENCE IMAGE
    # ========================================================

    if not os.path.exists(
        s2_2023_path
    ):

        raise FileNotFoundError(
            "\nSentinel-2 2023 reference image "
            "not found:\n"
            f"{s2_2023_path}"
        )


    # ========================================================
    # LOAD DATASET SAMPLE
    # ========================================================

    sample = dataset[
        dataset_index
    ]


    # ========================================================
    # PREDICT
    # ========================================================

    probability, prediction = predict_sample(
        model,
        sample
    )


    # ========================================================
    # OUTPUT NAMES
    # ========================================================

    prediction_path = os.path.join(
        output_directory,
        f"{patch_id}_prediction.tif"
    )


    probability_path = os.path.join(
        output_directory,
        f"{patch_id}_probability.tif"
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_prediction_geotiff(

        prediction=prediction,

        probability=probability,

        reference_path=s2_2023_path,

        output_prediction_path=prediction_path,

        output_probability_path=probability_path
    )


    return (
        prediction_path,
        probability_path
    )


# ============================================================
# SAVE ALL DATASET PREDICTION MAPS
# ============================================================

def save_all_prediction_maps(
    model,
    dataset,
    csv_df,
    split,
    output_directory
):

    print("\n" + "=" * 70)
    print(
        f"SAVING {split.upper()} PREDICTION MAPS"
    )
    print("=" * 70)


    n = min(
        len(dataset),
        len(csv_df)
    )


    print(
        f"Number of patches: {n}"
    )

    print(
        f"Output directory:\n"
        f"{output_directory}"
    )


    successful = 0
    failed = 0


    for dataset_index in range(n):

        try:

            row = csv_df.iloc[
                dataset_index
            ]

            patch_id = str(
                row["patch_id"]
            )


            prediction_path, probability_path = (
                save_sample_prediction(

                    model=model,

                    dataset=dataset,

                    csv_df=csv_df,

                    dataset_index=dataset_index,

                    split=split,

                    output_directory=output_directory
                )
            )


            successful += 1


            print(
                f"[{successful + failed}/{n}] "
                f"Saved: {patch_id}"
            )


        except Exception as e:

            failed += 1

            print(
                "\nERROR processing patch "
                f"{dataset_index}:"
            )

            print(
                str(e)
            )


    print("\n" + "-" * 70)

    print(
        f"{split.upper()} prediction maps completed."
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        f"Output:\n{output_directory}"
    )

    print("-" * 70)


# ============================================================
# PLOT ONE SAMPLE
#
# INTERACTIVE NEXT / PREVIOUS
# SYNCHRONIZED ZOOM / PAN
# ============================================================

def plot_one_sample(
    model,
    dataset,
    csv_df,
    index,
    split,
    sample_number,
    indices=None
):

    from matplotlib.widgets import Button


    # ========================================================
    # STATE
    # ========================================================

    if indices is None:

        indices = list(
            range(len(dataset))
        )


    current_position = indices.index(
        index
    )


    # ========================================================
    # FIGURE
    # ========================================================

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(25, 6),
        sharex=True,
        sharey=True
    )


    try:

        fig.canvas.manager.set_window_title(
            f"{split.upper()} | Interactive Patch Viewer"
        )

    except Exception:

        pass


    # Leave space at bottom for buttons

    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.86,
        bottom=0.18,
        wspace=0.05
    )


    # ========================================================
    # SYNCHRONIZED ZOOM / PAN
    # ========================================================

    updating = [False]


    def sync_axes(changed_ax):

        if updating[0]:

            return


        updating[0] = True


        try:

            xlim = changed_ax.get_xlim()
            ylim = changed_ax.get_ylim()


            for ax in axes:

                if ax is not changed_ax:

                    ax.set_xlim(
                        xlim
                    )

                    ax.set_ylim(
                        ylim
                    )


            fig.canvas.draw_idle()


        finally:

            updating[0] = False


    # ========================================================
    # CONNECT CALLBACKS
    # ========================================================

    for ax in axes:

        ax.callbacks.connect(
            "xlim_changed",
            sync_axes
        )

        ax.callbacks.connect(
            "ylim_changed",
            sync_axes
        )


    # ========================================================
    # LOAD PATCH
    # ========================================================

    def load_patch(position):

        # ----------------------------------------------------
        # Dataset index
        # ----------------------------------------------------

        dataset_index = indices[
            position
        ]


        # ----------------------------------------------------
        # CSV information
        # ----------------------------------------------------

        row = csv_df.iloc[
            dataset_index
        ]


        patch_id = str(
            row["patch_id"]
        )


        s2_2023_path = str(
            row["sentinel2_2023"]
        )


        s2_2025_path = str(
            row["sentinel2_2025"]
        )


        label_path = str(
            row["label"]
        )


        # ----------------------------------------------------
        # Check files
        # ----------------------------------------------------

        for path in [
            s2_2023_path,
            s2_2025_path,
            label_path
        ]:

            if not os.path.exists(
                path
            ):

                raise FileNotFoundError(
                    f"\nFile not found:\n{path}"
                )


        # ----------------------------------------------------
        # Load dataset sample
        # ----------------------------------------------------

        sample = dataset[
            dataset_index
        ]


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        probability, prediction = predict_sample(
            model,
            sample
        )


        # ----------------------------------------------------
        # RGB
        # ----------------------------------------------------

        rgb_2023 = load_rgb(
            s2_2023_path
        )


        rgb_2025 = load_rgb(
            s2_2025_path
        )


        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        label = load_label(
            label_path
        )


        # ----------------------------------------------------
        # Clear existing images
        # ----------------------------------------------------

        for ax in axes:

            ax.clear()


        # ====================================================
        # DISPLAY IMAGES
        # ====================================================

        axes[0].imshow(
            rgb_2023,
            interpolation="nearest"
        )


        axes[0].set_title(
            "Sentinel-2 2023 RGB",
            fontsize=12
        )


        axes[1].imshow(
            rgb_2025,
            interpolation="nearest"
        )


        axes[1].set_title(
            "Sentinel-2 2025 RGB",
            fontsize=12
        )


        axes[2].imshow(
            label,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )


        axes[2].set_title(
            "Ground Truth Change",
            fontsize=12
        )


        axes[3].imshow(
            prediction,
            cmap="gray",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )


        axes[3].set_title(
            f"Predicted Change\n"
            f"Threshold = {THRESHOLD}",
            fontsize=12
        )


        probability_plot = axes[4].imshow(
            probability,
            cmap="magma",
            vmin=0,
            vmax=1,
            interpolation="nearest"
        )


        axes[4].set_title(
            "Change Probability",
            fontsize=12
        )


        # ====================================================
        # SAME EXTENT
        # ====================================================

        h, w = rgb_2023.shape[:2]


        for ax in axes:

            ax.set_xticks([])
            ax.set_yticks([])

            ax.set_xlim(
                0,
                w
            )

            ax.set_ylim(
                h,
                0
            )


        # ====================================================
        # COLORBAR
        # ====================================================

        if hasattr(
            fig,
            "_probability_colorbar"
        ):

            try:

                fig._probability_colorbar.remove()

            except Exception:

                pass


        fig._probability_colorbar = fig.colorbar(
            probability_plot,
            ax=axes[4],
            fraction=0.046,
            pad=0.04
        )


        fig._probability_colorbar.set_label(
            "Probability",
            rotation=270,
            labelpad=15
        )


        # ====================================================
        # TITLE
        # ====================================================

        fig.suptitle(
            (
                f"{split.upper()} PATCH "
                f"{position + 1} / {len(indices)}   |   "
                f"Dataset Index: {dataset_index}   |   "
                f"Patch ID: {patch_id}\n"
                f"{config.STRATEGY}   |   "
                f"Zoom/pan synchronized across all images"
            ),
            fontsize=14,
            fontweight="bold"
        )


        # ====================================================
        # BUTTON STATUS
        # ====================================================

        if position == 0:

            previous_button.set_active(
                False
            )

        else:

            previous_button.set_active(
                True
            )


        if position == len(indices) - 1:

            next_button.set_active(
                False
            )

        else:

            next_button.set_active(
                True
            )


        # ====================================================
        # SAVE CURRENT PATCH IMAGE
        # ====================================================

        output_path = os.path.join(
            RESULTS_DIR,
            (
                f"{split}_"
                f"{position + 1:03d}_"
                f"{patch_id}_"
                f"{config.STRATEGY}.png"
            )
        )


        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )


        print(
            f"Displayed {split} patch "
            f"{position + 1}/{len(indices)}: "
            f"{patch_id}"
        )


        print(
            f"Saved: {output_path}"
        )


        fig.canvas.draw_idle()


    # ========================================================
    # BUTTON CALLBACKS
    # ========================================================

    def previous_patch(event):

        nonlocal current_position


        if current_position > 0:

            current_position -= 1


            load_patch(
                current_position
            )


    def next_patch(event):

        nonlocal current_position


        if current_position < len(indices) - 1:

            current_position += 1


            load_patch(
                current_position
            )


    # ========================================================
    # BUTTON LOCATIONS
    # ========================================================

    ax_previous = plt.axes(
        [0.35, 0.055, 0.12, 0.065]
    )


    ax_next = plt.axes(
        [0.53, 0.055, 0.12, 0.065]
    )


    # ========================================================
    # CREATE BUTTONS
    # ========================================================

    previous_button = Button(
        ax_previous,
        "Previous"
    )


    next_button = Button(
        ax_next,
        "Next"
    )


    # ========================================================
    # CONNECT BUTTONS
    # ========================================================

    previous_button.on_clicked(
        previous_patch
    )


    next_button.on_clicked(
        next_patch
    )


    # ========================================================
    # KEYBOARD SHORTCUTS
    # ========================================================

    def keyboard_navigation(event):

        if event.key in [
            "right",
            "n"
        ]:

            next_patch(
                event
            )


        elif event.key in [
            "left",
            "p"
        ]:

            previous_patch(
                event
            )


    fig.canvas.mpl_connect(
        "key_press_event",
        keyboard_navigation
    )


    # ========================================================
    # INITIAL PATCH
    # ========================================================

    load_patch(
        current_position
    )


    # ========================================================
    # SHOW
    # ========================================================

    plt.show()

    plt.close()


# ============================================================
# VISUALIZE DATASET — INTERACTIVE
# ============================================================

def visualize_dataset(
    model,
    dataset,
    csv_df,
    split,
    number_of_samples
):

    print("\n" + "=" * 70)

    print(
        f"{split.upper()} INTERACTIVE SAMPLE VISUALIZATION"
    )

    print("=" * 70)


    # ========================================================
    # NUMBER OF SAMPLES
    # ========================================================

    n = min(
        number_of_samples,
        len(dataset),
        len(csv_df)
    )


    print(
        f"Interactive viewer contains "
        f"{n} selected samples."
    )


    if n == 0:

        print(
            f"No samples available for {split}."
        )

        return


    # ========================================================
    # CHOOSE EVENLY DISTRIBUTED SAMPLES
    # ========================================================

    if n == 1:

        indices = [
            0
        ]

    else:

        indices = np.linspace(
            0,
            len(dataset) - 1,
            n,
            dtype=int
        ).tolist()


    # ========================================================
    # OPEN ONE INTERACTIVE WINDOW
    # ========================================================

    plot_one_sample(

        model=model,

        dataset=dataset,

        csv_df=csv_df,

        index=indices[0],

        split=split,

        sample_number=1,

        indices=indices
    )


# ============================================================
# CHECK DATASET / CSV
# ============================================================

def check_dataset_csv(
    dataset,
    csv_df,
    name
):

    print(
        f"\n{name} dataset: "
        f"{len(dataset)}"
    )

    print(
        f"{name} CSV: "
        f"{len(csv_df)}"
    )


    if len(dataset) != len(csv_df):

        raise ValueError(
            f"\n{name} dataset and {name} CSV "
            f"have different numbers of samples."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # LOAD CSV FILES
    # ========================================================

    train_df = load_csv(
        TRAIN_CSV,
        "TRAIN"
    )


    val_df = load_csv(
        VAL_CSV,
        "VALIDATION"
    )


    test_df = load_csv(
        TEST_CSV,
        "TEST"
    )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    model = load_trained_model()


    # ========================================================
    # CREATE DATA LOADERS
    # ========================================================

    print("\n" + "=" * 70)
    print("CREATING DATA LOADERS")
    print("=" * 70)


    from melbourne_dataset import create_dataloaders


    (
        train_loader,
        val_loader,
        test_loader,
        dual_train_loader,
        dual_val_loader,
        dual_test_loader
    ) = create_dataloaders(

        batch_size=1,

        num_workers=config.NUM_WORKERS,

        normalize=config.NORMALIZE
    )


    # ========================================================
    # SELECT DATASETS
    # ========================================================

    if config.STRATEGY == "early_fusion":

        train_dataset = train_loader.dataset

        val_dataset = val_loader.dataset

        test_dataset = test_loader.dataset


    elif config.STRATEGY == "dual_stream":

        train_dataset = dual_train_loader.dataset

        val_dataset = dual_val_loader.dataset

        test_dataset = dual_test_loader.dataset


    else:

        raise ValueError(
            f"Unknown strategy: "
            f"{config.STRATEGY}"
        )


    # ========================================================
    # CHECK DATASET / CSV LENGTH
    # ========================================================

    print("\n" + "=" * 70)
    print("DATASET / CSV CHECK")
    print("=" * 70)


    check_dataset_csv(
        train_dataset,
        train_df,
        "Train"
    )


    check_dataset_csv(
        val_dataset,
        val_df,
        "Validation"
    )


    check_dataset_csv(
        test_dataset,
        test_df,
        "Test"
    )


    print(
        "\nAll datasets match their CSV files."
    )


    # ========================================================
    # 1. PLOT LOSS
    # ========================================================

    plot_loss()


    # ========================================================
    # 2. SAVE TRAINING PREDICTION MAPS
    # ========================================================

    if SAVE_ALL_PREDICTION_MAPS:

        save_all_prediction_maps(

            model=model,

            dataset=train_dataset,

            csv_df=train_df,

            split="train",

            output_directory=TRAIN_PREDICTION_DIR
        )


    # ========================================================
    # 3. SAVE VALIDATION PREDICTION MAPS
    # ========================================================

    if SAVE_ALL_PREDICTION_MAPS:

        save_all_prediction_maps(

            model=model,

            dataset=val_dataset,

            csv_df=val_df,

            split="validation",

            output_directory=VAL_PREDICTION_DIR
        )


    # ========================================================
    # 4. SAVE TEST PREDICTION MAPS
    # ========================================================

    if SAVE_ALL_PREDICTION_MAPS:

        save_all_prediction_maps(

            model=model,

            dataset=test_dataset,

            csv_df=test_df,

            split="test",

            output_directory=TEST_PREDICTION_DIR
        )


    # ========================================================
    # 5. TRAINING SAMPLE VISUALIZATION
    # ========================================================

    visualize_dataset(

        model=model,

        dataset=train_dataset,

        csv_df=train_df,

        split="train",

        number_of_samples=N_TRAIN_SAMPLES
    )


    # ========================================================
    # 6. VALIDATION SAMPLE VISUALIZATION
    # ========================================================

    visualize_dataset(

        model=model,

        dataset=val_dataset,

        csv_df=val_df,

        split="validation",

        number_of_samples=N_VAL_SAMPLES
    )


    # ========================================================
    # 7. TEST SAMPLE VISUALIZATION
    # ========================================================

    visualize_dataset(

        model=model,

        dataset=test_dataset,

        csv_df=test_df,

        split="test",

        number_of_samples=N_TEST_SAMPLES
    )


    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    print("\n" + "=" * 70)
    print("VISUALIZATION AND PREDICTION EXPORT COMPLETED")
    print("=" * 70)


    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIR
    )


    print(
        "\nPrediction GeoTIFF directories:"
    )

    print(
        f"TRAIN:\n{TRAIN_PREDICTION_DIR}"
    )

    print(
        f"\nVALIDATION:\n{VAL_PREDICTION_DIR}"
    )

    print(
        f"\nTEST:\n{TEST_PREDICTION_DIR}"
    )


    print("\nGenerated files include:")

    print(
        "  loss_curve_<strategy>.png"
    )

    print(
        "  train_*.png"
    )

    print(
        "  validation_*.png"
    )

    print(
        "  test_*.png"
    )

    print(
        "  <patch_id>_prediction.tif"
    )

    print(
        "  <patch_id>_probability.tif"
    )


    print("\nPrediction map convention:")

    print(
        "  0 = unchanged"
    )

    print(
        "  1 = changed"
    )

    print(
        f"\nProbability threshold: {THRESHOLD}"
    )


    print(
        "\nGeoTIFF spatial reference:"
    )

    print(
        "  CRS       = copied from Sentinel-2 patch"
    )

    print(
        "  Transform = copied from Sentinel-2 patch"
    )

    print(
        "  Width     = copied from Sentinel-2 patch"
    )

    print(
        "  Height    = copied from Sentinel-2 patch"
    )

    print(
        "  Alignment = pixel-for-pixel with reference patch"
    )

