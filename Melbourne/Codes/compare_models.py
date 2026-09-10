# ============================================================
# MELBOURNE CHANGE DETECTION
#
# PREDICT ENTIRE DATASET USING ALL BEST MODELS
#
# Automatically:
#
#   1. Finds all best_*.pth checkpoints
#   2. Identifies model strategy from filename
#   3. Loads the correct model architecture
#   4. Predicts TRAIN / VALIDATION / TEST
#   5. Saves prediction maps as GeoTIFF
#   6. Saves probability maps as GeoTIFF
#   7. Preserves CRS / transform / resolution / dimensions
#   8. Creates TEST comparison figures
#   9. Shows Ground Truth + ALL model predictions together
#  10. Synchronizes zoom/pan across ALL images
#  11. Interactive threshold slider
#
# Supported strategies:
#
#   best_early_fusion*.pth
#   best_dual_stream*.pth
#
# Prediction convention:
#
#   0 = unchanged
#   1 = changed
#
# Probability:
#
#   0.0 - 1.0
#
# IMPORTANT:
#
# The threshold slider operates on SAVED probability maps.
#
# Therefore:
#
#   Change threshold
#       ->
#   ALL model predictions update immediately
#
# No model inference is performed when moving the slider.
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
from glob import glob
from unittest import result

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider

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


print("=" * 80)
print(
    "MELBOURNE CHANGE DETECTION - ALL BEST MODELS"
)
print("=" * 80)


print(
    f"Device: {DEVICE}"
)


if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"CUDA: {torch.version.cuda}"
    )


# ============================================================
# BASE DIRECTORIES
# ============================================================

ROOT = os.getcwd()

PROJECT_ROOT = os.path.dirname(
    ROOT
)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "MelbourneData"
)


# ============================================================
# DATA DIRECTORIES
# ============================================================

CSV_DIR = os.path.join(
    BASE_DIR,
    "csv"
)

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# CSV FILES
# ============================================================

TRAIN_CSV = os.path.join(
    CSV_DIR,
    "train.csv"
)

VAL_CSV = os.path.join(
    CSV_DIR,
    "val.csv"
)

TEST_CSV = os.path.join(
    CSV_DIR,
    "test.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

ALL_PREDICTIONS_DIR = os.path.join(
    RESULTS_DIR,
    "all_model_predictions"
)

TRAIN_OUTPUT_DIR = os.path.join(
    ALL_PREDICTIONS_DIR,
    "train"
)

VAL_OUTPUT_DIR = os.path.join(
    ALL_PREDICTIONS_DIR,
    "validation"
)

TEST_OUTPUT_DIR = os.path.join(
    ALL_PREDICTIONS_DIR,
    "test"
)

COMPARISON_DIR = os.path.join(
    RESULTS_DIR,
    "test_model_comparison"
)


for directory in [

    ALL_PREDICTIONS_DIR,
    TRAIN_OUTPUT_DIR,
    VAL_OUTPUT_DIR,
    TEST_OUTPUT_DIR,
    COMPARISON_DIR

]:

    os.makedirs(
        directory,
        exist_ok=True
    )


# ============================================================
# PARAMETERS
# ============================================================

# ------------------------------------------------------------
# Default probability threshold
# ------------------------------------------------------------

THRESHOLD = 0.27


# ------------------------------------------------------------
# Save comparison figures?
# ------------------------------------------------------------

SAVE_TEST_COMPARISON_FIGURES = True


# ------------------------------------------------------------
# Number of test figures to show interactively
#
# 0 = do not show any figure
# ------------------------------------------------------------

SHOW_TEST_FIGURES = 3


# ------------------------------------------------------------
# Save probability maps?
# ------------------------------------------------------------

SAVE_PROBABILITY_MAPS = True


# ------------------------------------------------------------
# Maximum number of columns
# ------------------------------------------------------------

MAX_COLUMNS = 4


# ------------------------------------------------------------
# Size of each subplot
# ------------------------------------------------------------

SUBPLOT_SIZE = 6


# ============================================================
# FIND ALL BEST MODELS
# ============================================================

def find_all_best_models():

    print("\n" + "=" * 80)
    print(
        "SEARCHING FOR BEST MODELS"
    )
    print("=" * 80)


    pattern = os.path.join(
        CHECKPOINT_DIR,
        "best_*.pth"
    )


    checkpoint_paths = sorted(
        glob(pattern)
    )


    if not checkpoint_paths:

        raise FileNotFoundError(

            "\nNo best model checkpoints found.\n"

            f"Search directory:\n"
            f"{CHECKPOINT_DIR}\n"

            f"Pattern:\n"
            f"{pattern}"
        )


    print(
        f"Found {len(checkpoint_paths)} checkpoint(s):"
    )


    valid_models = []


    for path in checkpoint_paths:

        filename = os.path.basename(
            path
        )

        lower_name = filename.lower()


        # ----------------------------------------------------
        # Determine strategy
        # ----------------------------------------------------

        if "dual_stream" in lower_name:

            strategy = "dual_stream"


        elif "early_fusion" in lower_name:

            strategy = "early_fusion"


        else:

            print(
                "\nWARNING: "
                f"Cannot identify strategy for:\n"
                f"  {filename}"
            )

            print(
                "Skipping this checkpoint."
            )

            continue


        model_info = {

            "path": path,

            "filename": filename,

            "strategy": strategy

        }


        valid_models.append(
            model_info
        )


        print(
            f"\n  Model: {filename}"
        )

        print(
            f"  Strategy: {strategy}"
        )


    if not valid_models:

        raise RuntimeError(
            "\nNo supported best models found."
        )


    print(
        f"\nSupported models: "
        f"{len(valid_models)}"
    )


    return valid_models


# ============================================================
# CREATE MODEL FROM STRATEGY
# ============================================================

def create_model(
    strategy
):


    # ========================================================
    # EARLY FUSION
    # ========================================================

    if strategy == "early_fusion":

        model = ContrastPyramidEarlyFusionUNet(

            in_channels=12,

            base_channels=32,

            out_channels=1

        )


    # ========================================================
    # DUAL STREAM
    # ========================================================

    elif strategy == "dual_stream":

        model = CrossModalTemporalGateUNet(

            s1_in_channels=4,

            s2_in_channels=8,

            base_channels=32,

            out_channels=1

        )


    else:

        raise ValueError(
            f"Unknown strategy: {strategy}"
        )


    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(
    model_info
):


    checkpoint_path = model_info[
        "path"
    ]

    strategy = model_info[
        "strategy"
    ]

    filename = model_info[
        "filename"
    ]


    print("\n" + "-" * 80)

    print(
        "Loading model:"
    )

    print(
        f"  {filename}"
    )

    print(
        f"  Strategy: {strategy}"
    )


    # --------------------------------------------------------
    # Create architecture
    # --------------------------------------------------------

    model = create_model(
        strategy
    )


    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(

        checkpoint_path,

        map_location=DEVICE

    )


    if "model_state_dict" not in checkpoint:

        raise KeyError(

            f"\nCheckpoint does not contain "
            f"'model_state_dict':\n"
            f"{checkpoint_path}"

        )


    model.load_state_dict(

        checkpoint[
            "model_state_dict"
        ]

    )


    model = model.to(
        DEVICE
    )


    model.eval()


    print(
        "  Model loaded successfully."
    )


    print(
        f"  Epoch: "
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


    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(

            f"\n{name} CSV not found:\n"
            f"{path}"

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
        f"{name}: {len(df)} samples"
    )


    return df


# ============================================================
# PREDICT ONE SAMPLE
# ============================================================

@torch.no_grad()
def predict_sample(
    model,
    strategy,
    sample
):


    # ========================================================
    # EARLY FUSION
    # ========================================================

    if strategy == "early_fusion":

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

    elif strategy == "dual_stream":

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
            f"Unknown strategy: {strategy}"
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

        .detach()

        .cpu()

        .numpy()

    )


    # ========================================================
    # FORCE 2D
    # ========================================================

    probability = np.squeeze(
        probability
    )


    if probability.ndim != 2:

        raise ValueError(

            "\nModel prediction is not 2D.\n"

            f"Shape: {probability.shape}"

        )


    # ========================================================
    # BINARY PREDICTION
    #
    # 0 = unchanged
    # 1 = changed
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
# SAVE GEOTIFF
# ============================================================

def save_geotiff(
    array,
    reference_path,
    output_path,
    dtype,
    nodata=None
):


    # ========================================================
    # READ REFERENCE GEOTIFF
    # ========================================================

    with rasterio.open(
        reference_path
    ) as src:

        profile = src.profile.copy()

        width = src.width

        height = src.height

        transform = src.transform

        crs = src.crs


    # ========================================================
    # CHECK SHAPE
    # ========================================================

    if array.shape != (

        height,

        width

    ):

        raise ValueError(

            "\nPrediction shape does not match "
            "reference image.\n"

            f"Prediction shape: "
            f"{array.shape}\n"

            f"Reference shape: "
            f"({height}, {width})\n"

            f"Reference:\n"
            f"{reference_path}"

        )


    # ========================================================
    # UPDATE PROFILE
    # ========================================================

    profile.update(

        driver="GTiff",

        height=height,

        width=width,

        count=1,

        dtype=dtype,

        crs=crs,

        transform=transform,

        compress="lzw",

        nodata=nodata

    )


    # ========================================================
    # WRITE
    # ========================================================

    with rasterio.open(

        output_path,

        "w",

        **profile

    ) as dst:

        dst.write(

            array.astype(dtype),

            1

        )


# ============================================================
# CREATE SAFE MODEL DIRECTORY NAME
# ============================================================

def model_output_name(
    model_info
):

    filename = model_info[
        "filename"
    ]


    name = os.path.splitext(
        filename
    )[0]


    return name


# ============================================================
# PREDICT ENTIRE DATASET
# ============================================================

def predict_dataset(

    model_info,

    model,

    dataset,

    csv_df,

    split,

    output_root

):


    model_name = model_output_name(
        model_info
    )


    model_directory = os.path.join(

        output_root,

        model_name

    )


    prediction_directory = os.path.join(

        model_directory,

        "prediction"

    )


    probability_directory = os.path.join(

        model_directory,

        "probability"

    )


    os.makedirs(

        prediction_directory,

        exist_ok=True

    )


    if SAVE_PROBABILITY_MAPS:

        os.makedirs(

            probability_directory,

            exist_ok=True

        )


    print("\n" + "=" * 80)

    print(
        f"PREDICTING {split.upper()}"
    )

    print(
        f"Model: {model_name}"
    )

    print(
        f"Samples: {len(dataset)}"
    )

    print("=" * 80)


    successful = 0

    failed = 0


    # ========================================================
    # LOOP THROUGH ALL PATCHES
    # ========================================================

    for index in range(
        len(dataset)
    ):


        try:

            # ------------------------------------------------
            # CSV row
            # ------------------------------------------------

            row = csv_df.iloc[
                index
            ]


            patch_id = str(
                row["patch_id"]
            )


            # ------------------------------------------------
            # Reference Sentinel-2
            # ------------------------------------------------

            reference_path = str(

                row[
                    "sentinel2_2023"
                ]

            )


            if not os.path.exists(
                reference_path
            ):

                raise FileNotFoundError(

                    f"Reference image not found:\n"
                    f"{reference_path}"

                )


            # ------------------------------------------------
            # Dataset sample
            # ------------------------------------------------

            sample = dataset[
                index
            ]


            # ------------------------------------------------
            # Prediction
            # ------------------------------------------------

            probability, prediction = (

                predict_sample(

                    model=model,

                    strategy=model_info[
                        "strategy"
                    ],

                    sample=sample

                )

            )


            # ------------------------------------------------
            # Output names
            # ------------------------------------------------

            prediction_path = os.path.join(

                prediction_directory,

                f"{patch_id}_prediction.tif"

            )


            probability_path = os.path.join(

                probability_directory,

                f"{patch_id}_probability.tif"

            )


            # ------------------------------------------------
            # Save prediction
            # ------------------------------------------------

            save_geotiff(

                array=prediction,

                reference_path=reference_path,

                output_path=prediction_path,

                dtype="uint8",

                nodata=None

            )


            # ------------------------------------------------
            # Save probability
            # ------------------------------------------------

            if SAVE_PROBABILITY_MAPS:

                save_geotiff(

                    array=probability,

                    reference_path=reference_path,

                    output_path=probability_path,

                    dtype="float32",

                    nodata=None

                )


            successful += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (

                successful % 10 == 0

                or

                index == len(dataset) - 1

            ):

                print(

                    f"[{index + 1}/{len(dataset)}] "

                    f"{patch_id}"

                )


        except Exception as e:

            failed += 1


            print(
                "\nERROR"
            )


            print(
                f"Index: {index}"
            )


            print(
                f"Error: {e}"
            )


    print("\n" + "-" * 80)

    print(
        f"{split.upper()} COMPLETE"
    )


    print(
        f"Model: {model_name}"
    )


    print(
        f"Successful: {successful}"
    )


    print(
        f"Failed: {failed}"
    )


    print(
        f"Prediction directory:\n"
        f"{prediction_directory}"
    )


    if SAVE_PROBABILITY_MAPS:

        print(

            f"Probability directory:\n"
            f"{probability_directory}"

        )


    print("-" * 80)


    return {

        "model_name":
            model_name,

        "prediction_directory":
            prediction_directory,

        "probability_directory": (

            probability_directory

            if SAVE_PROBABILITY_MAPS

            else None

        )

    }


# ============================================================
# LOAD RGB
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


    rgb = rgb.astype(
        np.float32
    )


    result = np.zeros_like(

        rgb,

        dtype=np.float32

    )


    # ========================================================
    # PERCENTILE STRETCH
    # ========================================================

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
# LOAD GROUND TRUTH
# ============================================================

def load_ground_truth(
    path
):


    with rasterio.open(
        path
    ) as src:

        label = src.read(
            1
        )


    # ========================================================
    # Ground truth:
    #
    # 0 = unchanged
    # >0 = change
    #
    # Convert:
    #
    # 0 = unchanged
    # 1 = changed
    # ========================================================

    label = (

        label > 0

    ).astype(
        np.uint8
    )


    return label


# ============================================================
# READ PREDICTION GEOTIFF
# ============================================================

def read_prediction(
    path
):


    with rasterio.open(
        path
    ) as src:

        prediction = src.read(
            1
        )

        transform = src.transform

        crs = src.crs

        width = src.width

        height = src.height


    return (

        prediction,

        transform,

        crs,

        width,

        height

    )


# ============================================================
# READ PROBABILITY GEOTIFF
# ============================================================

def read_probability(
    path
):


    with rasterio.open(
        path
    ) as src:

        probability = src.read(
            1
        )


    return probability


# ============================================================
# CHECK SPATIAL ALIGNMENT
# ============================================================

def check_alignment(

    reference_path,

    prediction_path

):


    with rasterio.open(
        reference_path
    ) as reference:

        with rasterio.open(
            prediction_path
        ) as prediction:


            same_crs = (

                reference.crs
                ==
                prediction.crs

            )


            same_transform = (

                reference.transform
                ==
                prediction.transform

            )


            same_width = (

                reference.width
                ==
                prediction.width

            )


            same_height = (

                reference.height
                ==
                prediction.height

            )


    return (

        same_crs

        and

        same_transform

        and

        same_width

        and

        same_height

    )


# ============================================================
# GET STRATEGY DISPLAY NAME
# ============================================================

def strategy_display_name(
    strategy
):


    if strategy == "early_fusion":

        return "Early Fusion"


    if strategy == "dual_stream":

        return "Dual Stream"


    return strategy


# ============================================================
# GET DATA DISPLAY NAME
# ============================================================
def data_display_name(checkpoint_name):
    """
    Determine the training-data label from the checkpoint filename.
    """

    checkpoint_name = checkpoint_name.lower()

    if "melbourne_oscd_finetune" in checkpoint_name:
        return "OSCD + Melbourne"

    elif "oscd" in checkpoint_name:
        return "OSCD"

    return "Unknown"


# ============================================================
# CREATE INTERACTIVE TEST COMPARISON VIEWER
#
# Features:
#
#   Previous button
#   Next button
#   Threshold slider
#
# The viewer uses SAVED probability maps.
#
# Therefore:
#
#   Change threshold
#       ->
#   ALL model predictions update
#
# without model inference.
#
# The Previous / Next buttons change the TEST PATCH.
#
# ============================================================


def create_test_comparison_viewer(

    test_df,

    prediction_results_by_model

):

    # ========================================================
    # NUMBER OF TEST PATCHES
    # ========================================================

    number_of_test_samples = len(
        test_df
    )

    if number_of_test_samples == 0:

        print(
            "\nNo test samples available."
        )

        return


    print("\n" + "=" * 80)

    print(
        "STARTING INTERACTIVE TEST COMPARISON VIEWER"
    )

    print("=" * 80)

    print(
        f"Test patches: {number_of_test_samples}"
    )

    print(
        "Use Previous / Next buttons to navigate."
    )

    print(
        "Use the threshold slider to update all model predictions."
    )

    print("=" * 80)


    # ========================================================
    # PREPARE ALL TEST RESULTS
    #
    # For every test patch we store:
    #
    #   RGB 2023
    #   RGB 2025
    #   Ground truth
    #   Probability maps for all models
    #
    # No model inference happens here.
    # ========================================================

    all_patch_results = []


    for index in range(
        number_of_test_samples
    ):

        row = test_df.iloc[
            index
        ]

        patch_id = str(
            row["patch_id"]
        )


        # ----------------------------------------------------
        # IMAGE PATHS
        # ----------------------------------------------------

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
        # LOAD RGB
        # ----------------------------------------------------

        try:

            rgb_2023 = load_rgb(
                s2_2023_path
            )

            rgb_2025 = load_rgb(
                s2_2025_path
            )

            ground_truth = load_ground_truth(
                label_path
            )

        except Exception as e:

            print(
                f"\nWARNING: Could not load patch "
                f"{patch_id}"
            )

            print(
                f"Error: {e}"
            )

            continue


        # ----------------------------------------------------
        # COLLECT MODEL PROBABILITY MAPS
        # ----------------------------------------------------

        model_results = []


        for (

            model_name,

            model_data

        ) in prediction_results_by_model.items():


            model_info = model_data[
                "model_info"
            ]

            model_paths = model_data[
                "paths"
            ]


            if index >= len(
                model_paths
            ):

                continue


            prediction_path = (
                model_paths[
                    index
                ]
            )


            if not os.path.exists(
                prediction_path
            ):

                continue


            # ------------------------------------------------
            # Probability directory
            # ------------------------------------------------

            prediction_directory = (
                os.path.dirname(
                    prediction_path
                )
            )


            model_directory = (
                os.path.dirname(
                    prediction_directory
                )
            )


            probability_directory = os.path.join(

                model_directory,

                "probability"

            )


            probability_path = os.path.join(

                probability_directory,

                f"{patch_id}_probability.tif"

            )


            if not os.path.exists(
                probability_path
            ):

                print(
                    "\nWARNING: Probability map "
                    "not found:"
                )

                print(
                    probability_path
                )

                continue


            # ------------------------------------------------
            # Read probability map
            # ------------------------------------------------

            probability = read_probability(

                probability_path

            )


            model_results.append(

                {

                    "model_name":
                        model_name,

                    "strategy":
                        model_info[
                            "strategy"
                        ],

                    "prediction_path":
                        prediction_path,

                    "probability_path":
                        probability_path,

                    "probability":
                        probability

                }

            )


        # ----------------------------------------------------
        # Store patch
        # ----------------------------------------------------

        if model_results:

            all_patch_results.append(

                {

                    "original_index":
                        index,

                    "patch_id":
                        patch_id,

                    "rgb_2023":
                        rgb_2023,

                    "rgb_2025":
                        rgb_2025,

                    "ground_truth":
                        ground_truth,

                    "models":
                        model_results

                }

            )


    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if not all_patch_results:

        print(
            "\nNo valid test comparison results found."
        )

        return


    print(
        f"\nLoaded {len(all_patch_results)} "
        f"test patches for interactive viewing."
    )


    # ========================================================
    # VIEWER STATE
    # ========================================================

    current_index = 0


    # ========================================================
    # FIRST PATCH
    # ========================================================

    first_patch = all_patch_results[
        current_index
    ]

    number_of_models = len(
        first_patch["models"]
    )


    # ========================================================
    # TOTAL IMAGES
    # ========================================================

    total_images = (

        3

        +

        number_of_models

    )


    # ========================================================
    # COLUMNS
    # ========================================================

    number_of_columns = min(

        MAX_COLUMNS,

        total_images

    )


    # ========================================================
    # ROWS
    # ========================================================

    number_of_rows = int(

        np.ceil(

            total_images
            /
            number_of_columns

        )

    )


    # ========================================================
    # FIGURE SIZE
    # ========================================================

    figure_width = (

        SUBPLOT_SIZE

        *

        number_of_columns

    )


    figure_height = (

        SUBPLOT_SIZE

        *

        number_of_rows

    )


    # Extra space for controls

    figure_height += 2.0


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig, axes = plt.subplots(

        number_of_rows,

        number_of_columns,

        figsize=(

            figure_width,

            figure_height

        ),

        squeeze=False,

        sharex=True,

        sharey=True

    )


    axes = axes.flatten()
    # ========================================================
    # OPEN FIGURE WINDOW MAXIMIZED
    # ========================================================

    manager = plt.get_current_fig_manager()

    try:
        manager.window.state("zoomed")   # Windows
    except Exception:
        try:
            manager.window.showMaximized()  # Qt backends
        except Exception:
            pass

    # ========================================================
    # THRESHOLD SLIDER
    # ========================================================

    slider_axis = fig.add_axes(

        [

            0.25,

            0.085,

            0.50,

            0.035

        ]

    )


    threshold_slider = Slider(

        ax=slider_axis,

        label="Threshold",

        valmin=0.00,

        valmax=1.00,

        valinit=THRESHOLD,

        valstep=0.01

    )


    # ========================================================
    # PREVIOUS BUTTON
    # ========================================================

    previous_axis = fig.add_axes(

        [

            0.25,

            0.025,

            0.12,

            0.045

        ]

    )


    # ========================================================
    # NEXT BUTTON
    # ========================================================

    next_axis = fig.add_axes(

        [

            0.63,

            0.025,

            0.12,

            0.045

        ]

    )


    from matplotlib.widgets import Button


    previous_button = Button(

        previous_axis,

        "Previous"

    )


    next_button = Button(

        next_axis,

        "Next"

    )


    # ========================================================
    # PATCH COUNTER
    # ========================================================

    counter_axis = fig.add_axes(

        [

            0.40,

            0.025,

            0.20,

            0.045

        ]

    )


    counter_axis.axis(
        "off"
    )


    counter_text = counter_axis.text(

        0.5,

        0.5,

        "",

        ha="center",

        va="center",

        fontsize=11,

        fontweight="bold"

    )


    # ========================================================
    # DISPLAY CURRENT PATCH
    # ========================================================

    def display_patch(
        patch_index
    ):

        # ----------------------------------------------------
        # Current patch
        # ----------------------------------------------------

        patch = all_patch_results[
            patch_index
        ]


        patch_id = patch[
            "patch_id"
        ]


        rgb_2023 = patch[
            "rgb_2023"
        ]


        rgb_2025 = patch[
            "rgb_2025"
        ]


        ground_truth = patch[
            "ground_truth"
        ]


        models = patch[
            "models"
        ]


        # ----------------------------------------------------
        # Clear all axes
        # ----------------------------------------------------

        for ax in axes:

            ax.clear()

            ax.set_xticks([])

            ax.set_yticks([])


        # ====================================================
        # SENTINEL-2 2023
        # ====================================================

        axes[0].imshow(

            rgb_2023,

            interpolation="nearest"

        )


        axes[0].set_title(

            "Sentinel-2 2023",

            fontsize=13,

            fontweight="bold",

            pad=10

        )


        # ====================================================
        # SENTINEL-2 2025
        # ====================================================

        axes[1].imshow(

            rgb_2025,

            interpolation="nearest"

        )


        axes[1].set_title(

            "Sentinel-2 2025",

            fontsize=13,

            fontweight="bold",

            pad=10

        )


        # ====================================================
        # GROUND TRUTH
        # ====================================================

        axes[2].imshow(

            ground_truth,

            cmap="gray",

            vmin=0,

            vmax=1,

            interpolation="nearest"

        )


        axes[2].set_title(

            "Ground Truth",

            fontsize=13,

            fontweight="bold",

            pad=10

        )


        # ====================================================
        # MODEL PREDICTIONS
        # ====================================================

        threshold = (

            threshold_slider.val

        )


        for model_index, model_result in enumerate(

            models

        ):


            axis_index = (

                3

                +

                model_index

            )


            ax = axes[
                axis_index
            ]


            probability = model_result[
                "probability"
            ]


            # ------------------------------------------------
            # Convert probability to prediction
            # ------------------------------------------------

            prediction = (

                probability

                >=

                threshold

            ).astype(

                np.uint8

            )


            # ------------------------------------------------
            # Display prediction
            # ------------------------------------------------

            ax.imshow(

                prediction,

                cmap="gray",

                vmin=0,

                vmax=1,

                interpolation="nearest"

            )


            # ------------------------------------------------
            # Model information
            # ------------------------------------------------

            strategy = strategy_display_name(

                model_result[
                    "strategy"
                ]

            )


            data_name = data_display_name(

                model_result[
                    "model_name"
                ]

            )


            ax.set_title(

                (

                    f"{strategy}\n"

                    f"Trained on {data_name}"

                ),

                fontsize=11,

                fontweight="bold",

                pad=10

            )


        # ====================================================
        # HIDE UNUSED AXES
        # ====================================================

        for index in range(

            total_images,

            len(axes)

        ):

            axes[index].axis(
                "off"
            )


        # ====================================================
        # SYNCHRONIZED SPATIAL EXTENT
        # ====================================================

        image_height = (

            rgb_2023.shape[0]

        )


        image_width = (

            rgb_2023.shape[1]

        )


        for index in range(

            total_images

        ):

            axes[index].set_xlim(

                0,

                image_width

            )


            axes[index].set_ylim(

                image_height,

                0

            )


            axes[index].set_aspect(

                "equal",

                adjustable="box"

            )


            axes[index].set_xticks(
                []
            )

            axes[index].set_yticks(
                []
            )


        # ====================================================
        # UPDATE COUNTER
        # ====================================================

        counter_text.set_text(

            (

                f"Patch "

                f"{patch_index + 1}"

                f" / "

                f"{len(all_patch_results)}"

            )

        )


        # ====================================================
        # UPDATE BUTTON STATES
        # ====================================================

        if patch_index == 0:

            previous_button.ax.set_alpha(
                0.5
            )

        else:

            previous_button.ax.set_alpha(
                1.0
            )


        if patch_index == len(all_patch_results) - 1:

            next_button.ax.set_alpha(
                0.5
            )

        else:

            next_button.ax.set_alpha(
                1.0
            )


        # ====================================================
        # MAIN TITLE
        # ====================================================

        fig.suptitle(

            (

                "TEST PATCH COMPARISON\n"

                f"Patch ID: {patch_id}\n"

                f"Threshold: {threshold:.2f}"

            ),

            fontsize=16,

            fontweight="bold"

        )


        # ====================================================
        # SAVE CURRENT FIGURE
        # ====================================================

        output_path = os.path.join(

            COMPARISON_DIR,

            f"{patch_id}_ALL_MODELS.png"

        )


        fig.savefig(

            output_path,

            dpi=300,

            bbox_inches="tight"

        )


        # ====================================================
        # REDRAW
        # ====================================================

        fig.canvas.draw_idle()


    # ========================================================
    # NEXT BUTTON FUNCTION
    # ========================================================

    def next_patch(
        event
    ):

        nonlocal current_index


        if current_index < len(

            all_patch_results

        ) - 1:

            current_index += 1

            display_patch(

                current_index

            )


    # ========================================================
    # PREVIOUS BUTTON FUNCTION
    # ========================================================

    def previous_patch(
        event
    ):

        nonlocal current_index


        if current_index > 0:

            current_index -= 1

            display_patch(

                current_index

            )


    # ========================================================
    # THRESHOLD UPDATE
    # ========================================================

    def update_threshold(
        value
    ):

        # ----------------------------------------------------
        # Get current patch
        # ----------------------------------------------------

        patch = all_patch_results[
            current_index
        ]


        threshold = (

            threshold_slider.val

        )


        # ----------------------------------------------------
        # Update model prediction images
        #
        # The easiest and safest approach is to redraw
        # the current patch.
        # ----------------------------------------------------

        display_patch(

            current_index

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


    threshold_slider.on_changed(

        update_threshold

    )


    # ========================================================
    # INITIAL DISPLAY
    # ========================================================

    display_patch(

        current_index

    )


    # ========================================================
    # SHOW INTERACTIVE WINDOW
    # ========================================================

    plt.show()


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION NAME
#
# If your MAIN section currently calls:
#
#     create_all_test_comparisons(...)
#
# this function now starts the interactive viewer.
# ============================================================

def create_all_test_comparisons(

    test_df,

    prediction_results_by_model

):

    create_test_comparison_viewer(

        test_df=test_df,

        prediction_results_by_model=(
            prediction_results_by_model
        )

    )
# ============================================================
# BUILD PREDICTION PATH LIST
# ============================================================

def build_prediction_path_list(

    model_info,

    csv_df,

    split,

    output_root

):


    model_name = model_output_name(

        model_info

    )


    model_directory = os.path.join(

        output_root,

        model_name,

        "prediction"

    )


    paths = []


    for index in range(

        len(csv_df)

    ):


        patch_id = str(

            csv_df.iloc[
                index
            ][
                "patch_id"
            ]

        )


        prediction_path = os.path.join(

            model_directory,

            f"{patch_id}_prediction.tif"

        )


        paths.append(

            prediction_path

        )


    return paths


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":


    print("\n")


    print("=" * 80)

    print(
        "STARTING MELBOURNE ALL-MODEL PREDICTION"
    )

    print("=" * 80)


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
    # FIND ALL MODELS
    # ========================================================

    model_infos = (

        find_all_best_models()

    )


    # ========================================================
    # CREATE DATA LOADERS
    # ========================================================

    print("\n" + "=" * 80)

    print(
        "CREATING DATA LOADERS"
    )

    print("=" * 80)


    from melbourne_dataset import (

        create_dataloaders

    )


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

    # --------------------------------------------------------
    # Early fusion
    # --------------------------------------------------------

    early_train_dataset = (

        train_loader.dataset

    )


    early_val_dataset = (

        val_loader.dataset

    )


    early_test_dataset = (

        test_loader.dataset

    )


    # --------------------------------------------------------
    # Dual stream
    # --------------------------------------------------------

    dual_train_dataset = (

        dual_train_loader.dataset

    )


    dual_val_dataset = (

        dual_val_loader.dataset

    )


    dual_test_dataset = (

        dual_test_loader.dataset

    )


    # ========================================================
    # CHECK DATASET LENGTHS
    # ========================================================

    if len(

        early_train_dataset

    ) != len(

        train_df

    ):

        raise ValueError(

            "Early fusion train dataset "
            "does not match train CSV."

        )


    if len(

        early_val_dataset

    ) != len(

        val_df

    ):

        raise ValueError(

            "Early fusion validation dataset "
            "does not match validation CSV."

        )


    if len(

        early_test_dataset

    ) != len(

        test_df

    ):

        raise ValueError(

            "Early fusion test dataset "
            "does not match test CSV."

        )


    if len(

        dual_train_dataset

    ) != len(

        train_df

    ):

        raise ValueError(

            "Dual stream train dataset "
            "does not match train CSV."

        )


    if len(

        dual_val_dataset

    ) != len(

        val_df

    ):

        raise ValueError(

            "Dual stream validation dataset "
            "does not match validation CSV."

        )


    if len(

        dual_test_dataset

    ) != len(

        test_df

    ):

        raise ValueError(

            "Dual stream test dataset "
            "does not match test CSV."

        )


    print(

        "\nDataset / CSV lengths verified."

    )


    # ========================================================
    # STORAGE FOR TEST PREDICTIONS
    #
    # We keep:
    #
    #   model_info
    #   prediction paths
    #
    # This allows comparison figures to know the strategy.
    # ========================================================

    test_prediction_results_by_model = {}


    # ========================================================
    # PROCESS EVERY BEST MODEL
    # ========================================================

    for (

        model_number,

        model_info

    ) in enumerate(

        model_infos,

        start=1

    ):


        print("\n")


        print("#" * 80)


        print(

            f"MODEL {model_number} / "

            f"{len(model_infos)}"

        )


        print(

            f"Checkpoint: "

            f"{model_info['filename']}"

        )


        print(

            f"Strategy: "

            f"{model_info['strategy']}"

        )


        print("#" * 80)


        # ====================================================
        # LOAD MODEL
        # ====================================================

        model = load_model(

            model_info

        )


        # ====================================================
        # SELECT DATASET
        # ====================================================

        if (

            model_info["strategy"]

            ==

            "early_fusion"

        ):


            train_dataset = (

                early_train_dataset

            )


            val_dataset = (

                early_val_dataset

            )


            test_dataset = (

                early_test_dataset

            )


        elif (

            model_info["strategy"]

            ==

            "dual_stream"

        ):


            train_dataset = (

                dual_train_dataset

            )


            val_dataset = (

                dual_val_dataset

            )


            test_dataset = (

                dual_test_dataset

            )


        else:

            raise ValueError(

                "Unsupported strategy."

            )


        # ====================================================
        # TRAIN
        # ====================================================

        predict_dataset(

            model_info=model_info,

            model=model,

            dataset=train_dataset,

            csv_df=train_df,

            split="train",

            output_root=TRAIN_OUTPUT_DIR

        )


        # ====================================================
        # VALIDATION
        # ====================================================

        predict_dataset(

            model_info=model_info,

            model=model,

            dataset=val_dataset,

            csv_df=val_df,

            split="validation",

            output_root=VAL_OUTPUT_DIR

        )


        # ====================================================
        # TEST
        # ====================================================

        predict_dataset(

            model_info=model_info,

            model=model,

            dataset=test_dataset,

            csv_df=test_df,

            split="test",

            output_root=TEST_OUTPUT_DIR

        )


        # ====================================================
        # SAVE TEST PREDICTION PATHS
        #
        # IMPORTANT:
        #
        # Keep model_info together with paths.
        # ====================================================

        model_name = model_output_name(

            model_info

        )


        test_prediction_results_by_model[

            model_name

        ] = {

            "model_info":
                model_info,

            "paths":
                build_prediction_path_list(

                    model_info=model_info,

                    csv_df=test_df,

                    split="test",

                    output_root=TEST_OUTPUT_DIR

                )

        }


        # ====================================================
        # FREE GPU MEMORY
        # ====================================================

        del model


        if torch.cuda.is_available():

            torch.cuda.empty_cache()


    # ========================================================
    # CREATE TEST COMPARISON
    # ========================================================

    if SAVE_TEST_COMPARISON_FIGURES:

        create_all_test_comparisons(

            test_df=test_df,

            prediction_results_by_model=(

                test_prediction_results_by_model

            )

        )


