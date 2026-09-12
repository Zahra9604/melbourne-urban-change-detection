"""
============================================================
MELBOURNE SENTINEL-1 / SENTINEL-2 CHANGE DETECTION
TRAINING SCRIPT
============================================================

Files:

    melbourne_dataset.py
    models.py
    training_config.py
    train.py


============================================================
SUPPORTED STRATEGIES
============================================================

1. EARLY FUSION

    Model:
        ContrastPyramidEarlyFusionUNet

    Input:
        [B, 12, H, W]

    Channel order:

        S1_2023 = 2
        S2_2023 = 4
        S1_2025 = 2
        S2_2025 = 4


2. DUAL STREAM

    Model:
        CrossModalTemporalGateUNet

    Sentinel-1:

        [B, 4, H, W]

        S1_2023 = 2
        S1_2025 = 2

    Sentinel-2:

        [B, 8, H, W]

        S2_2023 = 4
        S2_2025 = 4


============================================================
TRAINING WORKFLOWS
============================================================

WORKFLOW 1
----------

TRAINING_MODE = "resume_melbourne"

Continue a previously trained Melbourne model.

Restores:

    - model weights
    - optimizer state
    - scheduler state
    - AMP scaler state
    - epoch
    - best validation F1
    - early stopping counter

The NEW experiment is saved under MODEL_RUN_NAME.

Example:

    Previous:

        last_dual_stream_melbourne_v1.pth

    New:

        last_dual_stream_melbourne_v2.pth
        best_dual_stream_melbourne_v2.pth


============================================================

WORKFLOW 2
----------

TRAINING_MODE = "pretrained_oscd"

Use an OSCD pretrained model to initialize Melbourne training.

Loads ONLY:

    - model weights

Does NOT load:

    - optimizer
    - scheduler
    - AMP scaler
    - epoch
    - best validation F1
    - early stopping

Training starts from:

    epoch = 1

A new Melbourne optimizer/scheduler/scaler is created.

Example:

    OSCD:

        best_dual_stream_oscd.pth

    Melbourne:

        best_dual_stream_melbourne_oscd_finetune.pth
        last_dual_stream_melbourne_oscd_finetune.pth


============================================================

WORKFLOW 3
----------

TRAINING_MODE = "scratch"

Train Melbourne from randomly initialized weights.


============================================================
"""

# ============================================================
# IMPORTS
# ============================================================

import os
import csv
import time
import shutil
from tqdm import tqdm
import torch
import torch.nn as nn
from melbourne_dataset import create_dataloaders
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



# ============================================================
# PROJECT DIRECTORIES
# ============================================================
root = os.getcwd()
PROJECT_ROOT = os.path.dirname(root)

BASE_DIR = os.path.join(
    PROJECT_ROOT,
    "MelbourneData"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

CHECKPOINT_DIR = os.path.join(
    BASE_DIR,
    "checkpoints"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)



# ============================================================
# MODEL NAME
# ============================================================

if config.STRATEGY == "early_fusion":

    MODEL_NAME = (
        "ContrastPyramidEarlyFusionUNet"
    )

elif config.STRATEGY == "dual_stream":

    MODEL_NAME = (
        "CrossModalTemporalGateUNet"
    )

else:

    raise ValueError(
        "\nInvalid STRATEGY.\n"
        "\n"
        "Use:\n"
        "    'early_fusion'\n"
        "or:\n"
        "    'dual_stream'\n"
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

BEST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    f"best_{config.STRATEGY}_"
    f"{config.MODEL_RUN_NAME}.pth"
)


LAST_MODEL_PATH = os.path.join(
    CHECKPOINT_DIR,
    f"last_{config.STRATEGY}_{config.MODEL_RUN_NAME}.pth"
)


HISTORY_PATH = os.path.join(
    RESULTS_DIR,
    f"{config.STRATEGY}_"
    f"{config.MODEL_RUN_NAME}_history.csv"
)


# ============================================================
# DICE LOSS
# ============================================================

class DiceLoss(nn.Module):

    def __init__(
        self,
        smooth=1.0
    ):

        super().__init__()

        self.smooth = smooth


    def forward(
        self,
        logits,
        targets
    ):

        probabilities = torch.sigmoid(
            logits
        )

        probabilities = (
            probabilities.contiguous()
        )

        targets = (
            targets.contiguous()
        )

        intersection = (
            probabilities * targets
        ).sum(
            dim=(1, 2, 3)
        )

        denominator = (
            probabilities.sum(
                dim=(1, 2, 3)
            )
            +
            targets.sum(
                dim=(1, 2, 3)
            )
        )

        dice = (
            2.0 * intersection
            +
            self.smooth
        ) / (
            denominator
            +
            self.smooth
        )

        return (
            1.0 - dice
        ).mean()


# ============================================================
# BCE + DICE LOSS
# ============================================================

class BCEDiceLoss(nn.Module):

    def __init__(
        self,
        bce_weight=0.5,
        dice_weight=0.5
    ):

        super().__init__()

        self.bce_weight = (
            bce_weight
        )

        self.dice_weight = (
            dice_weight
        )

        self.bce = (
            nn.BCEWithLogitsLoss()
        )

        self.dice = DiceLoss()


    def forward(
        self,
        logits,
        targets
    ):

        bce_loss = self.bce(
            logits,
            targets
        )

        dice_loss = self.dice(
            logits,
            targets
        )

        total_loss = (
            self.bce_weight
            * bce_loss
            +
            self.dice_weight
            * dice_loss
        )

        return total_loss


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    logits,
    targets,
    threshold=0.5
):

    probabilities = torch.sigmoid(
        logits
    )

    predictions = (
        probabilities >= threshold
    ).float()

    targets = (
        targets.float()
    )

    predictions = (
        predictions.view(-1)
    )

    targets = (
        targets.view(-1)
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    tp = (
        (predictions == 1)
        &
        (targets == 1)
    ).sum().float()

    tn = (
        (predictions == 0)
        &
        (targets == 0)
    ).sum().float()

    fp = (
        (predictions == 1)
        &
        (targets == 0)
    ).sum().float()

    fn = (
        (predictions == 0)
        &
        (targets == 1)
    ).sum().float()

    eps = 1e-7

    # --------------------------------------------------------
    # PRECISION
    # --------------------------------------------------------

    precision = (
        tp
        /
        (
            tp
            +
            fp
            +
            eps
        )
    )

    # --------------------------------------------------------
    # RECALL
    # --------------------------------------------------------

    recall = (
        tp
        /
        (
            tp
            +
            fn
            +
            eps
        )
    )

    # --------------------------------------------------------
    # F1
    # --------------------------------------------------------

    f1 = (
        2.0
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
            +
            eps
        )
    )

    # --------------------------------------------------------
    # IOU
    # --------------------------------------------------------

    iou = (
        tp
        /
        (
            tp
            +
            fp
            +
            fn
            +
            eps
        )
    )

    # --------------------------------------------------------
    # ACCURACY
    # --------------------------------------------------------

    accuracy = (
        (tp + tn)
        /
        (
            tp
            +
            tn
            +
            fp
            +
            fn
            +
            eps
        )
    )

    return {

        "accuracy":
            accuracy.item(),

        "precision":
            precision.item(),

        "recall":
            recall.item(),

        "f1":
            f1.item(),

        "iou":
            iou.item()
    }


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    print()
    print("=" * 80)
    print("MODEL")
    print("=" * 80)

    # ========================================================
    # EARLY FUSION
    # ========================================================

    if config.STRATEGY == "early_fusion":

        print(
            "Strategy: EARLY FUSION"
        )

        print(
            "Model: "
            "ContrastPyramidEarlyFusionUNet"
        )

        print(
            "Input: [B, 12, 256, 256]"
        )

        model = (
            ContrastPyramidEarlyFusionUNet(
                in_channels=12,
                base_channels=32,
                out_channels=1
            )
        )

    # ========================================================
    # DUAL STREAM
    # ========================================================

    elif config.STRATEGY == "dual_stream":

        print(
            "Strategy: DUAL STREAM"
        )

        print(
            "Model: "
            "CrossModalTemporalGateUNet"
        )

        print(
            "Sentinel-1: "
            "[B, 4, 256, 256]"
        )

        print(
            "Sentinel-2: "
            "[B, 8, 256, 256]"
        )

        model = (
            CrossModalTemporalGateUNet(
                s1_in_channels=4,
                s2_in_channels=8,
                base_channels=32,
                out_channels=1
            )
        )

    else:

        raise ValueError(
            f"Unknown strategy: "
            f"{config.STRATEGY}"
        )

    model = model.to(
        DEVICE
    )

    print()

    print(
        "Device:",
        DEVICE
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Total parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters: "
        f"{trainable_parameters:,}"
    )

    return model


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    scaler
):

    model.train()

    running_loss = 0.0

    metric_sum = {

        "accuracy": 0.0,

        "precision": 0.0,

        "recall": 0.0,

        "f1": 0.0,

        "iou": 0.0
    }

    num_batches = len(
        loader
    )
    with tqdm(total=len(loader), desc='Training', unit='batch') as pbar:
        for batch_idx, batch in enumerate(
            loader
        ):

            optimizer.zero_grad(
                set_to_none=True # memory-efficient and potentially faster
            )

            # ====================================================
            # INPUT
            # ====================================================

            if config.STRATEGY == "early_fusion":

                image = (
                    batch["image"]
                    .to(
                        DEVICE,
                        non_blocking=True # CPU → GPU
                    )
                )

                label = (
                    batch["label"]
                    .to(
                        DEVICE,
                        non_blocking=True 
                    )
                )

            else:

                sentinel1 = (
                    batch["sentinel1"]
                    .to(
                        DEVICE,
                        non_blocking=True 
                    )
                )

                sentinel2 = (
                    batch["sentinel2"]
                    .to(
                        DEVICE,
                        non_blocking=True 
                    )
                )

                label = (
                    batch["label"]
                    .to(
                        DEVICE,
                        non_blocking=True 
                    )
                )

            # ====================================================
            # FORWARD
            # ====================================================
            # USE_AMP is a boolean flag that indicates whether to use Automatic Mixed Precision (AMP) for training. AMP allows for faster training and reduced memory usage by using mixed precision (a combination of 16-bit and 32-bit floating point numbers) during the forward and backward passes of the model.
            if config.USE_AMP:

                with torch.amp.autocast(

                    device_type="cuda",

                    enabled=(
                        DEVICE.type == "cuda"
                    )
                ):

                    if (
                        config.STRATEGY
                        ==
                        "early_fusion"
                    ):

                        logits = model(
                            image
                        )

                    else:

                        logits = model(
                            sentinel1,
                            sentinel2
                        )

                    loss = criterion(
                        logits,
                        label
                    )

            else:

                if (
                    config.STRATEGY
                    ==
                    "early_fusion"
                ):

                    logits = model(
                        image
                    )

                else:

                    logits = model(
                        sentinel1,
                        sentinel2
                    )

                loss = criterion(
                    logits,
                    label
                )

            # ====================================================
            # BACKWARD
            # ====================================================

            if config.USE_AMP:

                scaler.scale(
                    loss
                ).backward()

                if (
                    config.GRADIENT_CLIPPING
                    is not None
                ):

                    scaler.unscale_(
                        optimizer
                    )

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.GRADIENT_CLIPPING
                    )

                scaler.step(
                    optimizer
                )

                scaler.update()

            else:

                loss.backward()

                if (
                    config.GRADIENT_CLIPPING
                    is not None
                ):

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        config.GRADIENT_CLIPPING
                    )

                optimizer.step()

            # ====================================================
            # LOSS
            # ====================================================

            running_loss += (
                loss.item()
            )

            # ====================================================
            # METRICS
            # ====================================================

            metrics = calculate_metrics(
                logits.detach(),
                label
            )

            for key in metric_sum:

                metric_sum[key] += (
                    metrics[key]
                )

            # ====================================================
            # PROGRESS
            # ====================================================

            if (
                batch_idx + 1
            ) % 10 == 0:

                print(
                    f"\r"
                    f"Train "
                    f"[{batch_idx + 1}/"
                    f"{num_batches}] "
                    f"Loss: "
                    f"{loss.item():.4f}",
                    end=""
                )

        print()

        average_loss = (
            running_loss
            /
            num_batches
        )

        average_metrics = {

            key:
                value / num_batches

            for key, value
            in metric_sum.items()
        }

        return (
            average_loss,
            average_metrics
        )


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()  #not to calculate/store gradients.
def validate(
    model,
    loader,
    criterion
):

    model.eval() # Dropout / BatchNorm to evaluation behavior.

    running_loss = 0.0

    metric_sum = {

        "accuracy": 0.0,

        "precision": 0.0,

        "recall": 0.0,

        "f1": 0.0,

        "iou": 0.0
    }

    num_batches = len(
        loader
    )

    for batch in loader:

        # ====================================================
        # INPUT
        # ====================================================

        if config.STRATEGY == "early_fusion":

            image = (
                batch["image"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            label = (
                batch["label"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

        else:

            sentinel1 = (
                batch["sentinel1"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            sentinel2 = (
                batch["sentinel2"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            label = (
                batch["label"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

        # ====================================================
        # FORWARD
        # ====================================================

        if config.STRATEGY == "early_fusion":

            logits = model(
                image
            )

        else:

            logits = model(
                sentinel1,
                sentinel2
            )

        # ====================================================
        # LOSS
        # ====================================================

        loss = criterion(
            logits,
            label
        )

        running_loss += (
            loss.item()
        )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = calculate_metrics(
            logits,
            label
        )

        for key in metric_sum:

            metric_sum[key] += (
                metrics[key]
            )

    average_loss = (
        running_loss
        /
        num_batches
    )

    average_metrics = {

        key:
            value / num_batches

        for key, value
        in metric_sum.items()
    }

    return (
        average_loss,
        average_metrics
    )


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_f1,
    epochs_without_improvement
):

    checkpoint = {

        # ----------------------------------------------------
        # TRAINING INFORMATION
        # ----------------------------------------------------

        "epoch":
            epoch,

        "strategy":
            config.STRATEGY,

        "training_mode":
            config.TRAINING_MODE,

        "model_run_name":
            config.MODEL_RUN_NAME,

        "model_name":
            MODEL_NAME,

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        "model_state_dict":
            model.state_dict(),

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        "optimizer_state_dict":
            optimizer.state_dict(),

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------
        # learning-rate scheduler
        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        # ----------------------------------------------------
        # AMP
        # ----------------------------------------------------

        "scaler_state_dict":
            scaler.state_dict()
            if scaler is not None
            else None,

        # ----------------------------------------------------
        # TRAINING STATUS
        # ----------------------------------------------------

        "best_val_f1":
            best_val_f1,

        "epochs_without_improvement":
            epochs_without_improvement
    }

    torch.save(
        checkpoint,
        path
    )


# ============================================================
# LOAD FULL MELBOURNE CHECKPOINT
# ============================================================

def load_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler
):

    print()
    print("=" * 80)
    print("LOADING MELBOURNE CHECKPOINT")
    print("=" * 80)

    print(
        f"Checkpoint: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False # Load the entire saved checkpoint
    )

    # ========================================================
    # FULL CHECKPOINT
    # ========================================================

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict"
        in checkpoint
    ):

        # ----------------------------------------------------
        # MODEL
        # ----------------------------------------------------

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        if (
            "optimizer_state_dict"
            in checkpoint
            and
            checkpoint[
                "optimizer_state_dict"
            ]
            is not None
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        if (
            scheduler is not None
            and
            checkpoint.get(
                "scheduler_state_dict"
            )
            is not None
        ):

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        # ----------------------------------------------------
        # AMP SCALER
        # ----------------------------------------------------

        if (
            scaler is not None
            and
            checkpoint.get(
                "scaler_state_dict"
            )
            is not None
        ):

            scaler.load_state_dict(
                checkpoint[
                    "scaler_state_dict"
                ]
            )

        # ----------------------------------------------------
        # EPOCH
        # ----------------------------------------------------

        checkpoint_epoch = checkpoint.get(
            "epoch",
            0
        )

        start_epoch = (
            checkpoint_epoch + 1
        )

        # ----------------------------------------------------
        # BEST F1
        # ----------------------------------------------------

        best_val_f1 = checkpoint.get(
            "best_val_f1",
            -1.0
        )

        # ----------------------------------------------------
        # EARLY STOPPING
        # ----------------------------------------------------

        epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0
            )
        )

        print()
        print(
            "✓ Full checkpoint loaded."
        )

        print(
            f"Checkpoint epoch: "
            f"{checkpoint_epoch}"
        )

        print(
            f"Resuming from epoch: "
            f"{start_epoch}"
        )

        print(
            f"Best validation F1: "
            f"{best_val_f1:.4f}"
        )

        print(
            f"Early stopping counter: "
            f"{epochs_without_improvement}"
        )

        return (
            start_epoch,
            best_val_f1,
            epochs_without_improvement
        )

    # ========================================================
    # MODEL-ONLY CHECKPOINT
    # ========================================================

    else:

        model.load_state_dict(
            checkpoint
        )

        print()
        print(
            "Model-only checkpoint detected."
        )

        print(
            "Only model weights were loaded."
        )

        print(
            "Optimizer/scheduler cannot be restored."
        )

        return (
            1,
            -1.0,
            0
        )


# ============================================================
# LOAD OSCD PRETRAINED MODEL
# ============================================================

def load_pretrained_model(
    path,
    model
):

    print()
    print("=" * 80)
    print("LOADING OSCD PRETRAINED MODEL")
    print("=" * 80)

    print(
        f"Pretrained model: {path}"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE,
        weights_only=False
    )

    # ========================================================
    # FULL CHECKPOINT
    # ========================================================

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict"
        in checkpoint
    ):

        state_dict = checkpoint[
            "model_state_dict"
        ]

        print()
        print(
            "Detected full OSCD checkpoint."
        )

        print(
            "Loading MODEL WEIGHTS ONLY."
        )

    # ========================================================
    # MODEL-ONLY CHECKPOINT
    # ========================================================

    else:

        state_dict = checkpoint

        print()
        print(
            "Detected model-only OSCD checkpoint."
        )

        print(
            "Loading MODEL WEIGHTS."
        )

    # ========================================================
    # LOAD WEIGHTS
    # ========================================================

    model.load_state_dict(
        state_dict,
        strict=True
    )

    print()
    print(
        "✓ OSCD model weights loaded successfully."
    )

    print()
    print(
        "Optimizer: RESET"
    )

    print(
        "Scheduler: RESET"
    )

    print(
        "AMP scaler: RESET"
    )

    print(
        "Epoch: RESET TO 1"
    )

    print(
        "Best validation F1: RESET"
    )

    print(
        "Early stopping counter: RESET"
    )


# ============================================================
# SAVE HISTORY
# ============================================================

def save_history(
    history,
    path=None
):

    if path is None:

        path = HISTORY_PATH

    fieldnames = [

        "epoch",

        "train_loss",
        "val_loss",

        "train_accuracy",
        "val_accuracy",

        "train_precision",
        "val_precision",

        "train_recall",
        "val_recall",

        "train_f1",
        "val_f1",

        "train_iou",
        "val_iou",

        "learning_rate"
    ]

    with open(
        path,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            history
        )


# ============================================================
# LOAD HISTORY
# ============================================================

def load_history(
    path=None
):

    if path is None:

        path = HISTORY_PATH

    if not os.path.exists(
        path
    ):

        return []

    history = []

    with open(
        path,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            history.append({

                "epoch":
                    int(row["epoch"]),

                "train_loss":
                    float(row["train_loss"]),

                "val_loss":
                    float(row["val_loss"]),

                "train_accuracy":
                    float(row["train_accuracy"]),

                "val_accuracy":
                    float(row["val_accuracy"]),

                "train_precision":
                    float(row["train_precision"]),

                "val_precision":
                    float(row["val_precision"]),

                "train_recall":
                    float(row["train_recall"]),

                "val_recall":
                    float(row["val_recall"]),

                "train_f1":
                    float(row["train_f1"]),

                "val_f1":
                    float(row["val_f1"]),

                "train_iou":
                    float(row["train_iou"]),

                "val_iou":
                    float(row["val_iou"]),

                "learning_rate":
                    float(row["learning_rate"])
            })

    return history


# ============================================================
# INITIALIZE BEST MODEL FOR RESUME
# ============================================================

def initialize_resume_best_model(
    resume_path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    best_val_f1,
    epochs_without_improvement
):

    """
    When resuming a Melbourne experiment under a NEW name,
    create the new BEST checkpoint immediately.

    This means:

        old model:
            last_melbourne_v1.pth

        new model:
            best_melbourne_v2.pth

    If no later epoch improves F1, the new best checkpoint
    still contains the model state from the beginning of
    the resumed run.
    """

    save_checkpoint(

        BEST_MODEL_PATH,

        model,

        optimizer,

        scheduler,

        scaler,

        epoch,

        best_val_f1,

        epochs_without_improvement
    )

    print()
    print(
        "✓ Initial best checkpoint created for "
        "the new experiment."
    )

    print(
        BEST_MODEL_PATH
    )


# ============================================================
# TEST MODEL
# ============================================================

@torch.no_grad()
def test_model(
    model,
    loader,
    criterion
):

    print()
    print("=" * 80)
    print("TESTING BEST MODEL")
    print("=" * 80)

    model.eval()

    running_loss = 0.0

    metric_sum = {

        "accuracy": 0.0,

        "precision": 0.0,

        "recall": 0.0,

        "f1": 0.0,

        "iou": 0.0
    }

    num_batches = len(
        loader
    )

    for batch in loader:

        # ====================================================
        # INPUT
        # ====================================================

        if config.STRATEGY == "early_fusion":

            image = (
                batch["image"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            label = (
                batch["label"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

        else:

            sentinel1 = (
                batch["sentinel1"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            sentinel2 = (
                batch["sentinel2"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

            label = (
                batch["label"]
                .to(
                    DEVICE,
                    non_blocking=True
                )
            )

        # ====================================================
        # FORWARD
        # ====================================================

        if config.STRATEGY == "early_fusion":

            logits = model(
                image
            )

        else:

            logits = model(
                sentinel1,
                sentinel2
            )

        # ====================================================
        # LOSS
        # ====================================================

        loss = criterion(
            logits,
            label
        )

        running_loss += (
            loss.item()
        )

        # ====================================================
        # METRICS
        # ====================================================

        metrics = calculate_metrics(
            logits,
            label
        )

        for key in metric_sum:

            metric_sum[key] += (
                metrics[key]
            )

    test_loss = (
        running_loss
        /
        num_batches
    )

    test_metrics = {

        key:
            value / num_batches

        for key, value
        in metric_sum.items()
    }

    print()

    print(
        f"Test Loss:      "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy:  "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall:    "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test F1:        "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test IoU:       "
        f"{test_metrics['iou']:.4f}"
    )

    return (
        test_loss,
        test_metrics
    )


# ============================================================
# TRAIN
# ============================================================

def train():

    print()
    print("=" * 80)
    print("MELBOURNE CHANGE DETECTION")
    print("TRAINING")
    print("=" * 80)

    print()

    print(
        "Strategy:",
        config.STRATEGY
    )

    print(
        "Training mode:",
        config.TRAINING_MODE
    )

    print(
        "Model run name:",
        config.MODEL_RUN_NAME
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Batch size:",
        config.BATCH_SIZE
    )

    print(
        "Epochs:",
        config.EPOCHS
    )

    print(
        "Learning rate:",
        config.LEARNING_RATE
    )

    print(
        "Normalization:",
        config.NORMALIZE
    )

    print()

    print(
        "Output best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Output last checkpoint:"
    )

    print(
        LAST_MODEL_PATH
    )

    print()

    print(
        "Output history:"
    )

    print(
        HISTORY_PATH
    )


    # ========================================================
    # DATA
    # ========================================================

    print()
    print("=" * 80)
    print("CREATING DATA LOADERS")
    print("=" * 80)

    (
        early_train_loader,
        early_val_loader,
        early_test_loader,

        dual_train_loader,
        dual_val_loader,
        dual_test_loader

    ) = create_dataloaders(

        batch_size=config.BATCH_SIZE,

        num_workers=config.NUM_WORKERS,

        normalize=config.NORMALIZE
    )


    # ========================================================
    # SELECT LOADER
    # ========================================================

    if config.STRATEGY == "early_fusion":

        train_loader = (
            early_train_loader
        )

        val_loader = (
            early_val_loader
        )

        test_loader = (
            early_test_loader
        )

    else:

        train_loader = (
            dual_train_loader
        )

        val_loader = (
            dual_val_loader
        )

        test_loader = (
            dual_test_loader
        )


    print()

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(val_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )


    # ========================================================
    # MODEL
    # ========================================================

    model = create_model()


    # ========================================================
    # LOSS
    # ========================================================

    criterion = BCEDiceLoss(

        bce_weight=config.BCE_WEIGHT,

        dice_weight=config.DICE_WEIGHT
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.LEARNING_RATE,

        weight_decay=config.WEIGHT_DECAY
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=5
    )


    # ========================================================
    # AMP
    # ========================================================

    scaler = torch.amp.GradScaler(

        "cuda",

        enabled=(

            config.USE_AMP

            and

            DEVICE.type == "cuda"
        )
    )


    # ========================================================
    # INITIAL TRAINING STATE
    # ========================================================

    start_epoch = 1

    best_val_f1 = -1.0

    epochs_without_improvement = 0

    history = []


    # ========================================================
    # TRAINING WORKFLOW
    # ========================================================

    if config.TRAINING_MODE == "resume_melbourne":

        # ====================================================
        # WORKFLOW 1
        #
        # MELBOURNE → MELBOURNE
        # ====================================================

        print()
        print("=" * 80)
        print("WORKFLOW 1")
        print("RESUME MELBOURNE TRAINING")
        print("=" * 80)

        resume_path = (
            config.RESUME_CHECKPOINT_PATH
        )

        if not os.path.exists(
            resume_path
        ):

            raise FileNotFoundError(

                "\nMelbourne resume checkpoint "
                "was not found:\n\n"

                f"{resume_path}\n"
            )

        # ----------------------------------------------------
        # LOAD FULL CHECKPOINT
        # ----------------------------------------------------

        (
            start_epoch,

            best_val_f1,

            epochs_without_improvement

        ) = load_checkpoint(

            resume_path,

            model,

            optimizer,

            scheduler,

            scaler
        )

        # ----------------------------------------------------
        # LOAD OLD HISTORY
        # ----------------------------------------------------

        old_history_path = (
            config.RESUME_HISTORY_PATH
        )

        if os.path.exists(
            old_history_path
        ):

            print()
            print(
                "Loading previous history:"
            )

            print(
                old_history_path
            )

            history = load_history(
                old_history_path
            )

            # ------------------------------------------------
            # Remove history after checkpoint epoch
            # ------------------------------------------------

            history = [

                row

                for row in history

                if row["epoch"]
                <
                start_epoch
            ]

            print()

            print(
                f"Previous history rows loaded: "
                f"{len(history)}"
            )

        else:

            print()
            print(
                "WARNING:"
            )

            print(
                "Previous history file was not found."
            )

            print(
                "Starting new history file."
            )

            history = []

        # ----------------------------------------------------
        # SAVE HISTORY UNDER NEW EXPERIMENT NAME
        # ----------------------------------------------------

        save_history(
            history
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Create new best checkpoint.
        #
        # The previous Melbourne checkpoint is never
        # overwritten.
        # ----------------------------------------------------

        initialize_resume_best_model(

            resume_path,

            model,

            optimizer,

            scheduler,

            scaler,

            start_epoch - 1,

            best_val_f1,

            epochs_without_improvement
        )

        print()
        print(
            "✓ Previous Melbourne checkpoint remains untouched."
        )

        print(
            "✓ New Melbourne checkpoint will be created."
        )


    elif config.TRAINING_MODE == "pretrained_oscd":

        # ====================================================
        # WORKFLOW 2
        #
        # OSCD → MELBOURNE
        # ====================================================

        print()
        print("=" * 80)
        print("WORKFLOW 2")
        print("OSCD PRETRAINED → MELBOURNE")
        print("=" * 80)

        pretrained_path = (
            config.PRETRAINED_MODEL_PATH
        )

        if not os.path.exists(
            pretrained_path
        ):

            raise FileNotFoundError(

                "\nOSCD pretrained model "
                "was not found:\n\n"

                f"{pretrained_path}\n"
            )

        # ----------------------------------------------------
        # LOAD ONLY MODEL WEIGHTS
        # ----------------------------------------------------

        load_pretrained_model(

            pretrained_path,

            model
        )

        # ----------------------------------------------------
        # RESET TRAINING STATE
        # ----------------------------------------------------

        start_epoch = 1

        best_val_f1 = -1.0

        epochs_without_improvement = 0

        history = []

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        if os.path.exists(
            BEST_MODEL_PATH
        ):

            print()
            print(
                "WARNING:"
            )

            print(
                "The new BEST model already exists:"
            )

            print(
                BEST_MODEL_PATH
            )

            print()
            print(
                "It will be overwritten by this run."
            )

        print()
        print(
            "✓ OSCD weights loaded."
        )

        print(
            "✓ New Melbourne optimizer created."
        )

        print(
            "✓ New Melbourne scheduler created."
        )

        print(
            "✓ Training starts from epoch 1."
        )


    elif config.TRAINING_MODE == "scratch":

        # ====================================================
        # WORKFLOW 3
        #
        # MELBOURNE FROM SCRATCH
        # ====================================================

        print()
        print("=" * 80)
        print("WORKFLOW 3")
        print("TRAIN MELBOURNE FROM SCRATCH")
        print("=" * 80)

        start_epoch = 1

        best_val_f1 = -1.0

        epochs_without_improvement = 0

        history = []

        print()
        print(
            "✓ Model initialized randomly."
        )

        print(
            "✓ Training starts from epoch 1."
        )


    else:

        raise ValueError(

            "\nInvalid TRAINING_MODE:\n"

            f"    {config.TRAINING_MODE}\n\n"

            "Available options:\n"

            "    'scratch'\n"

            "    'resume_melbourne'\n"

            "    'pretrained_oscd'\n"
        )


    # ========================================================
    # TRAINING STATUS
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING STATUS")
    print("=" * 80)

    print(
        f"Start epoch: "
        f"{start_epoch}"
    )

    print(
        f"Total epochs: "
        f"{config.EPOCHS}"
    )

    print(
        f"Best validation F1: "
        f"{best_val_f1:.4f}"
    )

    print(
        f"Early stopping counter: "
        f"{epochs_without_improvement}"
    )


    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(

        start_epoch,

        config.EPOCHS + 1
    ):

        start_time = time.time()

        print()
        print("=" * 80)

        print(
            f"EPOCH "
            f"{epoch}/"
            f"{config.EPOCHS}"
        )

        print("=" * 80)


        # ====================================================
        # TRAIN
        # ====================================================

        (
            train_loss,
            train_metrics
        ) = train_one_epoch(

            model,

            train_loader,

            optimizer,

            criterion,

            scaler
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        (
            val_loss,
            val_metrics
        ) = validate(

            model,

            val_loader,

            criterion
        )


        # ====================================================
        # LEARNING RATE
        # ====================================================

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )


        # ====================================================
        # SCHEDULER
        # ====================================================

        scheduler.step(
            val_metrics["f1"]
        )


        # ====================================================
        # TIME
        # ====================================================

        epoch_time = (
            time.time()
            -
            start_time
        )


        # ====================================================
        # PRINT RESULTS
        # ====================================================

        print()

        print(
            f"Train Loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss:   "
            f"{val_loss:.4f}"
        )

        print()

        print(
            f"Train F1:   "
            f"{train_metrics['f1']:.4f}"
        )

        print(
            f"Val F1:     "
            f"{val_metrics['f1']:.4f}"
        )

        print()

        print(
            f"Train IoU:  "
            f"{train_metrics['iou']:.4f}"
        )

        print(
            f"Val IoU:    "
            f"{val_metrics['iou']:.4f}"
        )

        print()

        print(
            f"Precision:  "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"Recall:     "
            f"{val_metrics['recall']:.4f}"
        )

        print()

        print(
            f"Learning rate: "
            f"{current_lr:.2e}"
        )

        print(
            f"Time: "
            f"{epoch_time:.1f} sec"
        )


        # ====================================================
        # CHECK BEST MODEL
        # ====================================================

        is_best = (

            val_metrics["f1"]

            >

            best_val_f1
        )


        if is_best:

            best_val_f1 = (
                val_metrics["f1"]
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1


        # ====================================================
        # HISTORY
        # ====================================================

        history.append({

            "epoch":
                epoch,

            "train_loss":
                train_loss,

            "val_loss":
                val_loss,

            "train_accuracy":
                train_metrics["accuracy"],

            "val_accuracy":
                val_metrics["accuracy"],

            "train_precision":
                train_metrics["precision"],

            "val_precision":
                val_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "val_recall":
                val_metrics["recall"],

            "train_f1":
                train_metrics["f1"],

            "val_f1":
                val_metrics["f1"],

            "train_iou":
                train_metrics["iou"],

            "val_iou":
                val_metrics["iou"],

            "learning_rate":
                current_lr
        })


        save_history(
            history
        )


        # ====================================================
        # SAVE LAST CHECKPOINT
        # ====================================================

        save_checkpoint(

            LAST_MODEL_PATH,

            model,

            optimizer,

            scheduler,

            scaler,

            epoch,

            best_val_f1,

            epochs_without_improvement
        )

        print()
        print(
            "✓ Last checkpoint saved:"
        )

        print(
            LAST_MODEL_PATH
        )


        # ====================================================
        # SAVE BEST CHECKPOINT
        # ====================================================

        if is_best:

            save_checkpoint(

                BEST_MODEL_PATH,

                model,

                optimizer,

                scheduler,

                scaler,

                epoch,

                best_val_f1,

                epochs_without_improvement
            )

            print()
            print(
                "★ Best model saved!"
            )

            print(
                f"Best Val F1: "
                f"{best_val_f1:.4f}"
            )

            print(
                f"Path: "
                f"{BEST_MODEL_PATH}"
            )

        else:

            print()
            print(
                "No improvement: "
                f"{epochs_without_improvement}/"
                f"{config.EARLY_STOPPING_PATIENCE}"
            )


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            epochs_without_improvement
            >=
            config.EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping."
            )

            print(
                f"No validation F1 "
                f"improvement for "
                f"{epochs_without_improvement} "
                f"epochs."
            )

            break


    # ========================================================
    # CHECK BEST MODEL EXISTS
    # ========================================================

    if not os.path.exists(
        BEST_MODEL_PATH
    ):

        print()
        print(
            "ERROR: Best model checkpoint "
            "does not exist."
        )

        return


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()
    print("=" * 80)
    print("LOADING BEST MODEL")
    print("=" * 80)

    best_checkpoint = torch.load(

        BEST_MODEL_PATH,

        map_location=DEVICE,

        weights_only=False
    )


    if (
        isinstance(best_checkpoint, dict)
        and
        "model_state_dict"
        in best_checkpoint
    ):

        model.load_state_dict(

            best_checkpoint[
                "model_state_dict"
            ]
        )

        print()

        print(
            f"Best epoch: "
            f"{best_checkpoint['epoch']}"
        )

        print(
            f"Best validation F1: "
            f"{best_checkpoint['best_val_f1']:.4f}"
        )

    else:

        model.load_state_dict(
            best_checkpoint
        )

        print()

        print(
            "Loaded model-only checkpoint."
        )


    # ========================================================
    # TEST
    # ========================================================

    (
        test_loss,
        test_metrics
    ) = test_model(

        model,

        test_loader,

        criterion
    )


    # ========================================================
    # FINAL
    # ========================================================

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print()

    print(
        f"Strategy: "
        f"{config.STRATEGY}"
    )

    print(
        f"Training mode: "
        f"{config.TRAINING_MODE}"
    )

    print(
        f"Model run name: "
        f"{config.MODEL_RUN_NAME}"
    )

    print()

    print(
        "Best model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print()

    print(
        "Last checkpoint:"
    )

    print(
        LAST_MODEL_PATH
    )

    print()

    print(
        "History:"
    )

    print(
        HISTORY_PATH
    )

    print()

    print(
        f"Test Loss: "
        f"{test_loss:.4f}"
    )

    print(
        f"Test F1: "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"Test IoU: "
        f"{test_metrics['iou']:.4f}"
    )

    print(
        f"Test Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Test Recall: "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"Test Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print()
    print("=" * 80)
    print("DONE")
    print("=" * 80)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train()