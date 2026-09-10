# =============================================================================
# OSCD CHANGE DETECTION TRAINING
# =============================================================================
#
# Supports:
#
#   1. Early fusion
#   2. Dual stream
#
# Model selection is controlled entirely by training_config.py
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import os
import sys
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm


# =============================================================================
# PROJECT PATH
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# PROJECT MODULES
# =============================================================================

from oscd_dataset import OSCDataset
from preprocess import load_split_csv
from utils import collate_oscd_batch

from models import (
    ContrastPyramidEarlyFusionUNet,
    CrossModalTemporalGateUNet,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

import training_config as config


# =============================================================================
# DEVICE
# =============================================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# =============================================================================
# DIRECTORIES
# =============================================================================

WEIGHTS_DIR = r"C:\melbourne-urban-change-detection\Melbourne\MelbourneData\checkpoints"

os.makedirs(
    WEIGHTS_DIR,
    exist_ok=True
)




BEST_CHECKPOINT_PATH = os.path.join(
    WEIGHTS_DIR,
    config.BEST_CHECKPOINT_NAME
)


LAST_CHECKPOINT_PATH = os.path.join(
    WEIGHTS_DIR,
    config.LAST_CHECKPOINT_NAME
)


# =============================================================================
# RANDOM SEED
# =============================================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False

    torch.backends.cudnn.benchmark = True


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model():

    print()
    print("=" * 80)
    print("CREATING MODEL")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # EARLY FUSION
    # -------------------------------------------------------------------------

    if config.MODEL_TYPE == "early_fusion":

        print()
        print("Model type: EARLY FUSION")

        print(
            f"Input channels : "
            f"{config.EARLY_FUSION_CHANNELS}"
        )

        model = ContrastPyramidEarlyFusionUNet(
            in_channels=config.EARLY_FUSION_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            out_channels=config.OUTPUT_CHANNELS,
        )

    # -------------------------------------------------------------------------
    # DUAL STREAM
    # -------------------------------------------------------------------------

    elif config.MODEL_TYPE == "dual_stream":

        print()
        print("Model type: DUAL STREAM")

        print(
            f"S1 channels : "
            f"{config.S1_CHANNELS}"
        )

        print(
            f"S2 channels : "
            f"{config.S2_CHANNELS}"
        )

        model = CrossModalTemporalGateUNet(
            s1_in_channels=config.S1_CHANNELS,
            s2_in_channels=config.S2_CHANNELS,
            base_channels=config.BASE_CHANNELS,
            out_channels=config.OUTPUT_CHANNELS,
        )

    else:

        raise ValueError(
            "\nUnknown MODEL_TYPE: "
            f"{config.MODEL_TYPE}\n\n"
            "Valid options are:\n"
            "  early_fusion\n"
            "  dual_stream"
        )

    model = model.to(DEVICE)

    return model


# =============================================================================
# LOSS
# =============================================================================

class WeightedBCEDiceLoss(nn.Module):

    def __init__(
        self,
        pos_weight=3.0,
        dice_smooth=1.0
    ):

        super().__init__()

        self.dice_smooth = dice_smooth

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                [pos_weight],
                dtype=torch.float32
            )
        )

    def forward(
        self,
        logits,
        targets
    ):

        targets = targets.float()

        # ---------------------------------------------------------------------
        # BCE
        # ---------------------------------------------------------------------

        bce = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )

        # ---------------------------------------------------------------------
        # Dice
        # ---------------------------------------------------------------------

        probabilities = torch.sigmoid(
            logits
        )

        dims = (
            1,
            2,
            3
        )

        intersection = (
            probabilities * targets
        ).sum(
            dim=dims
        )

        denominator = (
            probabilities.sum(dim=dims)
            +
            targets.sum(dim=dims)
        )

        dice = (
            2.0 * intersection
            +
            self.dice_smooth
        ) / (
            denominator
            +
            self.dice_smooth
        )

        dice_loss = (
            1.0
            -
            dice.mean()
        )

        return bce + dice_loss


# =============================================================================
# METRICS
# =============================================================================

def calculate_confusion_matrix(
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
        targets >= 0.5
    ).float()

    predictions = predictions.reshape(-1)

    targets = targets.reshape(-1)

    tp = (
        (predictions == 1)
        &
        (targets == 1)
    ).sum().item()

    fp = (
        (predictions == 1)
        &
        (targets == 0)
    ).sum().item()

    fn = (
        (predictions == 0)
        &
        (targets == 1)
    ).sum().item()

    tn = (
        (predictions == 0)
        &
        (targets == 0)
    ).sum().item()

    return tp, fp, fn, tn


def metrics_from_confusion(
    tp,
    fp,
    fn,
    tn
):

    # -------------------------------------------------------------------------
    # IoU
    # -------------------------------------------------------------------------

    iou_denominator = (
        tp + fp + fn
    )

    if iou_denominator == 0:

        iou = 1.0

    else:

        iou = (
            tp
            /
            iou_denominator
        )

    # -------------------------------------------------------------------------
    # Dice
    # -------------------------------------------------------------------------

    dice_denominator = (
        2 * tp
        +
        fp
        +
        fn
    )

    if dice_denominator == 0:

        dice = 1.0

    else:

        dice = (
            2 * tp
            /
            dice_denominator
        )

    # -------------------------------------------------------------------------
    # Precision
    # -------------------------------------------------------------------------

    precision_denominator = (
        tp + fp
    )

    if precision_denominator == 0:

        precision = 0.0

    else:

        precision = (
            tp
            /
            precision_denominator
        )

    # -------------------------------------------------------------------------
    # Recall
    # -------------------------------------------------------------------------

    recall_denominator = (
        tp + fn
    )

    if recall_denominator == 0:

        recall = 0.0

    else:

        recall = (
            tp
            /
            recall_denominator
        )

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# =============================================================================
# FORWARD PASS
# =============================================================================

def forward_model(
    model,
    batch
):

    # =========================================================================
    # EARLY FUSION
    # =========================================================================

    if config.MODEL_TYPE == "early_fusion":

        sentinel1 = batch["sentinel1"].to(
            DEVICE,
            non_blocking=True
        )

        sentinel2 = batch["sentinel2"].to(
            DEVICE,
            non_blocking=True
        )

        # ---------------------------------------------------------------------
        # Expected:
        #
        # S1 = [T1 VV, T1 VH, T2 VV, T2 VH]
        # S2 = [T1 B2, B3, B4, B8,
        #       T2 B2, B3, B4, B8]
        #
        # Early-fusion model expects:
        #
        # [S1 T1,
        #  S2 T1,
        #  S1 T2,
        #  S2 T2]
        #
        # ---------------------------------------------------------------------

        s1_t1 = sentinel1[:, 0:2]

        s1_t2 = sentinel1[:, 2:4]

        s2_t1 = sentinel2[:, 0:4]

        s2_t2 = sentinel2[:, 4:8]

        stacked = torch.cat(
            [
                s1_t1,
                s2_t1,
                s1_t2,
                s2_t2,
            ],
            dim=1
        )

        outputs = model(
            stacked
        )

        return outputs, sentinel1

    # =========================================================================
    # DUAL STREAM
    # =========================================================================

    elif config.MODEL_TYPE == "dual_stream":

        sentinel1 = batch["sentinel1"].to(
            DEVICE,
            non_blocking=True
        )

        sentinel2 = batch["sentinel2"].to(
            DEVICE,
            non_blocking=True
        )

        outputs = model(
            sentinel1,
            sentinel2
        )

        return outputs, sentinel1

    else:

        raise ValueError(
            f"Unknown model type: {config.MODEL_TYPE}"
        )


# =============================================================================
# LABEL PREPARATION
# =============================================================================

def prepare_labels(batch):

    labels = batch["label"].to(
        DEVICE,
        non_blocking=True
    )

    if labels.ndim == 3:

        labels = labels.unsqueeze(1)

    labels = labels.float()

    return labels


# =============================================================================
# TRAIN ONE EPOCH
# =============================================================================

def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    epoch
):

    model.train()

    running_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    progress = tqdm(
        loader,
        desc=f"Training Epoch {epoch}"
    )

    for batch in progress:

        labels = prepare_labels(
            batch
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs, sentinel1 = forward_model(
            model,
            batch
        )

        # ---------------------------------------------------------------------
        # Shape check
        # ---------------------------------------------------------------------

        if outputs.shape != labels.shape:

            raise RuntimeError(
                "\nModel output and label shapes do not match!\n"
                f"Output : {outputs.shape}\n"
                f"Label  : {labels.shape}"
            )

        # ---------------------------------------------------------------------
        # Loss
        # ---------------------------------------------------------------------

        loss = criterion(
            outputs,
            labels
        )

        # ---------------------------------------------------------------------
        # Backpropagation
        # ---------------------------------------------------------------------

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.GRADIENT_CLIP_MAX_NORM
        )

        optimizer.step()

        # ---------------------------------------------------------------------
        # Metrics
        # ---------------------------------------------------------------------

        tp, fp, fn, tn = (
            calculate_confusion_matrix(
                outputs.detach(),
                labels,
                config.THRESHOLD
            )
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        running_loss += (
            loss.item()
            *
            sentinel1.size(0)
        )

        # ---------------------------------------------------------------------
        # Batch metrics
        # ---------------------------------------------------------------------

        batch_metrics = metrics_from_confusion(
            tp,
            fp,
            fn,
            tn
        )

        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            iou=f"{batch_metrics['iou']:.4f}",
            dice=f"{batch_metrics['dice']:.4f}"
        )

    # =========================================================================
    # EPOCH RESULTS
    # =========================================================================

    total_samples = len(
        loader.dataset
    )

    epoch_loss = (
        running_loss
        /
        total_samples
    )

    results = metrics_from_confusion(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    results["loss"] = epoch_loss

    return results


# =============================================================================
# VALIDATION
# =============================================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    with torch.no_grad():

        progress = tqdm(
            loader,
            desc="Validation"
        )

        for batch in progress:

            labels = prepare_labels(
                batch
            )

            outputs, sentinel1 = forward_model(
                model,
                batch
            )

            if outputs.shape != labels.shape:

                raise RuntimeError(
                    "\nModel output and label shapes do not match!\n"
                    f"Output : {outputs.shape}\n"
                    f"Label  : {labels.shape}"
                )

            loss = criterion(
                outputs,
                labels
            )

            tp, fp, fn, tn = (
                calculate_confusion_matrix(
                    outputs,
                    labels,
                    config.THRESHOLD
                )
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

            running_loss += (
                loss.item()
                *
                sentinel1.size(0)
            )

            batch_metrics = metrics_from_confusion(
                tp,
                fp,
                fn,
                tn
            )

            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{batch_metrics['iou']:.4f}",
                dice=f"{batch_metrics['dice']:.4f}"
            )

    total_samples = len(
        loader.dataset
    )

    epoch_loss = (
        running_loss
        /
        total_samples
    )

    results = metrics_from_confusion(
        total_tp,
        total_fp,
        total_fn,
        total_tn
    )

    results["loss"] = epoch_loss

    return results


# =============================================================================
# PRINT RESULTS
# =============================================================================

def print_epoch_results(
    epoch,
    train_results,
    val_results,
    learning_rate
):

    print()
    print("-" * 80)

    print(
        f"Epoch {epoch}"
    )

    print()

    print(
        f"Train Loss      : "
        f"{train_results['loss']:.5f}"
    )

    print(
        f"Train IoU       : "
        f"{train_results['iou']:.5f}"
    )

    print(
        f"Train Dice      : "
        f"{train_results['dice']:.5f}"
    )

    print(
        f"Train Precision : "
        f"{train_results['precision']:.5f}"
    )

    print(
        f"Train Recall    : "
        f"{train_results['recall']:.5f}"
    )

    print()

    print(
        f"Val Loss        : "
        f"{val_results['loss']:.5f}"
    )

    print(
        f"Val IoU         : "
        f"{val_results['iou']:.5f}"
    )

    print(
        f"Val Dice        : "
        f"{val_results['dice']:.5f}"
    )

    print(
        f"Val Precision   : "
        f"{val_results['precision']:.5f}"
    )

    print(
        f"Val Recall      : "
        f"{val_results['recall']:.5f}"
    )

    print()

    print(
        f"Learning rate   : "
        f"{learning_rate:.8f}"
    )

    print("-" * 80)


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================

def load_checkpoint(
    model,
    optimizer
):

    if not config.RESUME_CHECKPOINT:

        return 1, 0.0

    if not os.path.exists(
        BEST_CHECKPOINT_PATH
    ):

        print()
        print(
            "No checkpoint found."
        )

        print(
            "Starting from scratch."
        )

        return 1, 0.0

    print()
    print("=" * 80)
    print("LOADING CHECKPOINT")
    print("=" * 80)

    print(
        BEST_CHECKPOINT_PATH
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT_PATH,
        map_location=DEVICE
    )

    # -------------------------------------------------------------------------
    # New checkpoint format
    # -------------------------------------------------------------------------

    if (
        isinstance(checkpoint, dict)
        and
        "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        epoch = checkpoint.get(
            "epoch",
            0
        )

        best_val_dice = checkpoint.get(
            "best_val_dice",
            0.0
        )

        start_epoch = (
            epoch + 1
        )

        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )

        print(
            f"Best validation Dice: "
            f"{best_val_dice:.5f}"
        )

        return (
            start_epoch,
            best_val_dice
        )

    # -------------------------------------------------------------------------
    # Plain model state dictionary
    # -------------------------------------------------------------------------

    model.load_state_dict(
        checkpoint
    )

    print(
        "Loaded model weights."
    )

    return 1, 0.0


# =============================================================================
# TEST
# =============================================================================

def test_model(
    model,
    loader,
    criterion
):

    print()
    print("=" * 80)
    print("FINAL TEST")
    print("=" * 80)

    results = validate(
        model,
        loader,
        criterion
    )

    print()
    print("=" * 80)
    print("FINAL TEST RESULTS")
    print("=" * 80)

    print(
        f"Test Loss      : "
        f"{results['loss']:.5f}"
    )

    print(
        f"Test IoU       : "
        f"{results['iou']:.5f}"
    )

    print(
        f"Test Dice      : "
        f"{results['dice']:.5f}"
    )

    print(
        f"Test Precision : "
        f"{results['precision']:.5f}"
    )

    print(
        f"Test Recall    : "
        f"{results['recall']:.5f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        f"  TP : {results['tp']:,}"
    )

    print(
        f"  FP : {results['fp']:,}"
    )

    print(
        f"  FN : {results['fn']:,}"
    )

    print(
        f"  TN : {results['tn']:,}"
    )

    print("=" * 80)

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():

    # =========================================================================
    # SEED
    # =========================================================================

    set_seed(
        config.SEED
    )

    # =========================================================================
    # HEADER
    # =========================================================================

    print()
    print("=" * 80)
    print("OSCD CHANGE DETECTION TRAINING")
    print("=" * 80)

    print()
    print(
        f"Model type    : "
        f"{config.MODEL_TYPE}"
    )

    print(
        f"Project root  : "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Device        : "
        f"{DEVICE}"
    )

    print(
        f"Patch size    : "
        f"{config.PATCH_SIZE}"
    )

    print(
        f"Batch size    : "
        f"{config.BATCH_SIZE}"
    )

    print(
        f"Epochs        : "
        f"{config.EPOCHS}"
    )

    print(
        f"Learning rate : "
        f"{config.LEARNING_RATE}"
    )

    print(
        f"Positive wt   : "
        f"{config.POS_WEIGHT}"
    )

    print(
        f"Threshold     : "
        f"{config.THRESHOLD}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"GPU memory    : "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # =========================================================================
    # LOAD CSV
    # =========================================================================

    print()
    print("=" * 80)
    print("LOADING CSV FILES")
    print("=" * 80)

    train_df = load_split_csv(
        "train"
    )

    val_df = load_split_csv(
        "val"
    )

    test_df = load_split_csv(
        "test"
    )

    print(
        f"Train samples : "
        f"{len(train_df)}"
    )

    print(
        f"Val samples   : "
        f"{len(val_df)}"
    )

    print(
        f"Test samples  : "
        f"{len(test_df)}"
    )

    # =========================================================================
    # DATASETS
    # =========================================================================

    print()
    print("=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)

    train_ds = OSCDataset(
        train_df,
        augment=True
    )

    val_ds = OSCDataset(
        val_df,
        augment=False
    )

    test_ds = OSCDataset(
        test_df,
        augment=False
    )

    print(
        f"Train dataset : "
        f"{len(train_ds)}"
    )

    print(
        f"Val dataset   : "
        f"{len(val_ds)}"
    )

    print(
        f"Test dataset  : "
        f"{len(test_ds)}"
    )

    # =========================================================================
    # DATA LOADERS
    # =========================================================================

    pin_memory = (
        config.PIN_MEMORY
        and
        torch.cuda.is_available()
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=config.SHUFFLE_TRAIN,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_oscd_batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_oscd_batch
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        collate_fn=collate_oscd_batch
    )

    # =========================================================================
    # MODEL
    # =========================================================================

    model = create_model()

    # =========================================================================
    # OPTIMIZER
    # =========================================================================

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    # =========================================================================
    # SCHEDULER
    # =========================================================================

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.SCHEDULER_FACTOR,
        patience=config.SCHEDULER_PATIENCE,
        min_lr=config.SCHEDULER_MIN_LR
    )

    # =========================================================================
    # LOSS
    # =========================================================================

    criterion = WeightedBCEDiceLoss(
        pos_weight=config.POS_WEIGHT,
        dice_smooth=config.DICE_SMOOTH
    ).to(DEVICE)

    # =========================================================================
    # RESUME
    # =========================================================================

    start_epoch, best_val_dice = (
        load_checkpoint(
            model,
            optimizer
        )
    )

    # =========================================================================
    # TRAINING
    # =========================================================================

    print()
    print("=" * 80)
    print("START TRAINING")
    print("=" * 80)

    for epoch in range(
        start_epoch,
        config.EPOCHS + 1
    ):

        print()
        print("#" * 80)

        print(
            f"EPOCH {epoch}/{config.EPOCHS}"
        )

        print("#" * 80)

        # =====================================================================
        # TRAIN
        # =====================================================================

        train_results = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch
        )

        # =====================================================================
        # VALIDATION
        # =====================================================================

        val_results = validate(
            model,
            val_loader,
            criterion
        )

        # =====================================================================
        # SCHEDULER
        # =====================================================================

        scheduler.step(
            val_results["dice"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # =====================================================================
        # PRINT
        # =====================================================================

        print_epoch_results(
            epoch,
            train_results,
            val_results,
            current_lr
        )

        # =====================================================================
        # SAVE BEST
        # =====================================================================

        if (
            val_results["dice"]
            >
            best_val_dice
        ):

            best_val_dice = (
                val_results["dice"]
            )

            checkpoint = {

                "epoch":
                    epoch,

                "model_type":
                    config.MODEL_TYPE,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_dice":
                    best_val_dice,

                "val_iou":
                    val_results["iou"],

                "val_precision":
                    val_results["precision"],

                "val_recall":
                    val_results["recall"],

                "config": {

                    "patch_size":
                        config.PATCH_SIZE,

                    "s1_channels":
                        config.S1_CHANNELS,

                    "s2_channels":
                        config.S2_CHANNELS,

                    "early_fusion_channels":
                        config.EARLY_FUSION_CHANNELS,

                    "base_channels":
                        config.BASE_CHANNELS,

                    "pos_weight":
                        config.POS_WEIGHT,

                    "threshold":
                        config.THRESHOLD,
                }
            }

            torch.save(
                checkpoint,
                BEST_CHECKPOINT_PATH
            )

            print()
            print("★ NEW BEST MODEL")

            print(
                f"Best Val Dice : "
                f"{best_val_dice:.5f}"
            )

            print(
                f"Saved to      : "
                f"{BEST_CHECKPOINT_PATH}"
            )

        # =====================================================================
        # SAVE LAST
        # =====================================================================

        last_checkpoint = {

            "epoch":
                epoch,

            "model_type":
                config.MODEL_TYPE,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_val_dice":
                best_val_dice,
        }

        torch.save(
            last_checkpoint,
            LAST_CHECKPOINT_PATH
        )

    # =========================================================================
    # LOAD BEST MODEL
    # =========================================================================

    print()
    print("=" * 80)
    print("LOADING BEST MODEL FOR TESTING")
    print("=" * 80)

    if os.path.exists(
        BEST_CHECKPOINT_PATH
    ):

        checkpoint = torch.load(
            BEST_CHECKPOINT_PATH,
            map_location=DEVICE
        )

        if (
            isinstance(checkpoint, dict)
            and
            "model_state_dict" in checkpoint
        ):

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

        print(
            "Loaded:"
        )

        print(
            BEST_CHECKPOINT_PATH
        )

    else:

        print(
            "WARNING: Best checkpoint not found."
        )

    # =========================================================================
    # TEST
    # =========================================================================

    test_results = test_model(
        model,
        test_loader,
        criterion
    )

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)

    print(
        f"Model               : "
        f"{config.MODEL_TYPE}"
    )

    print(
        f"Best Validation Dice: "
        f"{best_val_dice:.5f}"
    )

    print(
        f"Test IoU             : "
        f"{test_results['iou']:.5f}"
    )

    print(
        f"Test Dice            : "
        f"{test_results['dice']:.5f}"
    )

    print(
        f"Test Precision       : "
        f"{test_results['precision']:.5f}"
    )

    print(
        f"Test Recall          : "
        f"{test_results['recall']:.5f}"
    )

    print()
    print(
        "Best model:"
    )

    print(
        BEST_CHECKPOINT_PATH
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    main()