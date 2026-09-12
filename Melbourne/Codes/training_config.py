# ============================================================
# MELBOURNE CHANGE DETECTION
# TRAINING CONFIGURATION
# ============================================================


# ============================================================
# MODEL STRATEGY
# ============================================================

# Available:
#
#   "early_fusion"
#   "dual_stream"

# STRATEGY = "dual_stream"
STRATEGY = "early_fusion"


# ============================================================
# TRAINING MODE
# ============================================================

# Available:
#
# "scratch"
#
#     Train Melbourne from random initialization.
#
#
# "resume_melbourne"
#
#     Continue an existing Melbourne training run.
#
#     Restores:
#         - model
#         - optimizer
#         - scheduler
#         - AMP scaler
#         - epoch
#         - best validation F1
#         - early stopping counter
#
#
# "pretrained_oscd"
#
#     Load model weights from an OSCD pretrained model.
#
#     Restores ONLY:
#         - model weights
#
#     Resets:
#         - optimizer
#         - scheduler
#         - AMP scaler
#         - epoch
#         - best validation F1
#         - early stopping counter

# TRAINING_MODE = "scratch"
# TRAINING_MODE = "resume_melbourne"
TRAINING_MODE = "pretrained_oscd"



# ============================================================
# NEW MODEL / EXPERIMENT NAME
# ============================================================

# This determines the name of the NEW Melbourne experiment.
#
# Examples:
#
#     melbourne_v1
#     melbourne_v2
#     melbourne_oscd_finetune
#
# IMPORTANT:
# Use a different name for each experiment.
if TRAINING_MODE == 'scratch':
    MODEL_RUN_NAME = "melbourne_v1"
elif TRAINING_MODE == 'resume_melbourne':
    MODEL_RUN_NAME = "melbourne_v2"

elif TRAINING_MODE == 'pretrained_oscd':
    MODEL_RUN_NAME = "melbourne_oscd_finetune"



# ============================================================
# MELBOURNE CHECKPOINT TO RESUME
# ============================================================

# Used ONLY when:
#
#     TRAINING_MODE = "resume_melbourne"
#
# Actual location:
#
#     Melbourne\Codes\checkpoints\
#     last_dual_stream_melbourne_v1.pth

RESUME_CHECKPOINT_PATH = (
    r"C:\melbourne-urban-change-detection\Melbourne\MelbourneData\checkpoints"
    r"\last_dual_stream_melbourne_v1.pth"
)


# ============================================================
# HISTORY OF PREVIOUS MELBOURNE RUN
# ============================================================

# Used ONLY when:
#
#     TRAINING_MODE = "resume_melbourne"
#
# Actual location:
#
#     Melbourne\Codes\results\
#     dual_stream_melbourne_v1_history.csv

RESUME_HISTORY_PATH = (
    r"C:\melbourne-urban-change-detection\Melbourne\MelbourneData\results\dual_stream_melbourne_v1_history.csv"
)


# ============================================================
# OSCD PRETRAINED MODEL
# ============================================================

# Used ONLY when:
#
#     TRAINING_MODE = "pretrained_oscd"
#
# Example:
#
#     C:\melbourne-urban-change-detection\OSCD\
#     weights\last_model.pth
if STRATEGY == "dual_stream":
    PRETRAINED_MODEL_PATH = (
    r"C:\melbourne-urban-change-detection\Melbourne\MelbourneData\checkpoints\best_dual_stream_oscd.pth"
        )

else:
    PRETRAINED_MODEL_PATH = (
    r"C:\melbourne-urban-change-detection\Melbourne\MelbourneData\checkpoints\best_early_fusion_oscd.pth"
        )



# ============================================================
# TRAINING PARAMETERS
# ============================================================

BATCH_SIZE = 8

EPOCHS = 100

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-4


# ============================================================
# LOSS
# ============================================================

BCE_WEIGHT = 0.5

DICE_WEIGHT = 0.5


# ============================================================
# DATA
# ============================================================

NORMALIZE = True

NUM_WORKERS = 4


# ============================================================
# AMP
# ============================================================

USE_AMP = True


# ============================================================
# GRADIENT CLIPPING
# ============================================================

GRADIENT_CLIPPING = 1.0


# ============================================================
# EARLY STOPPING
# ============================================================

EARLY_STOPPING_PATIENCE = 15 #a "how long have we been stuck?" counter :))