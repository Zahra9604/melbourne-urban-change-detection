import torch
import numpy as np
from torch.utils.data import DataLoader
from oscd_dataset import OSCDataset
from preprocess import load_split_csv
from utils import collate_oscd_batch
import os
import sys
from utils import load_trained_model

# --- Fix Module Search Path ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

CHECKPOINT_NAME = "best_model.pth"

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", CHECKPOINT_NAME)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_test():
    test_df = load_split_csv("test")
    test_ds = OSCDataset(test_df, patch_size=256, stride=128, augment=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, collate_fn=collate_oscd_batch)

    model = load_trained_model(WEIGHTS_PATH)
    model.eval()

    tp, fp, fn = 0, 0, 0
    with torch.no_grad():
        for batch in test_loader:
            s1 = batch["sentinel1"].to(DEVICE)
            s2 = batch["sentinel2"].to(DEVICE)
            labels = batch["label"].numpy()

            probs = torch.sigmoid(model(s1, s2)).cpu().numpy()[:, 0]
            preds = (probs >= 0.5).astype(np.uint8)

            tp += np.sum((preds == 1) & (labels == 1))
            fp += np.sum((preds == 1) & (labels == 0))
            fn += np.sum((preds == 0) & (labels == 1))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * (precision * recall) / max(precision + recall, 1e-6)

    print(f"\n--- OSCD Test Evaluation ---")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

if __name__ == "__main__":
    evaluate_test()