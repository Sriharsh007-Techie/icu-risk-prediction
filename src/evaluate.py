from preprocess import *

from dataset import ICUDataset

from tcn_model import TCNRiskPredictor


from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


from torch.utils.data import DataLoader


import torch

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

import os



# ==================================================
# Hyperparameters
# ==================================================

SEQUENCE_LENGTH = 16

BATCH_SIZE = 64

HIDDEN_DIM = 64



# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing device:")
print(device)



# ==================================================
# Load Dataset
# ==================================================

print("\nLoading dataset...")

X, y = load_complete_dataset(
    sequence_length=SEQUENCE_LENGTH
)

print("Dataset loaded")



# ==================================================
# Train / Validation / Test Split
# ==================================================

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)



# ==================================================
# Test Dataset
# ==================================================

test_dataset = ICUDataset(
    X_test,
    y_test
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)



# ==================================================
# Load Best Model
# ==================================================

CHECKPOINT_PATH = (
    "../checkpoints"
)

# ------------------------------------------
# Find latest experiment folder
# ------------------------------------------

all_runs = sorted(
    os.listdir(CHECKPOINT_PATH)
)

latest_run = all_runs[-1]


model_path = os.path.join(
    CHECKPOINT_PATH,
    latest_run,
    "best_model.pth"
)


print("\nLoading model:")
print(model_path)



model = TCNRiskPredictor(
    hidden_dim=HIDDEN_DIM
).to(device)


model.load_state_dict(
    torch.load(
        model_path,
        map_location=device
    )
)

model.eval()



# ==================================================
# Evaluation
# ==================================================

all_predictions = []

all_probabilities = []

all_labels = []



with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)

        batch_y = batch_y.to(device)


        outputs = model(batch_X)


        probabilities = torch.softmax(
            outputs,
            dim=1
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities[:, 1].cpu().numpy()
        )

        all_labels.extend(
            batch_y.cpu().numpy()
        )



# ==================================================
# Classification Report
# ==================================================

print("\n===================================")

print("CLASSIFICATION REPORT")

print("===================================")

print(
    classification_report(
        all_labels,
        all_predictions
    )
)



# ==================================================
# Confusion Matrix
# ==================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print("\nConfusion Matrix:")

print(cm)



# ==================================================
# ROC-AUC
# ==================================================

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

print(f"\nROC-AUC Score: {roc_auc:.4f}")



# ==================================================
# Create Evaluation Folder
# ==================================================

EVAL_DIR = "../evaluation"

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)



# ==================================================
# Plot Confusion Matrix
# ==================================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.savefig(
    f"{EVAL_DIR}/confusion_matrix.png"
)

plt.close()



# ==================================================
# ROC Curve
# ==================================================

fpr, tpr, thresholds = roc_curve(
    all_labels,
    all_probabilities
)


plt.figure(figsize=(6, 5))

plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle="--")

plt.title(
    f"ROC Curve (AUC = {roc_auc:.4f})"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.savefig(
    f"{EVAL_DIR}/roc_curve.png"
)

plt.close()



print("\n===================================")

print("Evaluation complete")

print("Plots saved in evaluation/")

print("===================================")