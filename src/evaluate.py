from preprocess import *

from dataset import ICUDataset

from tcn_model import TCNRiskPredictor


from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


from torch.utils.data import DataLoader


import torch

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import os



# ==================================================
# Configuration
# ==================================================

SEQUENCE_LENGTH = 32

BATCH_SIZE = 64

HIDDEN_DIM = 64



# ==================================================
# MODEL PATH
# ==================================================

MODEL_PATH = (
    "../checkpoints/2026-05-26_07-59-27/"
    "best_model_2026-05-26_07-59-27.pth"
)



# ==================================================
# Evaluation Output Directory
# ==================================================

EVAL_DIR = "../evaluation"

os.makedirs(
    EVAL_DIR,
    exist_ok=True
)



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
# Main Evaluation Function
# ==================================================

def main():

    # ----------------------------------------------
    # Load Dataset
    # ----------------------------------------------

    print("\nLoading dataset...")

    X, y = load_complete_dataset(
        sequence_length=SEQUENCE_LENGTH
    )

    print("\nDataset loaded")

    print("X shape:", X.shape)

    print("y shape:", y.shape)



    # ----------------------------------------------
    # Train / Validation / Test Split
    # ----------------------------------------------

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


    print("\nTest shape:", X_test.shape)



    # ----------------------------------------------
    # Create Test Dataset
    # ----------------------------------------------

    test_dataset = ICUDataset(
        X_test,
        y_test
    )


    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



    # ----------------------------------------------
    # Initialize Model
    # ----------------------------------------------

    INPUT_DIM = X.shape[2]

    print("\nInput Features:", INPUT_DIM)


    model = TCNRiskPredictor(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)



    # ----------------------------------------------
    # Load Trained Weights
    # ----------------------------------------------

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    print("\nModel loaded successfully")



    # ----------------------------------------------
    # Evaluation
    # ----------------------------------------------

    model.eval()


    predictions = []

    true_labels = []

    probabilities = []


    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)

            batch_y = batch_y.to(device)


            outputs = model(batch_X)


            probs = torch.softmax(
                outputs,
                dim=1
            )[:, 1]


            preds = torch.argmax(
                outputs,
                dim=1
            )


            predictions.extend(
                preds.cpu().numpy()
            )

            true_labels.extend(
                batch_y.cpu().numpy()
            )

            probabilities.extend(
                probs.cpu().numpy()
            )



    # ==================================================
    # Metrics
    # ==================================================

    accuracy = accuracy_score(
        true_labels,
        predictions
    )


    print("\n==================================")

    print("FINAL EVALUATION RESULTS")

    print("==================================")

    print(f"\nTest Accuracy: {accuracy:.4f}")



    # ----------------------------------------------
    # Classification Report
    # ----------------------------------------------

    print("\nClassification Report:\n")

    print(
        classification_report(
            true_labels,
            predictions
        )
    )



    # ==================================================
    # Confusion Matrix
    # ==================================================

    cm = confusion_matrix(
        true_labels,
        predictions
    )


    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues"
    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("True")


    confusion_matrix_path = (
        f"{EVAL_DIR}/confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path
    )

    plt.close()


    print(
        f"\nConfusion matrix saved to:"
    )

    print(confusion_matrix_path)



    # ==================================================
    # ROC Curve
    # ==================================================

    fpr, tpr, thresholds = roc_curve(
        true_labels,
        probabilities
    )


    roc_auc = auc(
        fpr,
        tpr
    )


    plt.figure(figsize=(7, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.4f}"
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend(loc="lower right")


    roc_curve_path = (
        f"{EVAL_DIR}/roc_curve.png"
    )

    plt.savefig(
        roc_curve_path
    )

    plt.close()


    print("\nROC curve saved to:")

    print(roc_curve_path)


    print("\nROC-AUC Score:")

    print(f"{roc_auc:.4f}")



# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":

    main()