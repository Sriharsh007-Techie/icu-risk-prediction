from preprocess import *

from dataset import ICUDataset

from tcn_model import TCNRiskPredictor


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score


from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter


from tqdm import tqdm


from datetime import datetime


import torch

import torch.nn as nn

import os



# ==================================================
# Hyperparameters
# ==================================================

SEQUENCE_LENGTH = 16

BATCH_SIZE = 64

LEARNING_RATE = 0.001

NUM_EPOCHS = 50

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
# Experiment Timestamp
# ==================================================

timestamp = datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S"
)



# ==================================================
# Experiment Directories
# ==================================================

LOG_DIR = f"../logs/{timestamp}"

CHECKPOINT_DIR = (
    f"../checkpoints/{timestamp}"
)


os.makedirs(
    LOG_DIR,
    exist_ok=True
)

os.makedirs(
    CHECKPOINT_DIR,
    exist_ok=True
)



# ==================================================
# TensorBoard
# ==================================================

writer = SummaryWriter(
    log_dir=LOG_DIR
)



# ==================================================
# Main Training Function
# ==================================================

def main():

    # ----------------------------------------------
    # Load Complete Dataset
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


    print("\nTrain shape:", X_train.shape)

    print("Validation shape:", X_val.shape)

    print("Test shape:", X_test.shape)



    # ----------------------------------------------
    # Create Datasets
    # ----------------------------------------------

    train_dataset = ICUDataset(
        X_train,
        y_train
    )

    val_dataset = ICUDataset(
        X_val,
        y_val
    )

    test_dataset = ICUDataset(
        X_test,
        y_test
    )



    # ----------------------------------------------
    # Create DataLoaders
    # ----------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )



    # ----------------------------------------------
    # Initialize Model
    # ----------------------------------------------

    model = TCNRiskPredictor(
        hidden_dim=HIDDEN_DIM
    ).to(device)


    criterion = nn.CrossEntropyLoss()


    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )



    # ----------------------------------------------
    # Training Loop
    # ----------------------------------------------

    best_val_accuracy = 0


    for epoch in range(NUM_EPOCHS):

        # ==========================================
        # TRAINING
        # ==========================================

        model.train()

        train_loss = 0

        train_predictions = []

        train_labels = []


        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
        )


        for batch_X, batch_y in train_progress:

            batch_X = batch_X.to(device)

            batch_y = batch_y.to(device)


            optimizer.zero_grad()


            outputs = model(batch_X)


            loss = criterion(
                outputs,
                batch_y
            )


            loss.backward()

            optimizer.step()


            train_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            train_predictions.extend(
                predictions.cpu().numpy()
            )

            train_labels.extend(
                batch_y.cpu().numpy()
            )


            train_progress.set_postfix({
                "loss": loss.item()
            })



        avg_train_loss = (
            train_loss / len(train_loader)
        )

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions
        )



        # ==========================================
        # VALIDATION
        # ==========================================

        model.eval()

        val_loss = 0

        val_predictions = []

        val_labels = []


        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(device)

                batch_y = batch_y.to(device)


                outputs = model(batch_X)


                loss = criterion(
                    outputs,
                    batch_y
                )


                val_loss += loss.item()


                predictions = torch.argmax(
                    outputs,
                    dim=1
                )


                val_predictions.extend(
                    predictions.cpu().numpy()
                )

                val_labels.extend(
                    batch_y.cpu().numpy()
                )



        avg_val_loss = (
            val_loss / len(val_loader)
        )

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )



        # ==========================================
        # TensorBoard Logging
        # ==========================================

        writer.add_scalar(
            "Loss/Train",
            avg_train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/Validation",
            avg_val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Train",
            train_accuracy,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Validation",
            val_accuracy,
            epoch
        )



        # ==========================================
        # Save Best Model
        # ==========================================

        if val_accuracy > best_val_accuracy:

            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                f"{CHECKPOINT_DIR}/best_model.pth"
            )



        # ==========================================
        # Epoch Summary
        # ==========================================

        print("\n----------------------------------")

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        print(f"Train Loss: {avg_train_loss:.4f}")

        print(f"Train Accuracy: {train_accuracy:.4f}")

        print(f"Validation Loss: {avg_val_loss:.4f}")

        print(f"Validation Accuracy: {val_accuracy:.4f}")

        print("----------------------------------")



    # ==================================================
    # TEST EVALUATION
    # ==================================================

    print("\nEvaluating on test set...")


    model.eval()

    test_predictions = []

    test_labels = []


    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)

            batch_y = batch_y.to(device)


            outputs = model(batch_X)


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            test_predictions.extend(
                predictions.cpu().numpy()
            )

            test_labels.extend(
                batch_y.cpu().numpy()
            )



    test_accuracy = accuracy_score(
        test_labels,
        test_predictions
    )


    print("\n==================================")

    print(f"Final Test Accuracy: {test_accuracy:.4f}")

    print("==================================")



    writer.close()



# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":

    main()