import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import StandardScaler


# --------------------------------------------------
# Important Physiological Signals
# --------------------------------------------------

SIGNALS = [
    "HR",
    "RespRate",
    "Temp",
    "NISysABP",
    "NIDiasABP",
    "NIMAP"
]


# --------------------------------------------------
# Load Single Patient File
# --------------------------------------------------

def load_patient_data(file_path):

    df = pd.read_csv(file_path)

    return df


# --------------------------------------------------
# Filter Important Signals
# --------------------------------------------------

def filter_signals(df):

    df = df[
        df["Parameter"].isin(SIGNALS)
    ]

    return df


# --------------------------------------------------
# Convert Long → Wide Format
# --------------------------------------------------

def pivot_signals(df):

    pivot_df = df.pivot_table(
        index="Time",
        columns="Parameter",
        values="Value",
        aggfunc="mean"
    )

    return pivot_df


# --------------------------------------------------
# Handle Missing Values
# --------------------------------------------------

def handle_missing_values(df):

    df = df.interpolate()

    df = df.ffill()

    df = df.bfill()

    return df


# --------------------------------------------------
# Normalize Features
# --------------------------------------------------

def normalize_features(df):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(
        scaled,
        columns=df.columns,
        index=df.index
    )

    return scaled_df, scaler


# --------------------------------------------------
# Generate Risk Labels
# --------------------------------------------------

def generate_risk_label(row):

    hr = row["HR"]

    rr = row["RespRate"]

    temp = row["Temp"]

    map_value = row["NIMAP"]


    if (
        hr > 100
        or rr > 22
        or temp > 38
        or map_value < 65
    ):

        return 1


    return 0


# --------------------------------------------------
# Create Time-Series Sequences
# --------------------------------------------------

def create_classification_sequences(
    data,
    labels,
    sequence_length=8
):

    X = []

    y = []


    for i in range(
        len(data) - sequence_length
    ):

        sequence = data[
            i:i + sequence_length
        ]

        label = labels[
            i + sequence_length
        ]

        X.append(sequence)

        y.append(label)


    return np.array(X), np.array(y)


# --------------------------------------------------
# Process One Patient File
# --------------------------------------------------

def process_patient_file(
    file_path,
    sequence_length=8
):

    try:

        # -----------------------------------------
        # Load patient data
        # -----------------------------------------

        df = load_patient_data(
            file_path
        )


        # -----------------------------------------
        # Filter signals
        # -----------------------------------------

        df = filter_signals(df)


        # -----------------------------------------
        # Pivot signals
        # -----------------------------------------

        pivot_df = pivot_signals(df)


        # -----------------------------------------
        # Check required signals
        # -----------------------------------------

        required_columns = SIGNALS.copy()

        for col in required_columns:

            if col not in pivot_df.columns:

                return None, None


        # -----------------------------------------
        # Handle missing values
        # -----------------------------------------

        pivot_df = handle_missing_values(
            pivot_df
        )


        # -----------------------------------------
        # Generate labels
        # -----------------------------------------

        pivot_df["RiskLabel"] = pivot_df.apply(
            generate_risk_label,
            axis=1
        )


        # -----------------------------------------
        # Normalize features
        # -----------------------------------------

        scaled_df, scaler = normalize_features(
            pivot_df.drop(
                columns=["RiskLabel"]
            )
        )


        # -----------------------------------------
        # Create sequences
        # -----------------------------------------

        X, y = create_classification_sequences(
            scaled_df.values,
            pivot_df["RiskLabel"].values,
            sequence_length=sequence_length
        )


        # -----------------------------------------
        # Skip tiny patients
        # -----------------------------------------

        if len(X) == 0:

            return None, None


        return X, y


    except Exception as e:

        print(
            f"Error processing {file_path}: {e}"
        )

        return None, None


# --------------------------------------------------
# Load All Patients From Folder
# --------------------------------------------------

def load_dataset_from_folder(
    folder_path,
    sequence_length=8
):

    all_X = []

    all_y = []


    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                folder_path,
                filename
            )

            X, y = process_patient_file(
                file_path,
                sequence_length
            )

            if X is not None:

                all_X.append(X)

                all_y.append(y)


    X = np.concatenate(all_X)

    y = np.concatenate(all_y)

    return X, y


# --------------------------------------------------
# Load Complete Dataset
# --------------------------------------------------

def load_complete_dataset(
    sequence_length=8
):

    # -----------------------------------------
    # Load set-a
    # -----------------------------------------

    X_a, y_a = load_dataset_from_folder(
        "../data/raw/set-a",
        sequence_length
    )


    # -----------------------------------------
    # Load set-b
    # -----------------------------------------

    X_b, y_b = load_dataset_from_folder(
        "../data/raw/set-b",
        sequence_length
    )


    # -----------------------------------------
    # Combine datasets
    # -----------------------------------------

    X = np.concatenate([
        X_a,
        X_b
    ])

    y = np.concatenate([
        y_a,
        y_b
    ])


    return X, y