import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler



# ==========================================
# Clinical Signals Used
# ==========================================

signals = [

    # Vital Signs
    "HR",
    "RespRate",
    "Temp",

    # Blood Pressure
    "NISysABP",
    "NIDiasABP",
    "NIMAP",

    # Neurological
    "GCS",

    # Renal / Output
    "Urine",

    # Kidney Function
    "BUN",
    "Creatinine",

    # Metabolic
    "Glucose",
    "HCO3",

    # Hematology
    "HCT",
    "Platelets",
    "WBC",

    # Electrolytes
    "K",
    "Na",
    "Mg"
]



# ==========================================
# Clinical Default Values
# ==========================================

default_values = {

    "HR": 80,
    "RespRate": 18,
    "Temp": 37,

    "NISysABP": 120,
    "NIDiasABP": 80,
    "NIMAP": 90,

    "GCS": 15,

    "Urine": 100,

    "BUN": 15,
    "Creatinine": 1.0,

    "Glucose": 110,
    "HCO3": 24,

    "HCT": 40,
    "Platelets": 250,
    "WBC": 8,

    "K": 4.0,
    "Na": 140,
    "Mg": 2.0
}



# ==========================================
# Load Single Patient File
# ==========================================

def load_patient_data(file_path):

    df = pd.read_csv(file_path)

    return df



# ==========================================
# Generate Risk Labels
# ==========================================

def generate_risk_label(row):

    risk = 0

    # --------------------------------------
    # Vital Sign Instability
    # --------------------------------------

    if row["HR"] > 100:
        risk = 1

    if row["RespRate"] > 22:
        risk = 1

    if row["Temp"] > 38:
        risk = 1

    if row["NIMAP"] < 65:
        risk = 1

    # --------------------------------------
    # Neurological
    # --------------------------------------

    if row["GCS"] < 13:
        risk = 1

    # --------------------------------------
    # Renal Dysfunction
    # --------------------------------------

    if row["Creatinine"] > 1.5:
        risk = 1

    if row["BUN"] > 25:
        risk = 1

    # --------------------------------------
    # Infection / Inflammation
    # --------------------------------------

    if row["WBC"] > 12:
        risk = 1

    # --------------------------------------
    # Metabolic Disturbance
    # --------------------------------------

    if row["Glucose"] > 180:
        risk = 1

    return risk



# ==========================================
# Process Single Patient File
# ==========================================

def process_patient_file(
    file_path,
    sequence_length=32
):

    # --------------------------------------
    # Load File
    # --------------------------------------

    df = load_patient_data(file_path)


    # --------------------------------------
    # Keep Only Selected Signals
    # --------------------------------------

    filtered_df = df[
        df["Parameter"].isin(signals)
    ]


    # Skip if empty after filtering
    if len(filtered_df) == 0:

        return np.array([]), np.array([])


    # --------------------------------------
    # Convert Long Format → Wide Format
    # --------------------------------------

    pivot_df = filtered_df.pivot_table(
        index="Time",
        columns="Parameter",
        values="Value",
        aggfunc="mean"
    )


    # --------------------------------------
    # Ensure Every Signal Exists
    # --------------------------------------

    for signal in signals:

        if signal not in pivot_df.columns:

            pivot_df[signal] = np.nan


    # Keep consistent signal order
    pivot_df = pivot_df[signals]


    # Skip completely empty patients
    if len(pivot_df) == 0:

        return np.array([]), np.array([])


    # ======================================
    # Missing Value Handling
    # ======================================

    # Forward fill
    pivot_df = pivot_df.ffill()

    # Backward fill
    pivot_df = pivot_df.bfill()

    # Clinical default fallback
    pivot_df = pivot_df.fillna(
        value=default_values
    )

    # Final safety fallback
    pivot_df = pivot_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    pivot_df = pivot_df.fillna(0)


    # ======================================
    # Generate Risk Labels
    # ======================================

    pivot_df["RiskLabel"] = pivot_df.apply(
        generate_risk_label,
        axis=1
    )


    # ======================================
    # Feature Normalization
    # ======================================

    scaler = MinMaxScaler()

    pivot_df[signals] = scaler.fit_transform(
        pivot_df[signals]
    )


    # ======================================
    # Create Temporal Sequences
    # ======================================

    X = []

    y = []


    for i in range(
        len(pivot_df) - sequence_length
    ):

        sequence = pivot_df[
            signals
        ].iloc[
            i:i + sequence_length
        ].values


        label = pivot_df[
            "RiskLabel"
        ].iloc[
            i + sequence_length
        ]


        X.append(sequence)

        y.append(label)


    X = np.array(X)

    y = np.array(y)


    return X, y



# ==========================================
# Load Complete Dataset
# ==========================================

def load_complete_dataset(
    sequence_length=32
):

    data_paths = [
        "../data/raw/set-a",
        "../data/raw/set-b"
    ]


    all_X = []

    all_y = []


    for data_path in data_paths:

        print(f"\nProcessing: {data_path}")


        for filename in os.listdir(data_path):

            if filename.endswith(".txt"):

                file_path = os.path.join(
                    data_path,
                    filename
                )


                try:

                    X, y = process_patient_file(
                        file_path,
                        sequence_length
                    )


                    if len(X) > 0:

                        all_X.append(X)

                        all_y.append(y)


                except Exception as e:

                    print(
                        f"\nError processing {filename}"
                    )

                    print(e)


    # ======================================
    # Combine All Patients
    # ======================================

    X = np.concatenate(all_X)

    y = np.concatenate(all_y)


    print("\n=================================")
    print("Final Dataset Shapes")
    print("=================================")

    print("X shape:", X.shape)
    print("y shape:", y.shape)


    return X, y