# ICU Risk Prediction using Attention-Based Temporal Convolutional Networks (TCN)

A deep learning-based medical AI system for ICU patient risk prediction using multivariate physiological time-series data from the PhysioNet ICU dataset.

This project implements a research-style end-to-end pipeline for:
- ICU physiological signal preprocessing
- Temporal sequence modeling
- Residual Dilated Temporal Convolutional Networks (TCN)
- Attention mechanisms
- Risk classification
- Experiment tracking with TensorBoard
- Scientific evaluation using ROC-AUC and confusion matrices

---

# Project Overview

Intensive Care Unit (ICU) patients generate continuous physiological signals such as:
- Heart Rate (HR)
- Respiration Rate
- Blood Pressure
- Temperature

These physiological signals evolve over time and can indicate clinical deterioration.

This project uses deep learning to model temporal physiological patterns and predict patient risk states using:
- Residual Temporal Convolutional Networks
- Dilated convolutions
- Attention mechanisms

---

# Dataset

Dataset Source:
- PhysioNet ICU Dataset

The project uses ICU physiological time-series data from PhysioNet.

Due to dataset size and licensing considerations, the raw dataset is not included in this repository.

## Download Dataset

Download the dataset manually from:

https://physionet.org/

After downloading, place the files inside:

```bash
data/raw/set-a/
data/raw/set-b/
```

The preprocessing pipeline automatically loads and processes all patient files.

---

# Features Used

Current physiological signals:
- HR
- RespRate
- Temp
- NISysABP
- NIDiasABP
- NIMAP

---

# Model Architecture

The implemented architecture includes:

Input Sequence  
↓  
Residual Temporal Blocks  
↓  
Dilated Conv1D Layers  
↓  
Batch Normalization  
↓  
Dropout  
↓  
Attention Mechanism  
↓  
Classifier Head  
↓  
Risk Prediction

Key architectural features:
- Residual learning
- Dilated temporal convolutions
- Temporal attention
- Deep temporal receptive fields
- GPU acceleration using PyTorch

---

# Project Structure

```bash
icu-risk-prediction/
│
├── data/
│
├── src/
│   ├── preprocess.py
│   ├── dataset.py
│   ├── tcn_model.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│
├── logs/
├── checkpoints/
├── evaluation/
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Preprocessing Pipeline

The preprocessing pipeline performs:
- Signal filtering
- Missing value interpolation
- Long-to-wide signal transformation
- Feature normalization
- Temporal sequence generation
- Multi-patient dataset aggregation

---

# Training Pipeline

The training framework includes:
- Train / Validation / Test split
- TensorBoard experiment tracking
- Timestamped experiment logging
- Checkpoint saving
- tqdm progress bars
- GPU training support

---

# Evaluation Metrics

The project evaluates:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

# Results

## Final Performance

| Metric | Score |
|---|---|
| Test Accuracy | ~76.9% |
| ROC-AUC | ~0.842 |

The model demonstrates strong temporal discrimination capability on multivariate ICU physiological data.

---

# ROC Curve

![ROC Curve](evaluation/roc_curve.png)

---

# Confusion Matrix

![Confusion Matrix](evaluation/confusion_matrix.png)

---

# Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- TensorBoard
- Matplotlib
- Seaborn
- tqdm

---

# Installation

Clone the repository:

```bash
git clone <your_repo_url>
```

Create virtual environment:

```bash
python -m venv venv
```

Activate environment:

Windows:
```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Run training:

```bash
cd src
python train.py
```

Launch TensorBoard:

```bash
tensorboard --logdir=logs
```

---

# Evaluation

Run evaluation:

```bash
cd src
python evaluate.py
```

Evaluation outputs:
- ROC curve
- confusion matrix
- classification metrics

are saved in:

```bash
evaluation/
```

---

# Future Improvements

Planned improvements:
- Additional clinical features
- Explainable AI (SHAP)
- Attention visualization
- Streamlit dashboard
- Multi-class risk prediction
- Mortality prediction
- Transformer-based temporal modeling

---

# Research Motivation

This project was developed as a medical AI research-oriented deep learning system focused on:
- Temporal physiological modeling
- ICU patient monitoring
- Clinical decision support
- Deep learning for healthcare

---

# Author

Sriharsh Kulkarni

M.Sc. Robotic Systems Engineering  
RWTH Aachen University