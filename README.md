# Hybrid Fraud Detection using Graph Embeddings (Node2Vec + XGBoost)

## Overview
This project focuses on detecting fraudulent financial transactions using a hybrid machine learning approach that combines:

- Graph-based embeddings (Node2Vec)
- Tabular transaction features
- Ensemble learning (XGBoost)

The goal is to improve fraud detection by capturing relationships between accounts, which traditional models often ignore.

---

## Problem Statement
Fraud detection datasets are highly imbalanced, where fraudulent transactions are rare. Traditional models like Random Forest rely only on tabular data and fail to capture interactions between entities (accounts).

---

## Approach

### 1. Baseline Model
- Random Forest + SMOTE
- Uses only tabular features

### 2. Hybrid Model (Proposed)
- Construct transaction graph (accounts = nodes, transactions = edges)
- Learn embeddings using Node2Vec
- Combine:
  - sender embeddings
  - receiver embeddings
  - transaction features
- Train XGBoost classifier

---

## Dataset
- PaySim dataset (Kaggle)
- Credit Card Fraud dataset (baseline)

> Note: Datasets are not included due to size. Download from Kaggle.

---

## Results

| Model | Precision | Recall | F1 Score | AUROC |
|------|----------|--------|----------|--------|
| Random Forest + SMOTE | 0.83 | 0.83 | 0.83 | 0.985 |
| Node2Vec + XGBoost | **0.98** | 0.74 | **0.84** | **0.997** |

---

## Key Insights
- Graph-based features significantly improve fraud detection
- Reduces false positives
- Captures coordinated fraud patterns across accounts

---

## Project Structure

```text
project_root/
├── notebooks/
│   ├── 01_baseline_rf_smote.ipynb        # Baseline model with class balancing
│   └── 02_hybrid_node2vec_xgboost.ipynb  # Graph embeddings + boosted trees
├── data/                                 # Raw and processed datasets (excluded)
└── paper/
    └── research_paper.pdf                # Project documentation
```

## How to Run

1. Open notebooks in Google Colab
2. Set Kaggle API credentials
3. Run all cells

---

## Technologies Used
- Python
- Scikit-learn
- PyTorch Geometric
- XGBoost
- Pandas / NumPy

---

## Future Improvements
- Graph Neural Networks (GCN, GAT)
- Temporal fraud detection
- Real-time deployment

---

## Author
Aditya Kumar Kapar, Mayank Guptaa, Shaunak Borker, Milan Kuiry, Harsh Garg
