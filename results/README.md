## Results

| Model | Precision | Recall | F1 Score | AUROC |
|------|----------|--------|----------|--------|
| Random Forest + SMOTE | 0.83 | 0.83 | 0.83 | 0.985 |
| Node2Vec + XGBoost | **0.98** | 0.74 | **0.84** | **0.997** |

### Key Takeaways
- Graph-based features improve fraud detection performance
- Hybrid model achieves better precision and AUROC
- Captures relationships between accounts, reducing false positives

The hybrid model improves AUROC from 0.985 → 0.997, showing better separation between fraudulent and normal transactions.
