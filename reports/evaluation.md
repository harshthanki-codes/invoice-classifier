## Model Evaluation Report

**Training samples:** 185  
**CV F1-macro:** `0.8213 ± 0.0386`  
**Production ready:** ⚠️ Not yet — add more training data

### Per-Class Performance

| Category | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Cloud/Software | 0.84 | 0.89 | 0.86 | 35 |
| Inventory | 0.81 | 0.87 | 0.84 | 30 |
| Logistics | 0.77 | 0.90 | 0.83 | 30 |
| Office Supplies | 0.85 | 0.73 | 0.79 | 30 |
| Travel | 0.89 | 0.83 | 0.86 | 30 |
| Utilities | 0.81 | 0.73 | 0.77 | 30 |

### Confusion Matrix

Rows = actual, Columns = predicted

| | **Cloud/So** | **Inventor** | **Logistic** | **Office S** | **Travel** | **Utilitie** |
|---|---|---|---|---|---|---|
| **Cloud/So** | 31 | 0 | 2 | 0 | 0 | 2 |
| **Inventor** | 0 | 26 | 1 | 3 | 0 | 0 |
| **Logistic** | 0 | 1 | 27 | 1 | 1 | 0 |
| **Office S** | 0 | 5 | 0 | 22 | 1 | 2 |
| **Travel** | 0 | 0 | 4 | 0 | 25 | 1 |
| **Utilitie** | 6 | 0 | 1 | 0 | 1 | 22 |

> **Confidence threshold:** Predictions below `0.72` return `review_recommended: true`.
> Tune this threshold based on your acceptable false-positive rate in production.