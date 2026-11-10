# Telco Customer Churn Prediction

Which telecom customers are about to leave, and who should get a retention offer? This project compares **logistic regression** and **decision trees** for predicting churn on the Telco customer dataset (7,032 customers, 18 features).

📓 **Notebook:** [`churn_logistic_regression_vs_decision_tree.ipynb`](churn_logistic_regression_vs_decision_tree.ipynb) · 📄 **Report:** [`reports/churn_prediction_report.pdf`](reports/churn_prediction_report.pdf)

## Approach

- **Preparation** – one-hot encoding of categorical features, normalisation, train/test split.
- **Logistic regression** – baseline, 10-fold cross-validation, ROC curve, then L1 and L2 regularisation tuned with `GridSearchCV`.
- **Decision tree** – comparing split criteria (Gini, entropy, log loss), studying `max_depth`, `min_samples_leaf`, `max_leaf_nodes` and `min_impurity_decrease` against overfitting, then a full `GridSearchCV`.

## Results (test set)

| Metric | Logistic regression (L2) | Decision tree (depth 5, 5 leaves) |
| :--- | :--- | :--- |
| Accuracy | **0.805** | 0.779 |
| Precision | 0.696 | **0.704** |
| Recall | **0.533** | 0.356 |
| AUC | 0.804 | **0.815** |

An unconstrained tree scored 99.8% on training data but only 72.4% on test data; pruning it to 4 features fixed the overfitting.

**Strongest churn signals**
- Logistic regression: fibre-optic internet, paying by electronic check, paperless billing, multiple lines.
- Decision tree: contract length (one- and two-year contracts strongly reduce churn), fibre-optic internet, tenure.

## Running

```bash
pip install -r requirements.txt
jupyter notebook churn_logistic_regression_vs_decision_tree.ipynb
```

Data: `data/churn.csv` (IBM Telco Customer Churn sample).
