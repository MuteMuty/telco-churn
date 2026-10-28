# Telco Customer Churn Prediction

A telecom company is losing customers to competitors. This project predicts **which customers are likely to leave next month**, so the sales and marketing team can offer them an incentive to stay.

The full analysis, with every decision explained, is in [`notebooks/churn_analysis.ipynb`](notebooks/churn_analysis.ipynb).

## Results

Measured once on a held-out test set of 1,409 customers the model never saw during training:

| Metric | Value |
|---|---|
| ROC-AUC | 0.842 |
| PR-AUC (average precision) | 0.636 (random guessing: 0.265) |
| Recall at threshold 0.31 | 0.746 |
| Precision at threshold 0.31 | 0.529 |
| F1 at threshold 0.31 | 0.619 |

**In business terms:** contacting the 37% of customers the model flags reaches **3 of every 4 customers who would leave**, and about half of those contacted are real churn risks. With a smaller budget, the riskiest 20% of customers still include half of all churners. The strongest churn signals are **short tenure, a month-to-month contract, fiber internet, and no tech support or online security**.

## Approach

1. **Cleaning.** TotalCharges was read as text because of 11 blank values. They belong exactly to the 11 new customers (tenure 0) who haven't been billed yet, so they become 0. SeniorCitizen is converted from 0/1 to "No"/"Yes" like the other yes/no columns, Churn to 0/1, and customerID is dropped.
2. **Exploratory analysis.** 26.5% of customers churn, so the classes are imbalanced and accuracy is misleading (always predicting "stays" is 73.5% accurate). Contract type is the clearest signal: 42.7% churn on month-to-month contracts vs 2.8% on two-year contracts.
3. **Features and preprocessing.** Two new features: `AvgMonthlyCharges` (TotalCharges / tenure, as the brief suggests) and `NumAddons` (number of add-on services). Numeric features are standardised and categorical ones one-hot encoded. Everything is in one scikit-learn pipeline, so cross-validation has no data leakage and the saved model accepts raw customer data.
4. **Model.** Logistic regression, random forest and gradient boosting were compared with stratified 5-fold cross-validation on the training set (80%). After tuning, all three are within 0.01 PR-AUC of each other. The chosen model is **L1-regularised logistic regression** (`C = 0.3`, no class weights): practically as accurate as the alternatives, explainable, doesn't overfit, and its probabilities match real churn rates.
5. **Evaluation.** The decision threshold (0.31) is chosen on out-of-fold predictions for the training set: the highest precision that still catches at least 75% of churners. Only then is the test set used, once. Permutation importance shows what drives the predictions.
6. **REST API.** Section 6 of the notebook describes how the CRM would use the model through a FastAPI service (`POST /predict`, input validation, batch scoring, monitoring).

**Metrics:** the main metric is **PR-AUC**, because it focuses on the churn class and doesn't depend on a threshold. ROC-AUC is reported for comparison, and recall, precision and F1 at the chosen threshold describe the actual yes/no decisions.

## Project structure

```
├── data/WA_Fn-UseC_-Telco-Customer-Churn.csv   dataset (Kaggle, IBM sample data)
├── notebooks/churn_analysis.ipynb              the full analysis
├── scripts/model_experiments.py                reproduces the model experiments (notebook section 4.2)
├── src/telco_churn/
│   ├── data.py        loading and cleaning
│   ├── features.py    new features and preprocessing pipeline
│   ├── model.py       candidate models, cross-validation, training, saving
│   ├── evaluate.py    threshold choice, test metrics, feature importance
│   └── plots.py       charts
├── models/            trained model (created by the notebook, not committed)
├── pyproject.toml     dependencies
└── uv.lock            exact versions of all dependencies
```

## How to use

**Requirements:** [uv](https://docs.astral.sh/uv/). It installs the right Python version (3.12) and all dependencies.

```bash
git clone https://github.com/MuteMuty/telco-churn.git
cd telco-churn
uv sync
```

**Run the analysis.** Open the notebook and run all cells (about 15 seconds):

```bash
uv run jupyter lab notebooks/churn_analysis.ipynb
```

or run it without opening it:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/churn_analysis.ipynb
```

This reproduces all results and saves the trained pipeline to `models/churn_model.joblib`.

**Reproduce the model experiments** (hyperparameter search, nested CV, ensemble, class weights; about 4 minutes):

```bash
uv run python scripts/model_experiments.py
```

**Use the trained model** (after running the notebook once). It takes raw customer data, with the same columns as the dataset:

```python
from telco_churn.data import clean_data, load_raw_data, split_features_target
from telco_churn.model import load_model

model = load_model()
customers, _ = split_features_target(clean_data(load_raw_data()))  # or your own data

churn_probability = model.predict_proba(customers)[:, 1]
flagged = churn_probability >= 0.31  # threshold chosen in the notebook, section 5.1
```

**Check the code style:**

```bash
uv run ruff check .
uv run ruff format --check .
```
