# DSN Mart Sales Prediction

**DSN AI Bootcamp 2026 - Qualification Hackathon (Machine Learning Track)**

Predicting total product-store sales for **DSN Mart**, a Nigerian retail chain, from
historical product and outlet data. This was the qualifying hackathon for the Data Science
Nigeria AI Bootcamp 2026.

**Evaluation metric:** Root Mean Squared Error (RMSE) - lower is better.

---

## Problem

DSN Mart operates stores across Nigeria, from small corner shops to flagship hypermarkets.
The task is to predict the total sales of a given product at a given store, based on what is
known about the product (weight, category, price, shelf visibility) and the outlet (size,
location tier, format, age).

- **Training data:** 6,818 product-store records
- **Test data:** 1,705 records to predict

---

## Approach

The project follows the full data science cycle: **Understand → Analyse → Model → Predict → Communicate.**

### 1. Data Cleaning
Four data-quality issues were identified and fixed with reasoned, leakage-free choices:
- **Inconsistent category capitalisation** - 48 raw labels standardised to 16 real categories.
- **Impossible zero shelf-visibility** - treated as missing (a sold product must occupy shelf
  space) and filled with each product's average.
- **Missing product weights** - filled with each product's own average weight.
- **Missing store sizes (~28%)** - three stores never reported size; one (a Corner Shop) was
  inferred as Small since all known Corner Shops are Small, and the two Standard Supermarkets
  (which come in all sizes) were marked "Unknown" rather than guessed.

### 2. Feature Engineering
- Log-transformed the right-skewed target to improve learning.
- Engineered `price_per_kg` (price relative to weight).

### 3. Modelling & Experiments
Multiple models and techniques were tested against a consistent validation set:

| Approach | Validation RMSE |
|---|---|
| Random Forest (baseline) | 1170 |
| XGBoost (default) | 1128 |
| XGBoost (GridSearch tuned) | 1133 |
| LightGBM | 1133 |
| XGBoost + Optuna (one-hot) | 1123 |
| XGBoost + Optuna (label) | 1123 |
| **CatBoost (tuned, native categoricals)** | **~1116 - final model** |

Feature-importance analysis showed sales were driven mainly by `store_format` and
`product_price`, which explained why engineered and target-based features added little.
Label and one-hot encoding performed almost identically once tuned; CatBoost, using its native
handling of categorical features, gave the best result.

### 4. Final Model
Tuned **CatBoost** (`iterations=800, learning_rate=0.03, depth=5, l2_leaf_reg=3`), trained on
raw categorical features, achieving a validation RMSE of approximately **1116**.

---

## Key Takeaway

Systematic testing of models, encodings, engineered features, hyperparameter tuning (GridSearch
and Optuna), and ensembling showed the dataset's signal was concentrated in a few features.
The final model was selected for its validation performance and generalisation, prioritising
honest, leakage-free validation over marginal leaderboard gains.

---

## Files

- `DSN_Mart_Sales_Prediction.ipynb` — full analysis, experiments, and final model.

## Tools

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · CatBoost · Optuna · Matplotlib · Seaborn

---

*Data provided by the DSN Bootcamp Qualification Hackathon 2026 (Kaggle). Competition data is
not redistributed in this repository.*
