# California House Price Prediction with Neural Networks (PyTorch)

**Artificial Intelligence course · Homework 8 (Machine Learning) · Erfan Kahani**

This project predicts the **median house value** of California districts with feed-forward neural networks built in **PyTorch**. It covers the whole regression workflow: preprocessing, a simple baseline network, a deeper network, and a comparison of the two.

**Best model: a deep neural network (64 → 32 → 16), with R² = 0.763 on the test set.**

---

## Dataset

`housing.csv` is the California Housing dataset: **20,640 districts**, 9 features and 1 target.

| Feature | Type |
|---|---|
| longitude, latitude | numerical |
| housing_median_age, total_rooms, total_bedrooms, population, households | numerical |
| median_income | numerical |
| ocean_proximity | categorical (`<1H OCEAN`, `INLAND`, `NEAR BAY`, `NEAR OCEAN`, `ISLAND`) |
| **median_house_value** | **target** |

---

## Methodology

### 1. Preprocessing
- **Target analysis:** the house prices are right-skewed and capped at about $500,001. These high values are real houses, not errors, so they were **kept**. The target was log-transformed instead, `y_log = log(1 + y)`, to reduce skew and keep MSE training stable. Predictions are converted back to dollars with `expm1`.
- **Split:** 80% train / 10% validation / 10% test (`random_state=4`).
- **Missing values:** `total_bedrooms` has missing values. They were filled with the **median** of the training set, because the median is robust to outliers.
- **Encoding:** `ocean_proximity` was **one-hot encoded**, and the columns were aligned across the three splits.
- **Scaling:** `StandardScaler`, fitted on the training set only.
- The data was converted to PyTorch tensors and loaded with `DataLoader` (batch size 64, training set shuffled).

### 2. Models
Both models use **MSE loss** and the **Adam** optimizer (learning rate 0.001).

| Model | Architecture | Parameters | Epochs |
|---|---|---|---|
| Simple NN | 13 → **8** → 1 (ReLU) | 121 | 30 + 60 |
| Deep NN | 13 → **64 → 32 → 16** → 1 (ReLU) | 3,521 | 150 |

---

## Results (test set, original dollar scale)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Simple NN | 0.659 | ≈ $66,175 | ≈ $43,981 |
| **Deep NN** | **0.763** | | |

- The simple network learns steadily but levels off early, because its capacity is limited.
- The deep network brings both training and validation loss lower, and the gap between them stays small, so there's no sign of overfitting. It explains about **76% of the variance** in house prices.
- The notebook ends with an **actual vs. predicted** scatter plot for the deep model.

---

## Repository Structure

```
├── notebook.ipynb      # Full notebook: preprocessing, simple NN, deep NN, comparison
├── housing.csv         # California Housing dataset
├── requirements.txt
└── README.md
```

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

Run all cells from top to bottom. The notebook reads `housing.csv` from the same folder. Results can vary slightly between runs, because the network weights are initialized randomly.
