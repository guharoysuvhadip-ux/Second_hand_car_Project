# Second_hand_car_Price_Prediction_Project

A data analysis and machine-learning project that explores a used-car sales dataset and predicts a car's **selling price** from its manufacturer, model, engine size, fuel type, year of manufacture and mileage.

Built in Python (Google Colab) with pandas, NumPy, Matplotlib, Seaborn and scikit-learn.

---

## Project Overview

Pricing a used car is hard: age, mileage, engine and brand all pull the price in different directions. This project studies those relationships and builds a baseline model that estimates a fair price for a car.

### Project Objectives

- Clean and validate the raw car sales data.
- Explore how mileage, age, engine size and fuel type affect price.
- Train a regression model to predict price and measure its accuracy.
- Use the model to price a new, unseen car.

---

## Dataset

| Item | Detail |
|---|---|
| File | `car_sales_data.csv` |
| Rows | 50,000 cars |
| Columns | 7 |
| Missing values | None |
| Duplicate rows | 12 |

| Column | Type | Description |
|---|---|---|
| `Manufacturer` | Text | Brand (e.g. Ford, Toyota, VW, Porsche) |
| `Model` | Text | Car model (e.g. Fiesta, RAV4, Polo) |
| `Engine size` | Number | Engine size in litres (1.0 to 5.0) |
| `Fuel type` | Text | Petrol, Diesel, Hybrid, etc. |
| `Year of manufacture` | Integer | 1984 to 2022 |
| `Mileage` | Integer | 630 to 453,537 |
| `Price` | Integer | Target variable, 76 to 168,081 (average about 13,829) |

---

## Workflow

### 1. Data Inspection
- Checked shape, data types, summary statistics and missing values.

### 2. Cleaning & Feature Engineering
- Confirmed integer types for year and mileage.
- Created a `Car Age` column (2026 minus year of manufacture).
- Found and dropped 12 duplicate rows (50,000 → 49,988).

### 3. Outlier Removal
- Box plots for year of manufacture and mileage.
- Applied the IQR rule (1.5 × IQR) to `Mileage` and `Engine size`.
- Rows: **50,000 → 45,700** (4,300 removed).

### 4. Exploratory Data Analysis
- Mileage distribution (histogram with KDE).
- Scatter plots: mileage vs price, year vs price, engine size vs price.
- Average price by engine size, model and year.
- Bar plots of price by engine size and by year.
- Correlation heatmap of the numeric features.

### 5. Modeling
- **Features:** `Manufacturer`, `Model`, `Engine size`, `Fuel type`, `Year of manufacture`, `Mileage`
- **Target:** `Price`
- **Split:** 80% train (40,000) / 20% test (10,000), `random_state=42`
- **Pipeline:** one-hot encoding of `Manufacturer`, `Model` and `Fuel type`, followed by Linear Regression
- A coefficient chart shows the features with the biggest positive and negative effect on price.

### 6. Price Prediction Example
Predicting a new car:

| Manufacturer | Model | Engine | Fuel | Year | Mileage |
|---|---|---|---|---|---|
| Toyota | RAV4 | 1.8 | Hybrid | 2018 | 50,000 |

**Predicted price: about 32,160**

---

## Results

| Metric | Value |
|---|---|
| R² score | **0.68** |
| Mean absolute error | **≈ 5,978** |

The model explains about 68% of the variation in car prices on unseen data. On average its predictions are off by roughly 6,000.

---

## Limitations & Next Steps

- **Cleaning is not carried through to the model.** Several cells re-read the raw CSV, so the duplicate removal, outlier removal and `Car Age` column are lost before training (the model is trained on all 50,000 original rows). Keep one working `df` and pass it through every step.
- **Linear Regression** is a simple baseline. Price falls off non-linearly with age and mileage, so try Random Forest, Gradient Boosting or XGBoost, or predict `log(Price)`.
- **Use `Car Age`** as a feature instead of (or alongside) `Year of manufacture`.
- Add interaction features (e.g. age × mileage) and cross-validation.
- A few code fixes: the typo `file_mane`, `'price'` should be `'Price'` in the mileage scatter, `num_df` is used in the heatmap without being defined, and the pie chart is titled "brands" but plots year.
- The currency isn't stated in the dataset; the notebook prints `$`.

---

## How to Run

1. Open the notebook in Google Colab, or locally with Jupyter.
2. Upload `car_sales_data.csv` when prompted (or change the file path).
3. Run all cells in order.

**Requirements**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

The notebook uses `google.colab.files.upload()`. Remove it and read the CSV directly if running locally.

---

## Repository Structure

```
├── Second_hand_car_Project.ipynb   # main notebook
├── car_sales_data.csv              # dataset
└── README.md
```

## Tech Stack

Python · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · Google Colab
