# Student Performance Prediction

Predicting secondary school students' exam scores from demographic, family and lifestyle characteristics, by comparing five supervised learning models with 10-fold cross-validation in R.

[View the full report](report.html) · [Analysis script](predict_score.R)

## Overview

This project was a supervised learning competition in a university machine learning course. The goal was to predict a normalised exam `score` for 79 unseen students, evaluated by **mean squared error (MSE)**.

The test set has no labels, so all models were evaluated with **10-fold cross-validation** on the 316 training students. Every model used the same folds, which keeps the comparison fair and free of data leakage.

## Data

| | Students | Predictors |
|---|---|---|
| Training set (`data/train.rds`) | 316 | 30 + `score` |
| Test set (`data/test.rds`) | 79 | 30 |

The data describe students from two Portuguese secondary schools and are based on the [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance) (Cortez & Silva, 2008). The predictors include 17 categorical variables (e.g. school, sex, parents' jobs, plans for higher education) and 13 numeric or ordinal variables (e.g. age, study time, past failures, alcohol use, absences). There are no missing values.

## Approach

**Pre-processing**
- Log-transformed the right-skewed `absences` variable: `log(absences + 1)`
- One-hot encoded the categorical variables for KNN and XGBoost (`model.matrix`)
- Standardised predictors for KNN, using only the training part of each fold to avoid leakage

**Models compared**

| Model | Notes |
|---|---|
| Baseline | Always predicts the mean training score |
| Linear regression | All predictors, no tuning |
| K-nearest neighbours | Number of neighbours tuned by CV (k = 15) |
| Regression tree | `rpart` defaults |
| Random forest | `randomForest` defaults |
| XGBoost | Grid search over learning rate, depth, leaf size and column sampling; early stopping picks the number of trees |

## Results

10-fold cross-validated MSE (lower is better):

| Model | CV MSE |
|---|---|
| **XGBoost** | **0.745** |
| **Random forest** | **0.763** |
| Linear regression | 0.831 |
| KNN (k = 15) | 0.886 |
| Regression tree | 0.887 |
| Baseline (mean) | 0.983 |

- Both tree ensembles clearly beat the other models and the baseline.
- XGBoost had the lowest CV error, but the gap to the random forest is small. XGBoost's estimate is also slightly optimistic, because its hyperparameters were tuned on the same data.
- The **random forest** was selected as the final model. It is nearly as accurate, needs no tuning, and its CV estimate is unbiased. It reduces the MSE by about 22% compared to the baseline.
- The number of **past class failures** was the strongest predictor of the score across models.

## Repository structure

```
├── predict_score.R       # Full analysis: EDA, models, cross-validation, predictions
├── report.html           # Rendered report with all outputs and figures
├── predictions.rds       # Final predictions for the 79 test students
└── data/
    ├── train.rds         # Training data with score
    └── test.rds          # Test data without score
```

## How to run

Requirements: R (≥ 4.1). Install the packages once:

```r
install.packages(c("tidyverse", "rpart", "rpart.plot", "randomForest", "FNN", "xgboost"))
```

Then, from the repository root, either run the analysis:

```bash
Rscript predict_score.R
```

or render it to the HTML report (needs `rmarkdown`):

```r
rmarkdown::render("predict_score.R", output_file = "report.html")
```

Both write `predictions.rds`. All random steps use fixed seeds, so the results are reproducible.

## Tech stack

R · tidyverse · ggplot2 · rpart · randomForest · FNN · xgboost · R Markdown
