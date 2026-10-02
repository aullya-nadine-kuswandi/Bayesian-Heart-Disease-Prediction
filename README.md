# Bayesian Logistic Regression for Heart Disease Prediction

A comparison of **frequentist logistic regression** and **Bayesian logistic regression** with two different Gaussian priors for predicting heart disease on the Heart Disease UCI dataset. Bayesian inference is done with MCMC in JAGS, and models are compared using classification metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC) and Bayesian criteria (DIC, LOOIC, WAIC).

## Key Findings

- **Bayesian Model 2** (Normal prior, variance 1) gives the best overall balance: highest accuracy (0.8500), precision (0.8529), and F1-score (0.8657), and the lowest DIC, LOOIC, and WAIC.
- Both Bayesian models reach higher **recall** (0.8788) than the frequentist model (0.8485).
- The frequentist model has a marginally higher ROC-AUC (0.9091), but the gap is small.
- The same eight predictors (`sex`, `cp`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`) come out as significant (frequentist) or credible (Bayesian) across all three models.
- MCMC converged cleanly for both Bayesian models (Gelman-Rubin PSRF = 1 for all parameters).

## Dataset

[Heart Disease dataset](https://www.kaggle.com/johnsmith88/heart-disease-dataset) (UCI, via Kaggle): **303 patients**, 13 clinical features, and a binary target (`1` = heart disease, `0` = no heart disease; 165 positive, 138 negative).

| Feature | Description |
|---------|-------------|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dL) |
| `fbs` | Fasting blood sugar > 120 mg/dL (1 = yes, 0 = no) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | ST depression induced by exercise relative to rest |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels (0–3) colored by fluoroscopy |
| `thal` | Thalassemia status |

> The dataset file `heart-disease.csv` is not included in this repository. Download it from Kaggle and place it in the project root.

## Methodology

### Preprocessing

- 80/20 train/test split (`set.seed(123)`, `caret::createDataPartition`)
- Numeric features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are **robust-scaled** with `(x − median) / IQR`, using parameters computed on the training set only
- Design matrix built with `model.matrix`

### Models

| Model | Prior on coefficients | Notes |
|-------|----------------------|-------|
| Frequentist | None | `glm(family = binomial)` |
| Bayesian Model 1 | `β ~ Normal(0, τ²=2)` (JAGS: `dnorm(0, 0.5)`) | Weaker shrinkage |
| Bayesian Model 2 | `β ~ Normal(0, τ²=1)` (JAGS: `dnorm(0, 1)`) | Stronger shrinkage |

JAGS parameterizes the normal by **precision**, so `dnorm(0, 0.5)` is a prior with variance 2.

### MCMC Configuration

| Setting | Value |
|---------|-------|
| Chains | 3 |
| Burn-in | 10,000 iterations |
| Iterations per chain | 50,000 (thinning = 5) |
| Convergence checks | Trace plots, Gelman-Rubin PSRF |
| Posterior predictive check | Observed vs. simulated proportions in 10 probability bins |

### Classification Threshold

Predictions use the positive-class proportion in the data (165 / 303 ≈ **0.5446**) as the probability cutoff instead of 0.5. Bayesian predictions on the test set use the posterior mean of the coefficients.

## Results

Evaluated on the held-out test set.

### Predictive Performance

| Metric | Frequentist | Bayesian Model 1 `N(0, 2)` | Bayesian Model 2 `N(0, 1)` |
|--------|:-----------:|:--------------------------:|:--------------------------:|
| Accuracy | 0.8167 | 0.8333 | **0.8500** |
| Precision | 0.8235 | 0.8286 | **0.8529** |
| Recall | 0.8485 | **0.8788** | **0.8788** |
| F1-Score | 0.8358 | 0.8529 | **0.8657** |
| ROC-AUC | **0.9091** | 0.9057 | 0.9035 |
| PR-AUC | 0.9108 | **0.9217** | 0.9152 |

### Bayesian Model Comparison (lower is better)

| Criterion | Bayesian Model 1 | Bayesian Model 2 |
|-----------|:----------------:|:----------------:|
| DIC | 193.7 | **193.5** |
| LOOIC | 195.80 | **194.57** |
| WAIC | 195.54 | **194.38** |

### Which Predictors Matter?

Based on 95% credible intervals (Bayesian) and p-values (frequentist), results are consistent across models:

| Effect | Predictors |
|--------|-----------|
| Positive association with heart disease | `cp`, `thalach`, `slope` |
| Negative association with heart disease | `sex`, `exang`, `oldpeak`, `ca`, `thal` |
| No clear evidence (interval includes zero) | `age`, `trestbps`, `chol`, `fbs`, `restecg` |

## Repository Structure

```
.
├── Bayesian_Heart_Disease_Prediction.Rmd   # Full analysis: data prep, models, diagnostics, evaluation
├── heart-disease.csv                       # Dataset (download from Kaggle)
└── README.md
```

## Getting Started

### Prerequisites

- **R** (4.0+)
- **[JAGS](https://mcmc-jags.sourceforge.io/)** installed on your system (required by `rjags`)

### Install R Packages

```r
install.packages(c(
  "tidyverse", "rjags", "caret", "loo", "ROCR", "coda"
))
```

### Run the Analysis

1. Place `heart-disease.csv` in the same folder as the `.Rmd` file.
2. Open `Bayesian_Heart_Disease_Prediction.Rmd` in RStudio and click **Knit**, or run:

```r
rmarkdown::render("Bayesian_Heart_Disease_Prediction.Rmd")
```

Sampling both Bayesian models takes a few minutes, and the posterior predictive check loops over all posterior draws, so rendering can be slow.

## Limitations

- Small dataset (303 patients), and therefore a small test set, so differences between models are modest and may not generalize.
- Only logistic regression is considered; nonlinear relationships are not captured.
- Only two prior specifications are compared, and results may be sensitive to prior choice.
- MCMC is more computationally expensive than frequentist estimation.

## Future Work

- Larger and more diverse datasets
- Bayesian hierarchical or nonlinear models
- Hamiltonian Monte Carlo or variational inference
- A broader sensitivity analysis over priors

## Tech Stack

R, JAGS (`rjags`), `coda`, `loo`, `caret`, `ROCR`, `tidyverse` / `ggplot2`

## Authors

Aullya Nadine Kuswandi, Ellyn Auria, Felice Yang, Jaiden Ananta Halim, Data Science, Binus University
