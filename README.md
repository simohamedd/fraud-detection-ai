# Fraud Detection AI

[![Build and test](https://github.com/simohamedd/fraud-detection-ai/actions/workflows/build.yml/badge.svg)](https://github.com/simohamedd/fraud-detection-ai/actions/workflows/build.yml)

Detects fraudulent card transactions with a logistic regression model written from scratch in Java: no ML libraries, just the maths.

## How it works
1. **Data:** `SyntheticData` generates realistic card transactions where about 3% are fraud. Fraud tends to be larger, at night, far from home, online, at risky merchants and in bursts, with plenty of overlap.
2. **Features:** log amount, night-time flag, log distance from home, merchant risk, online flag, purchases in the last hour.
3. **Scaling:** `StandardScaler` puts every feature on the same scale, fitted on training data only.
4. **Model:** `LogisticRegression` trained by mini-batch gradient descent with L2 regularisation. Fraud examples are weighted up so the rare class isn't ignored.
5. **Threshold:** chosen on a separate validation set to maximise F1.
6. **Explanations:** `explain()` breaks each score into how much every signal raised or lowered the risk, so an analyst can see why a payment was blocked.
7. **Saving:** `save()` and `load()` store the trained model as a small text file, so it can score new transactions without retraining.
8. **Evaluation:** precision, recall, F1, confusion matrix and ROC AUC on a test set the model never saw.

## Run
```bash
mvn -q compile exec:java
```
Prints the test results, the weight the model gives each signal, and scores two example purchases.

## Test
```bash
mvn test
```

## Licence
MIT. See [LICENSE](LICENSE).
