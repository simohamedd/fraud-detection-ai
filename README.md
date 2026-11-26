# Fraud Detection AI

[![Build and test](https://github.com/simohamedd/fraud-detection-ai/actions/workflows/build.yml/badge.svg)](https://github.com/simohamedd/fraud-detection-ai/actions/workflows/build.yml)

Detects fraudulent card transactions with a logistic regression model written from scratch in Java: no ML libraries, just the maths.

## How it works
1. **Data:** `SyntheticData` generates realistic card transactions where about 3% are fraud. Fraud tends to be larger, at night, far from home, online, at risky merchants and in bursts, with plenty of overlap.
2. **Features:** log amount, night-time flag, log distance from home, merchant risk, online flag, purchases in the last hour.
3. **Scaling:** `StandardScaler` puts every feature on the same scale, fitted on training data only.
4. **Model:** `LogisticRegression` trained by mini-batch gradient descent with L2 regularisation. Fraud examples are weighted up so the rare class isn't ignored. That weighting would also inflate every probability (it multiplies the learned odds by about 32), so the bias is corrected by log(weight) after training: a transaction scored at 20% risk really is fraud about one time in five.
5. **Threshold:** chosen on a separate validation set to maximise F1. Both training and calibration need at least one fraud and one genuine purchase, and throw an `IllegalArgumentException` otherwise; a failed calibration keeps the previous threshold. If you'd rather work to a recall target, `calibrateForRecall(validation, 0.9)` picks the highest threshold that still catches 90% of the validation fraud, so there are as few false alarms as that target allows.
6. **Explanations:** `explain()` breaks each score into how much every signal raised or lowered the risk, so an analyst can see why a payment was blocked.
7. **Saving:** `save()` and `load()` store the trained model as a small text file, so it can score new transactions without retraining.
8. **Evaluation:** precision, recall, F1, confusion matrix, ROC AUC and average precision on a test set the model never saw. With only 3% fraud, average precision (the area under the precision-recall curve) is the stricter measure: a model that guesses at random gets about 0.03 rather than ROC AUC's 0.5. Both of those only judge the ranking, so log loss is reported too: it checks the risk percentages themselves and punishes confident mistakes.

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
