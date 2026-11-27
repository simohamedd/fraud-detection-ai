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
8. **Evaluation:** precision, recall, F1, confusion matrix, ROC AUC and average precision on a test set the model never saw. With only 3% fraud, average precision (the area under the precision-recall curve) is the stricter measure: a model that guesses at random gets about 0.03 rather than ROC AUC's 0.5. Both of those only judge the ranking, so log loss is reported too: it checks the risk percentages themselves and punishes confident mistakes. To see where the percentages are off, `Metrics.calibrationTable` groups test purchases by predicted risk and compares the average prediction in each group with the share that really was fraud, and `Metrics.expectedCalibrationError` sums up the gaps (weighted by group size) in one number. The demo prints both; on the demo data the error is well under one percentage point.

## Choosing a threshold by cost
F1 and recall targets count every fraud the same, but letting a £900 fraud through is much worse than missing a £9 one. `calibrateForCost(validation, 5)` picks the threshold with the lowest total cost on the validation set, where each missed fraud loses its amount and each false alarm costs £5 to review. `Metrics.minCostThreshold` and `Metrics.cost` do the same for any scores, labels and per-transaction costs. On the demo data this roughly halves the cost of the F1 threshold, by accepting more false alarms in exchange for missing fewer of the large frauds.

## Working to an alert budget
A review team can only look at so many alerts. `calibrateForAlertRate(recent, 0.01)` picks the lowest threshold that flags at most 1% of the given transactions, and `Metrics.thresholdForAlertRate` does the same for any scores. Labels aren't needed, so it can be recalibrated on recent traffic. The budget is exact on the transactions it was set on and approximate on new ones (the demo flags 5.4% of the test set against a 5% budget), so leave a little headroom. Tied scores are flagged together, so a tie that would go over the budget is left out and slightly fewer alerts than the budget may be raised. The demo prints what recall and precision a 5% budget gets on the test set; a budget below the fraud rate (about 3%) can only catch part of the fraud however good the model is.

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
