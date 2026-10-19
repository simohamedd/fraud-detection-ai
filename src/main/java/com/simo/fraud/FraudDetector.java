package com.simo.fraud;

import java.util.List;

/** Ties together feature scaling, the model and a decision threshold. */
public class FraudDetector {

    private final StandardScaler scaler = new StandardScaler();
    private final LogisticRegression model;
    private double threshold = 0.5;

    public FraudDetector() {
        this(new LogisticRegression(0.1, 1e-4, 40, 1));
    }

    FraudDetector(LogisticRegression model) {
        this.model = model;
    }

    public static FraudDetector train(List<Transaction> data, long seed) {
        double[][] x = features(data);
        boolean[] y = labels(data);
        FraudDetector d = new FraudDetector(new LogisticRegression(0.1, 1e-4, 40, LogisticRegression.balancedWeight(y)));
        d.scaler.fit(x);
        d.model.fit(d.scaler.transform(x), y, seed);
        return d;
    }

    /** Chooses the threshold that maximises F1 on a held-out validation set. */
    public void calibrate(List<Transaction> validation) {
        threshold = Metrics.bestF1Threshold(scores(validation), labels(validation));
    }

    public double score(Transaction t) {
        return model.predictProbability(scaler.transform(t.features()));
    }

    public boolean isFraud(Transaction t) {
        return score(t) >= threshold;
    }

    public double[] scores(List<Transaction> data) {
        return data.stream().mapToDouble(this::score).toArray();
    }

    public double threshold() {
        return threshold;
    }

    public double[] weights() {
        return model.weights();
    }

    static double[][] features(List<Transaction> data) {
        return data.stream().map(Transaction::features).toArray(double[][]::new);
    }

    static boolean[] labels(List<Transaction> data) {
        boolean[] y = new boolean[data.size()];
        for (int i = 0; i < y.length; i++) y[i] = data.get(i).fraud();
        return y;
    }
}
