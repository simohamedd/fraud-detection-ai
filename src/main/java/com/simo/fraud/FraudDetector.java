package com.simo.fraud;

import java.util.ArrayList;
import java.util.Comparator;
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

    /** How much one feature pushed the score up (positive) or down (negative), in log-odds. */
    public record Contribution(String feature, double logOdds) { }

    /** Breaks a score down into each feature's contribution, biggest effect first. */
    public List<Contribution> explain(Transaction t) {
        double[] x = scaler.transform(t.features());
        double[] w = model.weights();
        List<Contribution> out = new ArrayList<>();
        for (int j = 0; j < w.length; j++) out.add(new Contribution(Transaction.FEATURES[j], w[j] * x[j]));
        out.sort(Comparator.comparingDouble((Contribution c) -> Math.abs(c.logOdds())).reversed());
        return out;
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
