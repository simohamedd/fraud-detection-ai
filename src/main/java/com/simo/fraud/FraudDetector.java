package com.simo.fraud;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Properties;
import java.io.Reader;
import java.io.Writer;
import java.util.Comparator;
import java.util.List;

/** Ties together feature scaling, the model and a decision threshold. */
public class FraudDetector {

    private final StandardScaler scaler;
    private final LogisticRegression model;
    private double threshold = 0.5;

    public FraudDetector() {
        this(new LogisticRegression(0.1, 1e-4, 40, 1));
    }

    FraudDetector(LogisticRegression model) {
        this(new StandardScaler(), model);
    }

    private FraudDetector(StandardScaler scaler, LogisticRegression model) {
        this.scaler = scaler;
        this.model = model;
    }

    /** Saves the trained model (scaling, weights and threshold) as a small text file. */
    public void save(Path file) throws IOException {
        Properties p = new Properties();
        p.setProperty("features", String.join(",", Transaction.FEATURES));
        p.setProperty("mean", join(scaler.mean()));
        p.setProperty("std", join(scaler.std()));
        p.setProperty("weights", join(model.weights()));
        p.setProperty("bias", Double.toString(model.bias()));
        p.setProperty("threshold", Double.toString(threshold));
        try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
            p.store(w, "Fraud detection model");
        }
    }

    /** Loads a model saved with {@link #save(Path)}, ready to score without retraining. */
    public static FraudDetector load(Path file) throws IOException {
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        }
        if (!String.join(",", Transaction.FEATURES).equals(p.getProperty("features"))) {
            throw new IOException("This model was saved with different features and can't be loaded");
        }
        FraudDetector d = new FraudDetector(
                StandardScaler.of(split(p.getProperty("mean")), split(p.getProperty("std"))),
                LogisticRegression.fromWeights(split(p.getProperty("weights")), Double.parseDouble(p.getProperty("bias"))));
        d.threshold = Double.parseDouble(p.getProperty("threshold"));
        return d;
    }

    private static String join(double[] v) {
        return String.join(",", Arrays.stream(v).mapToObj(Double::toString).toList());
    }

    private static double[] split(String s) {
        if (s == null) throw new IllegalArgumentException("Model file is missing a value");
        return Arrays.stream(s.split(",")).mapToDouble(Double::parseDouble).toArray();
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
