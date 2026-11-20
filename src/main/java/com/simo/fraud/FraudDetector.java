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

    /**
     * Loads a model saved with {@link #save(Path)}, ready to score without retraining.
     *
     * @throws IOException if the file can't be read, or is incomplete or damaged in a way that
     *         would make the detector crash or stop flagging fraud
     */
    public static FraudDetector load(Path file) throws IOException {
        Properties p = new Properties();
        try (Reader r = Files.newBufferedReader(file, StandardCharsets.UTF_8)) {
            p.load(r);
        }
        if (!String.join(",", Transaction.FEATURES).equals(p.getProperty("features"))) {
            throw new IOException("This model was saved with different features and can't be loaded");
        }
        int n = Transaction.FEATURES.length;
        double[] mean = values(p, "mean", n);
        double[] std = values(p, "std", n);
        for (int j = 0; j < n; j++) {
            if (!(std[j] > 0)) {
                throw new IOException("Model file has a non-positive std for " + Transaction.FEATURES[j] + ": " + std[j]);
            }
        }
        double[] weights = values(p, "weights", n);
        double bias = value(p, "bias");
        double threshold = value(p, "threshold");
        if (threshold < 0 || threshold > 1) {
            throw new IOException("Model file has a threshold outside 0-1: " + threshold);
        }
        FraudDetector d = new FraudDetector(StandardScaler.of(mean, std), LogisticRegression.fromWeights(weights, bias));
        d.threshold = threshold;
        return d;
    }

    private static String join(double[] v) {
        return String.join(",", Arrays.stream(v).mapToObj(Double::toString).toList());
    }

    private static double[] values(Properties p, String key, int expected) throws IOException {
        String raw = required(p, key);
        String[] parts = raw.split(",", -1);
        if (parts.length != expected) {
            throw new IOException("Model file has " + parts.length + " values for '" + key + "', expected " + expected);
        }
        double[] out = new double[expected];
        for (int j = 0; j < expected; j++) out[j] = parse(key + "[" + j + "]", parts[j]);
        return out;
    }

    private static double value(Properties p, String key) throws IOException {
        return parse(key, required(p, key));
    }

    private static String required(Properties p, String key) throws IOException {
        String v = p.getProperty(key);
        if (v == null || v.isBlank()) throw new IOException("Model file is missing '" + key + "'");
        return v;
    }

    private static double parse(String name, String text) throws IOException {
        double v;
        try {
            v = Double.parseDouble(text.trim());
        } catch (NumberFormatException e) {
            throw new IOException("Model file has an invalid number for " + name + ": '" + text + "'", e);
        }
        if (!Double.isFinite(v)) throw new IOException("Model file has a non-finite value for " + name + ": " + v);
        return v;
    }

    /**
     * Trains a detector on labelled transactions.
     *
     * @throws IllegalArgumentException if the data is empty or doesn't contain both fraud and
     *         genuine purchases, since there would be nothing for the model to separate
     */
    public static FraudDetector train(List<Transaction> data, long seed) {
        requireBothClasses(data, "Training data");
        double[][] x = features(data);
        boolean[] y = labels(data);
        FraudDetector d = new FraudDetector(new LogisticRegression(0.1, 1e-4, 40, LogisticRegression.balancedWeight(y)));
        d.scaler.fit(x);
        d.model.fit(d.scaler.transform(x), y, seed);
        return d;
    }

    /**
     * Chooses the threshold that maximises F1 on a held-out validation set.
     *
     * @throws IllegalArgumentException if the set is empty or lacks fraud or genuine purchases;
     *         the current threshold is kept in that case
     */
    public void calibrate(List<Transaction> validation) {
        requireBothClasses(validation, "Validation data");
        threshold = Metrics.bestF1Threshold(scores(validation), labels(validation));
    }

    private static void requireBothClasses(List<Transaction> data, String what) {
        if (data == null || data.isEmpty()) throw new IllegalArgumentException(what + " is empty");
        long fraud = data.stream().filter(Transaction::fraud).count();
        if (fraud == 0) {
            throw new IllegalArgumentException(what + " has no fraud examples among " + data.size() + " transactions");
        }
        if (fraud == data.size()) {
            throw new IllegalArgumentException(what + " has no genuine purchases among " + data.size() + " transactions");
        }
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
