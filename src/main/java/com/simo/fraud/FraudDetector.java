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
import java.util.Random;

/** Ties together feature scaling, the model and a decision threshold. */
public class FraudDetector {

    private final StandardScaler scaler;
    private final LogisticRegression model;
    private double threshold = 0.5;
    private boolean trained;

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
        requireTrained("save");
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
        d.trained = true;
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
        d.trained = true;
        return d;
    }

    /**
     * Chooses the threshold that maximises F1 on a held-out validation set.
     *
     * @throws IllegalArgumentException if the set is empty or lacks fraud or genuine purchases;
     *         the current threshold is kept in that case
     */
    public void calibrate(List<Transaction> validation) {
        requireTrained("calibrate");
        requireBothClasses(validation, "Validation data");
        threshold = Metrics.bestF1Threshold(scores(validation), labels(validation));
    }

    /**
     * Sets the threshold to the highest value that still catches {@code targetRecall} of the fraud
     * in a held-out validation set, keeping false alarms as low as that target allows.
     *
     * @throws IllegalArgumentException if the target isn't in (0, 1] or the set is empty or lacks
     *         fraud or genuine purchases; the current threshold is kept in that case
     */
    public void calibrateForRecall(List<Transaction> validation, double targetRecall) {
        requireTrained("calibrate");
        requireBothClasses(validation, "Validation data");
        threshold = Metrics.thresholdForRecall(scores(validation), labels(validation), targetRecall);
    }

    /**
     * Sets the threshold that costs the least on a held-out validation set, where a missed fraud
     * loses its amount and each false alarm costs {@code reviewCost} (an analyst's time, a call to
     * the customer). Large frauds then count for more than small ones, which F1 ignores.
     *
     * @throws IllegalArgumentException if the review cost isn't a positive number or the set is
     *         empty or lacks fraud or genuine purchases; the current threshold is kept in that case
     */
    public void calibrateForCost(List<Transaction> validation, double reviewCost) {
        requireTrained("calibrate");
        requireBothClasses(validation, "Validation data");
        threshold = Metrics.minCostThreshold(scores(validation), labels(validation), amounts(validation), reviewCost);
    }

    /**
     * Sets the lowest threshold that flags at most {@code maxAlertRate} of a held-out set, for when
     * the review team can only handle a fixed share of transactions. Labels aren't needed, so recent
     * unlabelled traffic works as well as a validation set.
     *
     * @throws IllegalArgumentException if the rate isn't in (0, 1] or the set is empty; the current
     *         threshold is kept in that case
     */
    public void calibrateForAlertRate(List<Transaction> recent, double maxAlertRate) {
        requireTrained("calibrate");
        if (recent == null || recent.isEmpty()) throw new IllegalArgumentException("Transactions to calibrate on are empty");
        threshold = Metrics.thresholdForAlertRate(scores(recent), maxAlertRate);
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

    /** Whether the detector has been trained or loaded and can score transactions. */
    public boolean isTrained() {
        return trained;
    }

    private void requireTrained(String action) {
        if (!trained) {
            throw new IllegalStateException("Can't " + action + " an untrained detector: use FraudDetector.train() or load() first");
        }
    }

    public double score(Transaction t) {
        requireTrained("score with");
        return scoreFeatures(t.features());
    }

    private double scoreFeatures(double[] features) {
        return model.predictProbability(scaler.transform(features));
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
        requireTrained("explain a score from");
        double[] x = scaler.transform(t.features());
        double[] w = model.weights();
        List<Contribution> out = new ArrayList<>();
        for (int j = 0; j < w.length; j++) out.add(new Contribution(Transaction.FEATURES[j], w[j] * x[j]));
        out.sort(Comparator.comparingDouble((Contribution c) -> Math.abs(c.logOdds())).reversed());
        return out;
    }

    /** How much average precision drops when one feature is shuffled; bigger means the model relies on it more. */
    public record Importance(String feature, double drop) { }

    /**
     * Permutation importance on a held-out set: each feature in turn is shuffled across the
     * transactions, breaking its link with the label, and the drop in average precision is
     * averaged over {@code repeats} shuffles. Unlike the raw weights, this shows how much each
     * signal matters for ranking fraud on data the model hasn't seen. A drop near zero means the
     * model barely uses that feature. Results are sorted, most important first.
     *
     * @throws IllegalArgumentException if {@code repeats} is less than 1 or the set is empty or
     *         lacks fraud or genuine purchases
     */
    public List<Importance> permutationImportance(List<Transaction> data, int repeats, long seed) {
        requireTrained("measure feature importance for");
        if (repeats < 1) throw new IllegalArgumentException("repeats must be at least 1, got " + repeats);
        requireBothClasses(data, "Data for feature importance");
        double[][] x = features(data);
        boolean[] y = labels(data);
        double base = Metrics.averagePrecision(scoreRows(x), y);
        Random r = new Random(seed);
        int[] order = new int[x.length];
        List<Importance> out = new ArrayList<>();
        for (int j = 0; j < Transaction.FEATURES.length; j++) {
            double total = 0;
            for (int rep = 0; rep < repeats; rep++) {
                for (int i = 0; i < order.length; i++) order[i] = i;
                for (int i = order.length - 1; i > 0; i--) {
                    int k = r.nextInt(i + 1);
                    int t = order[i]; order[i] = order[k]; order[k] = t;
                }
                double[][] shuffled = new double[x.length][];
                for (int i = 0; i < x.length; i++) {
                    shuffled[i] = x[i].clone();
                    shuffled[i][j] = x[order[i]][j];
                }
                total += base - Metrics.averagePrecision(scoreRows(shuffled), y);
            }
            out.add(new Importance(Transaction.FEATURES[j], total / repeats));
        }
        out.sort(Comparator.comparingDouble(Importance::drop).reversed());
        return out;
    }

    private double[] scoreRows(double[][] x) {
        double[] s = new double[x.length];
        for (int i = 0; i < x.length; i++) s[i] = scoreFeatures(x[i]);
        return s;
    }

    public double threshold() {
        return threshold;
    }

    public double[] weights() {
        requireTrained("read the weights of");
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

    /** Each transaction's amount, which is what a missed fraud loses. */
    static double[] amounts(List<Transaction> data) {
        return data.stream().mapToDouble(Transaction::amount).toArray();
    }
}
