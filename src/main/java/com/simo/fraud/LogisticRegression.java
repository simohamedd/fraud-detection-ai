package com.simo.fraud;

import java.util.Random;

/**
 * Binary logistic regression trained with mini-batch gradient descent and L2 regularisation.
 * Positive examples can be weighted more heavily, which matters when fraud is rare.
 */
public class LogisticRegression {

    private final double learningRate;
    private final double l2;
    private final int epochs;
    private final double positiveWeight;
    private double[] w;
    private double b;

    public LogisticRegression(double learningRate, double l2, int epochs, double positiveWeight) {
        this.learningRate = learningRate;
        this.l2 = l2;
        this.epochs = epochs;
        this.positiveWeight = positiveWeight;
    }

    /** Positive weight = negatives / positives, so both classes count equally in the loss. */
    public static double balancedWeight(boolean[] y) {
        int pos = 0;
        for (boolean v : y) if (v) pos++;
        return pos == 0 ? 1 : (double) (y.length - pos) / pos;
    }

    public LogisticRegression fit(double[][] x, boolean[] y, long seed) {
        int n = x.length, d = x[0].length, batch = 64;
        w = new double[d];
        b = 0;
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) idx[i] = i;
        Random r = new Random(seed);
        for (int epoch = 0; epoch < epochs; epoch++) {
            shuffle(idx, r);
            for (int start = 0; start < n; start += batch) {
                int end = Math.min(n, start + batch);
                double[] gw = new double[d];
                double gb = 0, total = 0;
                for (int k = start; k < end; k++) {
                    int i = idx[k];
                    double weight = y[i] ? positiveWeight : 1;
                    double err = (sigmoid(dot(x[i])) - (y[i] ? 1 : 0)) * weight;
                    for (int j = 0; j < d; j++) gw[j] += err * x[i][j];
                    gb += err;
                    total += weight;
                }
                for (int j = 0; j < d; j++) w[j] -= learningRate * (gw[j] / total + l2 * w[j]);
                b -= learningRate * gb / total;
            }
        }
        return this;
    }

    /** Restores a model trained earlier. */
    static LogisticRegression fromWeights(double[] weights, double bias) {
        LogisticRegression m = new LogisticRegression(0, 0, 0, 1);
        m.w = weights.clone();
        m.b = bias;
        return m;
    }

    /** Probability that the example is positive (fraud). */
    public double predictProbability(double[] row) {
        if (w == null) throw new IllegalStateException("Call fit() first");
        return sigmoid(dot(row));
    }

    public double[] weights() {
        return w.clone();
    }

    public double bias() {
        return b;
    }

    private double dot(double[] row) {
        double s = b;
        for (int j = 0; j < w.length; j++) s += w[j] * row[j];
        return s;
    }

    static double sigmoid(double z) {
        if (z >= 0) return 1 / (1 + Math.exp(-z));
        double e = Math.exp(z);
        return e / (1 + e);
    }

    private static void shuffle(int[] a, Random r) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i]; a[i] = a[j]; a[j] = t;
        }
    }
}
