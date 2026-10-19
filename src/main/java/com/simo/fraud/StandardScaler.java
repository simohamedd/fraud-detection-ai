package com.simo.fraud;

/** Rescales each feature to mean 0 and standard deviation 1, using statistics from the training set only. */
public class StandardScaler {
    private double[] mean;
    private double[] std;

    public StandardScaler fit(double[][] x) {
        int d = x[0].length;
        mean = new double[d];
        std = new double[d];
        for (double[] row : x) for (int j = 0; j < d; j++) mean[j] += row[j];
        for (int j = 0; j < d; j++) mean[j] /= x.length;
        for (double[] row : x) for (int j = 0; j < d; j++) std[j] += (row[j] - mean[j]) * (row[j] - mean[j]);
        for (int j = 0; j < d; j++) {
            std[j] = Math.sqrt(std[j] / x.length);
            if (std[j] < 1e-12) std[j] = 1;
        }
        return this;
    }

    double[] mean() { return mean.clone(); }
    double[] std() { return std.clone(); }

    static StandardScaler of(double[] mean, double[] std) {
        StandardScaler s = new StandardScaler();
        s.mean = mean.clone();
        s.std = std.clone();
        return s;
    }

    public double[] transform(double[] row) {
        if (mean == null) throw new IllegalStateException("Call fit() first");
        double[] out = new double[row.length];
        for (int j = 0; j < row.length; j++) out[j] = (row[j] - mean[j]) / std[j];
        return out;
    }

    public double[][] transform(double[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) out[i] = transform(x[i]);
        return out;
    }
}
