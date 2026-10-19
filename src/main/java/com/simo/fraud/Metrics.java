package com.simo.fraud;

import java.util.Arrays;
import java.util.Comparator;

/** Evaluation for a binary classifier: confusion matrix, precision, recall, F1 and ROC AUC. */
public final class Metrics {

    public record Confusion(int truePositives, int falsePositives, int trueNegatives, int falseNegatives) {
        public double precision() { return ratio(truePositives, truePositives + falsePositives); }
        public double recall() { return ratio(truePositives, truePositives + falseNegatives); }
        public double f1() {
            double p = precision(), r = recall();
            return p + r == 0 ? 0 : 2 * p * r / (p + r);
        }
        public double accuracy() {
            return ratio(truePositives + trueNegatives, truePositives + falsePositives + trueNegatives + falseNegatives);
        }
        private static double ratio(int a, int b) { return b == 0 ? 0 : (double) a / b; }
    }

    private Metrics() {
    }

    public static Confusion confusion(double[] scores, boolean[] y, double threshold) {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i = 0; i < y.length; i++) {
            boolean predicted = scores[i] >= threshold;
            if (predicted && y[i]) tp++;
            else if (predicted) fp++;
            else if (y[i]) fn++;
            else tn++;
        }
        return new Confusion(tp, fp, tn, fn);
    }

    /** Area under the ROC curve: the chance a random fraud scores higher than a random genuine purchase. */
    public static double rocAuc(double[] scores, boolean[] y) {
        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble(i -> scores[i]));
        double rankSumPos = 0;
        long pos = 0;
        int i = 0;
        while (i < order.length) {
            int j = i;
            while (j + 1 < order.length && scores[order[j + 1]] == scores[order[i]]) j++;
            double avgRank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++) if (y[order[k]]) { rankSumPos += avgRank; pos++; }
            i = j + 1;
        }
        long neg = scores.length - pos;
        if (pos == 0 || neg == 0) return Double.NaN;
        return (rankSumPos - pos * (pos + 1) / 2.0) / (pos * neg);
    }

    /** The threshold with the best F1 score, searched over the observed scores. */
    public static double bestF1Threshold(double[] scores, boolean[] y) {
        double best = 0.5, bestF1 = -1;
        for (int step = 1; step < 100; step++) {
            double t = step / 100.0;
            double f1 = confusion(scores, y, t).f1();
            if (f1 > bestF1) { bestF1 = f1; best = t; }
        }
        return best;
    }
}
