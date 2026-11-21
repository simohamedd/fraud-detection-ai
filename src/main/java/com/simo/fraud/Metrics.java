package com.simo.fraud;

import java.util.Arrays;
import java.util.Comparator;

/** Evaluation for a binary classifier: confusion matrix, precision, recall, F1, ROC AUC and average precision. */
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

    /**
     * Average precision: the area under the precision-recall curve.
     *
     * <p>Scores are taken from highest to lowest, and each fraud caught adds its share of recall
     * weighted by the precision at that point. Tied scores are flagged together, as they would be
     * by a threshold. Unlike ROC AUC this ignores how well the many genuine purchases are ranked
     * among themselves, so it is the more honest number when fraud is rare: a random model scores
     * about the fraud rate, a perfect one 1. Returns NaN when there are no fraud examples.
     */
    public static double averagePrecision(double[] scores, boolean[] y) {
        if (scores.length != y.length) throw new IllegalArgumentException("scores and labels differ in length");
        int totalPos = 0;
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(scores[i])) throw new IllegalArgumentException("score " + i + " is NaN");
            if (y[i]) totalPos++;
        }
        if (totalPos == 0) return Double.NaN;

        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> scores[i]).reversed());

        double ap = 0;
        int tp = 0, flagged = 0;
        int i = 0;
        while (i < order.length) {
            double s = scores[order[i]];
            int newPos = 0;
            while (i < order.length && scores[order[i]] == s) {
                if (y[order[i]]) newPos++;
                flagged++;
                i++;
            }
            tp += newPos;
            ap += (double) newPos / totalPos * tp / flagged;
        }
        return ap;
    }

    /**
     * The threshold with the best F1 score, searched over the observed scores.
     *
     * <p>Every distinct score is tried as a cut-off in one pass over the sorted scores, so the
     * result is exact rather than limited to a 0.01 grid. The threshold returned sits halfway
     * between the chosen score and the next lower one, leaving a margin on both sides.
     * Returns 0.5 when there are no fraud examples to learn from.
     */
    public static double bestF1Threshold(double[] scores, boolean[] y) {
        if (scores.length != y.length) throw new IllegalArgumentException("scores and labels differ in length");
        int totalPos = 0;
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(scores[i])) throw new IllegalArgumentException("score " + i + " is NaN");
            if (y[i]) totalPos++;
        }
        if (totalPos == 0) return 0.5;

        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> scores[i]).reversed());

        double best = 0.5, bestF1 = -1;
        int tp = 0, fp = 0;
        int i = 0;
        while (i < order.length) {
            double s = scores[order[i]];
            // Flag every example scoring at least s, ties included.
            while (i < order.length && scores[order[i]] == s) {
                if (y[order[i]]) tp++; else fp++;
                i++;
            }
            double f1 = 2.0 * tp / (2.0 * tp + fp + (totalPos - tp));
            if (f1 > bestF1) {
                bestF1 = f1;
                best = i < order.length ? (s + scores[order[i]]) / 2 : s;
            }
        }
        return best;
    }

    /**
     * The highest threshold that still catches at least {@code targetRecall} of the fraud.
     *
     * <p>Fraud teams often work to a recall target ("catch 90% of fraud") rather than the best F1,
     * and then want as few false alarms as that allows. Scores are walked from highest to lowest,
     * flagging tied scores together, until the target is met. Like {@link #bestF1Threshold}, the
     * threshold returned sits halfway between that score and the next lower one.
     *
     * @throws IllegalArgumentException if the target isn't in (0, 1], the inputs differ in length,
     *         a score is NaN, or there are no fraud examples to measure recall on
     */
    public static double thresholdForRecall(double[] scores, boolean[] y, double targetRecall) {
        if (!(targetRecall > 0 && targetRecall <= 1)) {
            throw new IllegalArgumentException("target recall must be in (0, 1], got " + targetRecall);
        }
        if (scores.length != y.length) throw new IllegalArgumentException("scores and labels differ in length");
        int totalPos = 0;
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(scores[i])) throw new IllegalArgumentException("score " + i + " is NaN");
            if (y[i]) totalPos++;
        }
        if (totalPos == 0) throw new IllegalArgumentException("no fraud examples to measure recall on");

        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> scores[i]).reversed());

        // Compare counts rather than ratios so a target like 0.9 of 10 isn't missed by rounding.
        int needed = (int) Math.ceil(targetRecall * totalPos - 1e-9);
        int tp = 0;
        int i = 0;
        while (true) {
            double s = scores[order[i]];
            while (i < order.length && scores[order[i]] == s) {
                if (y[order[i]]) tp++;
                i++;
            }
            if (tp >= needed) return i < order.length ? (s + scores[order[i]]) / 2 : s;
        }
    }
}
