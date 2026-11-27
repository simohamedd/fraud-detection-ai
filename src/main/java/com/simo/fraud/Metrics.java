package com.simo.fraud;

import java.util.Arrays;
import java.util.Comparator;

/** Evaluation for a binary classifier: confusion matrix, precision, recall, F1, ROC AUC, average precision and log loss, plus ways to choose a threshold. */
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

    /**
     * Area under the ROC curve: the chance a random fraud scores higher than a random genuine purchase.
     * Tied scores count as half. Returns NaN when either class is missing.
     *
     * @throws IllegalArgumentException if the inputs differ in length or a score is NaN, since a NaN
     *         would otherwise be ranked above every real score and quietly inflate the result
     */
    public static double rocAuc(double[] scores, boolean[] y) {
        if (scores.length != y.length) throw new IllegalArgumentException("scores and labels differ in length");
        for (int i = 0; i < scores.length; i++) {
            if (Double.isNaN(scores[i])) throw new IllegalArgumentException("score " + i + " is NaN");
        }
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
     * Log loss (binary cross-entropy): the average of -log(probability given to the true outcome).
     *
     * <p>ROC AUC and average precision only look at how scores are ranked, so a model that says
     * 51% for every fraud and 49% for every genuine purchase can look perfect on both. Log loss
     * checks the probabilities themselves and punishes confident mistakes hard, which matters
     * when the risk percentage is shown to an analyst. Lower is better: 0 is perfect, and always
     * predicting the fraud rate p gives the entropy of the labels, about 0.135 for 3% fraud.
     * Probabilities are clipped to [1e-15, 1 - 1e-15] so one certain mistake gives a large but
     * finite loss.
     *
     * @throws IllegalArgumentException if the inputs are empty or differ in length, or a score is
     *         NaN or outside 0-1
     */
    public static double logLoss(double[] scores, boolean[] y) {
        if (scores.length != y.length) throw new IllegalArgumentException("scores and labels differ in length");
        if (scores.length == 0) throw new IllegalArgumentException("no scores to evaluate");
        final double eps = 1e-15;
        double total = 0;
        for (int i = 0; i < scores.length; i++) {
            double s = scores[i];
            if (!(s >= 0 && s <= 1)) throw new IllegalArgumentException("score " + i + " is not a probability: " + s);
            double p = Math.min(Math.max(y[i] ? s : 1 - s, eps), 1 - eps);
            total -= Math.log(p);
        }
        return total / scores.length;
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

    /**
     * The lowest threshold that flags at most {@code maxAlertRate} of the examples.
     *
     * <p>A review team can only look at so many alerts a day, so the useful question is often
     * "which threshold keeps us within 1% of transactions?". Scores are walked from highest to
     * lowest and tied scores are flagged together, so a group that would push the count over the
     * budget is left out entirely and the rate flagged can come in under the target. Like
     * {@link #bestF1Threshold}, the threshold returned sits halfway between the last score flagged
     * and the next lower one. If even the top score's group is over budget, the threshold sits
     * above every score and nothing is flagged.
     *
     * @throws IllegalArgumentException if the rate isn't in (0, 1], there are no scores, or a
     *         score is NaN
     */
    public static double thresholdForAlertRate(double[] scores, double maxAlertRate) {
        if (!(maxAlertRate > 0 && maxAlertRate <= 1)) {
            throw new IllegalArgumentException("alert rate must be in (0, 1], got " + maxAlertRate);
        }
        if (scores.length == 0) throw new IllegalArgumentException("no scores to choose a threshold from");
        for (int i = 0; i < scores.length; i++) {
            if (Double.isNaN(scores[i])) throw new IllegalArgumentException("score " + i + " is NaN");
        }
        double[] sorted = scores.clone();
        Arrays.sort(sorted);
        int n = sorted.length;
        // Compare counts rather than ratios so 1% of 300 allows exactly 3 alerts.
        int allowed = (int) Math.floor(maxAlertRate * n + 1e-9);

        int flagged = 0;
        int i = n - 1;
        double lastFlagged = Double.NaN;
        while (i >= 0) {
            double s = sorted[i];
            int j = i;
            while (j >= 0 && sorted[j] == s) j--;
            if (flagged + (i - j) > allowed) break;
            flagged += i - j;
            lastFlagged = s;
            i = j;
        }
        double top = sorted[n - 1];
        if (flagged == 0) return top < 1 ? (top + 1) / 2 : Math.nextUp(top);
        return i >= 0 ? (lastFlagged + sorted[i]) / 2 : lastFlagged;
    }

    /**
     * Total cost of flagging every example scoring at least {@code threshold}: each fraud that
     * slips through costs its {@code missCost}, and each genuine purchase flagged costs
     * {@code falseAlarmCost}.
     *
     * @throws IllegalArgumentException under the same conditions as {@link #minCostThreshold}
     */
    public static double cost(double[] scores, boolean[] y, double[] missCost, double falseAlarmCost, double threshold) {
        checkCostInputs(scores, y, missCost, falseAlarmCost);
        double total = 0;
        for (int i = 0; i < y.length; i++) {
            boolean flagged = scores[i] >= threshold;
            if (y[i] && !flagged) total += missCost[i];
            else if (!y[i] && flagged) total += falseAlarmCost;
        }
        return total;
    }

    /**
     * The threshold that minimises {@link #cost}: what missed fraud loses plus what false alarms
     * cost to deal with.
     *
     * <p>F1 and recall targets count every fraud the same, but missing a £900 fraud is far worse
     * than missing a £9 one. Passing each transaction's amount as its miss cost lets the threshold
     * drop for the expensive cases' sake while a review cost keeps false alarms in check. Every
     * distinct score is tried in one pass from highest to lowest, as is flagging nothing at all;
     * on a tie the higher threshold wins, since it raises fewer alerts. Like
     * {@link #bestF1Threshold}, the threshold returned sits halfway between the chosen score and
     * the next lower one.
     *
     * @throws IllegalArgumentException if the arrays differ in length, a score is NaN, a miss cost
     *         is negative or not finite, or the false alarm cost isn't a positive finite number
     */
    public static double minCostThreshold(double[] scores, boolean[] y, double[] missCost, double falseAlarmCost) {
        checkCostInputs(scores, y, missCost, falseAlarmCost);
        if (scores.length == 0) throw new IllegalArgumentException("no scores to choose a threshold from");

        Integer[] order = new Integer[scores.length];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble((Integer i) -> scores[i]).reversed());

        // Start with nothing flagged: every fraud is missed. That's only reachable if a threshold
        // above the top score still fits in 0-1.
        double cost = 0;
        for (int i = 0; i < y.length; i++) if (y[i]) cost += missCost[i];
        double top = scores[order[0]];
        double best = Double.NaN, bestCost = Double.POSITIVE_INFINITY;
        if (top < 1) {
            best = (top + 1) / 2;
            bestCost = cost;
        }

        int i = 0;
        while (i < order.length) {
            double s = scores[order[i]];
            while (i < order.length && scores[order[i]] == s) {
                int k = order[i];
                cost += y[k] ? -missCost[k] : falseAlarmCost;
                i++;
            }
            if (cost < bestCost - 1e-9) {
                bestCost = cost;
                best = i < order.length ? (s + scores[order[i]]) / 2 : s;
            }
        }
        return best;
    }

    /** One group of a reliability table: examples whose predicted risk fell in [lower, upper). */
    public record CalibrationBin(double lower, double upper, int count, double meanPredicted, double observedRate) {
        /** How far the predicted risk is from what actually happened in this group. */
        public double gap() { return Math.abs(meanPredicted - observedRate); }
    }

    /**
     * Groups examples into {@code bins} equal-width ranges of predicted probability and, for each
     * range, compares the average prediction with the share that really were positive. When the
     * probabilities are honest the two match: of the purchases scored around 20%, about one in
     * five is fraud. Empty ranges are left out. A score of exactly 1 goes in the top range.
     *
     * @throws IllegalArgumentException if there are no bins or no scores, the inputs differ in
     *         length, or a score is NaN or outside 0-1
     */
    public static java.util.List<CalibrationBin> calibrationTable(double[] scores, boolean[] y, int bins) {
        if (bins < 1) throw new IllegalArgumentException("need at least one bin, got " + bins);
        if (scores.length != y.length) throw new IllegalArgumentException("scores and labels differ in length");
        if (scores.length == 0) throw new IllegalArgumentException("no scores to evaluate");
        int[] count = new int[bins];
        int[] positives = new int[bins];
        double[] sum = new double[bins];
        for (int i = 0; i < scores.length; i++) {
            double s = scores[i];
            if (!(s >= 0 && s <= 1)) throw new IllegalArgumentException("score " + i + " is not a probability: " + s);
            int b = Math.min((int) (s * bins), bins - 1);
            count[b]++;
            sum[b] += s;
            if (y[i]) positives[b]++;
        }
        java.util.List<CalibrationBin> out = new java.util.ArrayList<>();
        for (int b = 0; b < bins; b++) {
            if (count[b] == 0) continue;
            out.add(new CalibrationBin((double) b / bins, (double) (b + 1) / bins, count[b],
                    sum[b] / count[b], (double) positives[b] / count[b]));
        }
        return out;
    }

    /**
     * Expected calibration error: the gap between predicted and observed fraud rate in each range
     * of {@link #calibrationTable}, averaged with each range weighted by how many examples it holds.
     * 0 means the risk percentages can be taken at face value. Unlike log loss it ignores ranking,
     * so a model can rank well and still score badly here if its percentages are off.
     *
     * @throws IllegalArgumentException under the same conditions as {@link #calibrationTable}
     */
    public static double expectedCalibrationError(double[] scores, boolean[] y, int bins) {
        double total = 0;
        for (CalibrationBin b : calibrationTable(scores, y, bins)) total += b.count() * b.gap();
        return total / scores.length;
    }

    private static void checkCostInputs(double[] scores, boolean[] y, double[] missCost, double falseAlarmCost) {
        if (scores.length != y.length || missCost.length != y.length) {
            throw new IllegalArgumentException("scores, labels and miss costs differ in length");
        }
        if (!(falseAlarmCost > 0 && Double.isFinite(falseAlarmCost))) {
            throw new IllegalArgumentException("false alarm cost must be a positive number, got " + falseAlarmCost);
        }
        for (int i = 0; i < y.length; i++) {
            if (Double.isNaN(scores[i])) throw new IllegalArgumentException("score " + i + " is NaN");
            if (!(missCost[i] >= 0 && Double.isFinite(missCost[i]))) {
                throw new IllegalArgumentException("miss cost " + i + " must be zero or more, got " + missCost[i]);
            }
        }
    }
}
