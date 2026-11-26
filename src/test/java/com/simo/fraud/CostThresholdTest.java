package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class CostThresholdTest {

    // Scores high to low: a small fraud, a genuine purchase, then a big fraud below it.
    private static final double[] SCORES = {0.9, 0.7, 0.6, 0.3};
    private static final boolean[] Y = {true, false, true, false};
    private static final double[] AMOUNTS = {10, 40, 800, 25};

    @Test
    void costCountsMissedFraudAndFalseAlarms() {
        assertEquals(800, Metrics.cost(SCORES, Y, AMOUNTS, 5, 0.8), 1e-12);
        assertEquals(5, Metrics.cost(SCORES, Y, AMOUNTS, 5, 0.5), 1e-12);
        assertEquals(10, Metrics.cost(SCORES, Y, AMOUNTS, 5, 0.1), 1e-12);
        assertEquals(810, Metrics.cost(SCORES, Y, AMOUNTS, 5, 0.95), 1e-12);
    }

    @Test
    void acceptsAFalseAlarmToCatchAnExpensiveFraud() {
        double t = Metrics.minCostThreshold(SCORES, Y, AMOUNTS, 5);
        assertEquals(0.45, t, 1e-12);
        assertEquals(5, Metrics.cost(SCORES, Y, AMOUNTS, 5, t), 1e-12);
    }

    @Test
    void letsSmallFraudGoWhenReviewsCostMore() {
        double[] scores = {0.9, 0.5, 0.5, 0.5};
        boolean[] y = {true, true, false, false};
        double[] amounts = {500, 3, 20, 20};
        // Flagging the 0.5 group catches £3 of fraud for two £5 reviews, so stop above it.
        double t = Metrics.minCostThreshold(scores, y, amounts, 5);
        assertEquals(0.7, t, 1e-12);
        assertEquals(3, Metrics.cost(scores, y, amounts, 5, t), 1e-12);
    }

    @Test
    void flagsNothingWhenEveryAlertCostsMoreThanItSaves() {
        double[] scores = {0.8, 0.4};
        boolean[] y = {false, true};
        double t = Metrics.minCostThreshold(scores, y, new double[] {0, 2}, 5);
        assertEquals(0.9, t, 1e-12);
        assertEquals(0, Metrics.confusion(scores, y, t).truePositives() + Metrics.confusion(scores, y, t).falsePositives());
    }

    @Test
    void prefersFewerAlertsOnATie() {
        // Flagging the genuine 0.6 costs £5 and catching the £5 fraud below it saves £5.
        double[] scores = {0.8, 0.6, 0.4};
        boolean[] y = {true, false, true};
        double t = Metrics.minCostThreshold(scores, y, new double[] {100, 0, 5}, 5);
        assertEquals(0.7, t, 1e-12);
    }

    @Test
    void neverCostsMoreThanAnyOtherThreshold() {
        Random r = new Random(3);
        for (int trial = 0; trial < 200; trial++) {
            int n = 1 + r.nextInt(30);
            double[] s = new double[n], amounts = new double[n];
            boolean[] y = new boolean[n];
            for (int i = 0; i < n; i++) {
                s[i] = r.nextInt(10) / 10.0;
                y[i] = r.nextDouble() < 0.4;
                amounts[i] = r.nextInt(50);
            }
            double best = Metrics.cost(s, y, amounts, 4, Metrics.minCostThreshold(s, y, amounts, 4));
            for (double t = 0; t <= 1.0001; t += 0.05) {
                assertTrue(best <= Metrics.cost(s, y, amounts, 4, t) + 1e-9, "trial " + trial + " threshold " + t);
            }
        }
    }

    @Test
    void detectorCalibratesToTheCheapestThreshold() {
        List<Transaction> all = SyntheticData.generate(20_000, 11);
        FraudDetector d = FraudDetector.train(all.subList(0, 12_000), 11);
        List<Transaction> validation = all.subList(12_000, 20_000);
        double[] s = d.scores(validation);
        boolean[] y = FraudDetector.labels(validation);
        double[] amounts = FraudDetector.amounts(validation);

        d.calibrate(validation);
        double f1Cost = Metrics.cost(s, y, amounts, 5, d.threshold());
        d.calibrateForCost(validation, 5);
        double costCost = Metrics.cost(s, y, amounts, 5, d.threshold());
        assertTrue(costCost <= f1Cost, "cost threshold £" + costCost + " vs F1 threshold £" + f1Cost);
    }

    @Test
    void rejectsBadInput() {
        double[] s = {0.5};
        boolean[] y = {true};
        assertThrows(IllegalArgumentException.class, () -> Metrics.minCostThreshold(s, y, new double[] {1}, 0));
        assertThrows(IllegalArgumentException.class, () -> Metrics.minCostThreshold(s, y, new double[] {1}, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Metrics.minCostThreshold(s, y, new double[] {-1}, 5));
        assertThrows(IllegalArgumentException.class, () -> Metrics.minCostThreshold(s, y, new double[2], 5));
        assertThrows(IllegalArgumentException.class, () -> Metrics.minCostThreshold(new double[] {Double.NaN}, y, new double[1], 5));
        assertThrows(IllegalArgumentException.class, () -> Metrics.minCostThreshold(new double[0], new boolean[0], new double[0], 5));

        FraudDetector d = FraudDetector.train(SyntheticData.generate(5_000, 2), 2);
        double before = d.threshold();
        assertThrows(IllegalArgumentException.class, () -> d.calibrateForCost(SyntheticData.generate(2_000, 4), -5));
        assertEquals(before, d.threshold(), 0);
    }
}
