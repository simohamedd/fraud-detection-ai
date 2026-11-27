package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class AlertRateTest {

    private static int flagged(double[] scores, double threshold) {
        int n = 0;
        for (double s : scores) if (s >= threshold) n++;
        return n;
    }

    @Test
    void flagsExactlyTheBudgetWhenThereAreNoTies() {
        double[] s = {0.1, 0.9, 0.4, 0.7, 0.2, 0.8, 0.3, 0.6, 0.5, 0.05};
        double t = Metrics.thresholdForAlertRate(s, 0.3);
        assertEquals(0.65, t, 1e-12);
        assertEquals(3, flagged(s, t));
    }

    @Test
    void leavesOutATieThatWouldGoOverBudget() {
        double[] s = {0.9, 0.6, 0.6, 0.6, 0.2};
        // Budget of 2: the top score fits, but the three tied at 0.6 would make 4.
        double t = Metrics.thresholdForAlertRate(s, 0.4);
        assertEquals(0.75, t, 1e-12);
        assertEquals(1, flagged(s, t));
    }

    @Test
    void countsRatherThanRatiosSoRoundingDoesNotLoseAnAlert() {
        double[] s = new double[300];
        for (int i = 0; i < s.length; i++) s[i] = i / 300.0;
        assertEquals(3, flagged(s, Metrics.thresholdForAlertRate(s, 0.01)));
    }

    @Test
    void flagsNothingWhenEvenTheTopGroupIsOverBudget() {
        double[] s = {0.8, 0.8, 0.3, 0.1};
        double t = Metrics.thresholdForAlertRate(s, 0.25);
        assertEquals(0.9, t, 1e-12);
        assertEquals(0, flagged(s, t));
        assertEquals(0, flagged(new double[] {1.0, 1.0}, Metrics.thresholdForAlertRate(new double[] {1.0, 1.0}, 0.5)));
    }

    @Test
    void fullBudgetFlagsEverything() {
        double[] s = {0.4, 0.2, 0.7};
        double t = Metrics.thresholdForAlertRate(s, 1);
        assertEquals(0.2, t, 1e-12);
        assertEquals(3, flagged(s, t));
    }

    @Test
    void detectorStaysWithinBudgetOnHeldOutData() {
        List<Transaction> all = SyntheticData.generate(12_000, 11);
        FraudDetector d = FraudDetector.train(all.subList(0, 6_000), 11);
        List<Transaction> recent = all.subList(6_000, 12_000);
        d.calibrateForAlertRate(recent, 0.02);
        long alerts = recent.stream().filter(d::isFraud).count();
        assertTrue(alerts <= 120, "raised " + alerts + " alerts");
        assertTrue(alerts >= 100, "budget barely used: " + alerts + " alerts");
    }

    @Test
    void rejectsBadInputAndKeepsTheThreshold() {
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForAlertRate(new double[] {0.5}, 0));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForAlertRate(new double[] {0.5}, 1.5));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForAlertRate(new double[] {0.5}, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForAlertRate(new double[0], 0.1));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForAlertRate(new double[] {Double.NaN}, 0.1));

        List<Transaction> data = SyntheticData.generate(3_000, 2);
        FraudDetector d = FraudDetector.train(data, 2);
        double before = d.threshold();
        assertThrows(IllegalArgumentException.class, () -> d.calibrateForAlertRate(List.of(), 0.1));
        assertThrows(IllegalArgumentException.class, () -> d.calibrateForAlertRate(data, 2));
        assertEquals(before, d.threshold());
        assertThrows(IllegalStateException.class, () -> new FraudDetector().calibrateForAlertRate(data, 0.1));
    }
}
