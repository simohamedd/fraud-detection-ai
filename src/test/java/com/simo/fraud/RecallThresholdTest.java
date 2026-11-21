package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class RecallThresholdTest {

    @Test
    void stopsAsSoonAsTheTargetIsMet() {
        // Four fraud cases; catching three (75%) only needs the top four scores.
        double[] scores = {0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3};
        boolean[] y = {true, true, false, true, false, true, false};
        double t = Metrics.thresholdForRecall(scores, y, 0.75);
        assertEquals(0.65, t, 1e-12);
        Metrics.Confusion c = Metrics.confusion(scores, y, t);
        assertEquals(3, c.truePositives());
        assertEquals(1, c.falsePositives());
    }

    @Test
    void fullRecallReachesTheLowestFraud() {
        double[] scores = {0.95, 0.9, 0.8, 0.7, 0.6, 0.4, 0.3};
        boolean[] y = {true, true, false, true, false, true, false};
        double t = Metrics.thresholdForRecall(scores, y, 1.0);
        assertEquals(0.35, t, 1e-12);
        assertEquals(1.0, Metrics.confusion(scores, y, t).recall(), 0);
    }

    @Test
    void tiedScoresAreFlaggedTogether() {
        double[] scores = {0.9, 0.5, 0.5, 0.5, 0.1};
        boolean[] y = {true, false, true, false, false};
        double t = Metrics.thresholdForRecall(scores, y, 1.0);
        assertEquals(0.3, t, 1e-12);
        assertEquals(2, Metrics.confusion(scores, y, t).falsePositives());
    }

    @Test
    void lowestScoreIsReturnedWhenEverythingMustBeFlagged() {
        assertEquals(0.2, Metrics.thresholdForRecall(new double[] {0.8, 0.2}, new boolean[] {false, true}, 1.0), 0);
    }

    @Test
    void targetIsCountedExactlyDespiteRounding() {
        // 0.14 * 50 is 7.000000000000001 in floating point; seven catches must be enough.
        int n = 50;
        double[] scores = new double[n];
        boolean[] y = new boolean[n];
        for (int i = 0; i < n; i++) { scores[i] = 1 - i / 100.0; y[i] = true; }
        double t = Metrics.thresholdForRecall(scores, y, 0.14);
        assertEquals(7, Metrics.confusion(scores, y, t).truePositives());
    }

    @Test
    void alwaysMeetsTheTargetWithTheHighestPossibleThreshold() {
        Random r = new Random(17);
        for (int run = 0; run < 50; run++) {
            int n = 300;
            double[] scores = new double[n];
            boolean[] y = new boolean[n];
            y[0] = true;
            for (int i = 0; i < n; i++) {
                if (i > 0) y[i] = r.nextDouble() < 0.1;
                scores[i] = Math.round(((y[i] ? 0.6 : 0.4) + r.nextGaussian() * 0.2) * 100) / 100.0;
            }
            double target = 0.5 + 0.5 * r.nextDouble();
            double t = Metrics.thresholdForRecall(scores, y, target);
            assertTrue(Metrics.confusion(scores, y, t).recall() >= target - 1e-12);
            // Flagging one fewer distinct score would miss the target.
            double higher = Double.POSITIVE_INFINITY;
            for (double s : scores) if (s >= t) higher = Math.min(higher, s);
            assertTrue(Metrics.confusion(scores, y, Math.nextUp(higher)).recall() < target,
                    "a higher threshold would also have met " + target);
        }
    }

    @Test
    void detectorCalibratesToARecallTarget() {
        List<Transaction> all = SyntheticData.generate(12_000, 9);
        FraudDetector d = FraudDetector.train(all.subList(0, 8_000), 9);
        List<Transaction> validation = all.subList(8_000, 12_000);
        d.calibrateForRecall(validation, 0.9);
        Metrics.Confusion c = Metrics.confusion(d.scores(validation), FraudDetector.labels(validation), d.threshold());
        assertTrue(c.recall() >= 0.9, "recall " + c.recall());
    }

    @Test
    void rejectsBadInputAndKeepsTheThreshold() {
        double[] s = {0.2, 0.8};
        boolean[] y = {false, true};
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForRecall(s, y, 0));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForRecall(s, y, 1.1));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForRecall(s, y, Double.NaN));
        assertThrows(IllegalArgumentException.class, () -> Metrics.thresholdForRecall(s, new boolean[] {true}, 0.5));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.thresholdForRecall(new double[] {Double.NaN, 0.3}, y, 0.5));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.thresholdForRecall(s, new boolean[] {false, false}, 0.5));

        List<Transaction> all = SyntheticData.generate(6_000, 2);
        FraudDetector d = FraudDetector.train(all.subList(0, 4_000), 2);
        double before = d.threshold();
        List<Transaction> genuineOnly = all.subList(4_000, 6_000).stream().filter(t -> !t.fraud()).toList();
        assertThrows(IllegalArgumentException.class, () -> d.calibrateForRecall(genuineOnly, 0.9));
        assertThrows(IllegalArgumentException.class, () -> d.calibrateForRecall(all.subList(4_000, 6_000), 2));
        assertEquals(before, d.threshold(), 0);
    }
}
