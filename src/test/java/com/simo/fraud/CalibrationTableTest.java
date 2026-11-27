package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class CalibrationTableTest {

    @Test
    void groupsScoresByRangeAndComparesWithWhatHappened() {
        double[] s = {0.05, 0.15, 0.12, 0.85, 0.95, 1.0};
        boolean[] y = {false, true, false, true, true, false};
        List<Metrics.CalibrationBin> bins = Metrics.calibrationTable(s, y, 2);
        assertEquals(2, bins.size());

        Metrics.CalibrationBin low = bins.get(0);
        assertEquals(0, low.lower(), 1e-12);
        assertEquals(0.5, low.upper(), 1e-12);
        assertEquals(3, low.count());
        assertEquals((0.05 + 0.15 + 0.12) / 3, low.meanPredicted(), 1e-12);
        assertEquals(1.0 / 3, low.observedRate(), 1e-12);

        // A score of exactly 1 belongs to the top range rather than one past it.
        Metrics.CalibrationBin high = bins.get(1);
        assertEquals(3, high.count());
        assertEquals((0.85 + 0.95 + 1.0) / 3, high.meanPredicted(), 1e-12);
        assertEquals(2.0 / 3, high.observedRate(), 1e-12);
    }

    @Test
    void leavesOutEmptyRanges() {
        List<Metrics.CalibrationBin> bins = Metrics.calibrationTable(new double[] {0.01, 0.92}, new boolean[] {false, true}, 10);
        assertEquals(2, bins.size());
        assertEquals(0.9, bins.get(1).lower(), 1e-12);
    }

    @Test
    void calibrationErrorWeightsEachRangeByItsSize() {
        double[] s = {0.05, 0.15, 0.12, 0.85, 0.95, 1.0};
        boolean[] y = {false, true, false, true, true, false};
        double low = Math.abs((0.05 + 0.15 + 0.12) / 3 - 1.0 / 3);
        double high = Math.abs((0.85 + 0.95 + 1.0) / 3 - 2.0 / 3);
        assertEquals((3 * low + 3 * high) / 6, Metrics.expectedCalibrationError(s, y, 2), 1e-12);
    }

    @Test
    void honestProbabilitiesHaveNoCalibrationError() {
        // Four at 25% with one fraud, four at 75% with three.
        double[] s = {0.25, 0.25, 0.25, 0.25, 0.75, 0.75, 0.75, 0.75};
        boolean[] y = {true, false, false, false, true, true, true, false};
        assertEquals(0, Metrics.expectedCalibrationError(s, y, 4), 1e-12);
    }

    @Test
    void trainedModelIsWellCalibratedButAnInflatedOneIsNot() {
        List<Transaction> all = SyntheticData.generate(16_000, 3);
        FraudDetector d = FraudDetector.train(all.subList(0, 10_000), 3);
        List<Transaction> test = all.subList(10_000, 16_000);
        double[] scores = d.scores(test);
        boolean[] y = FraudDetector.labels(test);
        double ece = Metrics.expectedCalibrationError(scores, y, 10);
        assertTrue(ece < 0.01, "calibration error " + ece);

        // Same ranking, but odds multiplied by 30 as the uncorrected class weighting would do.
        double[] inflated = new double[scores.length];
        for (int i = 0; i < scores.length; i++) inflated[i] = 30 * scores[i] / (1 + 29 * scores[i]);
        assertTrue(Metrics.expectedCalibrationError(inflated, y, 10) > 3 * ece);
    }

    @Test
    void rejectsBadInput() {
        assertThrows(IllegalArgumentException.class, () -> Metrics.calibrationTable(new double[] {0.5}, new boolean[1], 0));
        assertThrows(IllegalArgumentException.class, () -> Metrics.calibrationTable(new double[0], new boolean[0], 10));
        assertThrows(IllegalArgumentException.class, () -> Metrics.calibrationTable(new double[] {0.5}, new boolean[2], 10));
        assertThrows(IllegalArgumentException.class, () -> Metrics.calibrationTable(new double[] {Double.NaN}, new boolean[1], 10));
        assertThrows(IllegalArgumentException.class, () -> Metrics.calibrationTable(new double[] {1.5}, new boolean[1], 10));
        assertThrows(IllegalArgumentException.class, () -> Metrics.expectedCalibrationError(new double[] {-0.1}, new boolean[1], 10));
    }
}
