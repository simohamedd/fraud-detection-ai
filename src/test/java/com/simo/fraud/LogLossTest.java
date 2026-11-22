package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class LogLossTest {

    @Test
    void averagesMinusLogOfTheProbabilityGivenToTheTruth() {
        double[] s = {0.9, 0.2, 0.6};
        boolean[] y = {true, false, false};
        double expected = -(Math.log(0.9) + Math.log(0.8) + Math.log(0.4)) / 3;
        assertEquals(expected, Metrics.logLoss(s, y), 1e-12);
    }

    @Test
    void perfectPredictionsCostNothingAndCoinFlipsCostLogTwo() {
        assertEquals(0, Metrics.logLoss(new double[] {1, 0}, new boolean[] {true, false}), 1e-12);
        assertEquals(Math.log(2), Metrics.logLoss(new double[] {0.5, 0.5}, new boolean[] {true, false}), 1e-12);
    }

    @Test
    void certainMistakeIsLargeButFinite() {
        double loss = Metrics.logLoss(new double[] {0.0}, new boolean[] {true});
        assertTrue(Double.isFinite(loss) && loss > 30, "loss was " + loss);
    }

    @Test
    void punishesConfidenceThatRankingMetricsIgnore() {
        boolean[] y = {true, false};
        double[] timid = {0.51, 0.49};
        double[] sure = {0.99, 0.01};
        assertEquals(Metrics.rocAuc(timid, y), Metrics.rocAuc(sure, y));
        assertTrue(Metrics.logLoss(sure, y) < Metrics.logLoss(timid, y));
    }

    @Test
    void trainedModelBeatsGuessingTheFraudRate() {
        List<Transaction> all = SyntheticData.generate(12_000, 5);
        FraudDetector d = FraudDetector.train(all.subList(0, 8_000), 5);
        List<Transaction> test = all.subList(8_000, 12_000);
        boolean[] y = FraudDetector.labels(test);
        int fraud = 0;
        for (boolean v : y) if (v) fraud++;
        double rate = (double) fraud / y.length;
        double[] constant = new double[y.length];
        java.util.Arrays.fill(constant, rate);
        assertTrue(Metrics.logLoss(d.scores(test), y) < Metrics.logLoss(constant, y));
    }

    @Test
    void rejectsBadInput() {
        assertThrows(IllegalArgumentException.class, () -> Metrics.logLoss(new double[0], new boolean[0]));
        assertThrows(IllegalArgumentException.class, () -> Metrics.logLoss(new double[] {0.5}, new boolean[2]));
        assertThrows(IllegalArgumentException.class, () -> Metrics.logLoss(new double[] {Double.NaN}, new boolean[1]));
        assertThrows(IllegalArgumentException.class, () -> Metrics.logLoss(new double[] {1.2}, new boolean[1]));
        assertThrows(IllegalArgumentException.class, () -> Metrics.logLoss(new double[] {-0.1}, new boolean[1]));
    }
}
