package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

class AveragePrecisionTest {

    @Test
    void perfectRankingScoresOne() {
        double[] scores = {0.1, 0.9, 0.2, 0.8};
        boolean[] y = {false, true, false, true};
        assertEquals(1.0, Metrics.averagePrecision(scores, y), 1e-12);
    }

    @Test
    void weightsEachCatchByPrecisionAtThatPoint() {
        // Ranked: fraud, genuine, fraud, genuine -> 1/2 * 1 + 1/2 * 2/3.
        double[] scores = {0.9, 0.8, 0.7, 0.1};
        boolean[] y = {true, false, true, false};
        assertEquals(0.5 + 1.0 / 3, Metrics.averagePrecision(scores, y), 1e-12);
    }

    @Test
    void tiedScoresCountTogether() {
        // A tie can't be split by any threshold, so the fraud only comes with its genuine twin.
        assertEquals(0.5, Metrics.averagePrecision(new double[] {0.5, 0.5}, new boolean[] {true, false}), 1e-12);
        assertEquals(0.5, Metrics.averagePrecision(new double[] {0.5, 0.5}, new boolean[] {false, true}), 1e-12);
    }

    @Test
    void randomScoresLandNearTheFraudRate() {
        Random r = new Random(3);
        int n = 20_000;
        double[] scores = new double[n];
        boolean[] y = new boolean[n];
        int pos = 0;
        for (int i = 0; i < n; i++) {
            y[i] = r.nextDouble() < 0.03;
            if (y[i]) pos++;
            scores[i] = r.nextDouble();
        }
        assertEquals((double) pos / n, Metrics.averagePrecision(scores, y), 0.01);
    }

    @Test
    void trainedModelBeatsChanceByAWideMargin() {
        List<Transaction> all = SyntheticData.generate(12_000, 5);
        FraudDetector d = FraudDetector.train(all.subList(0, 8_000), 5);
        List<Transaction> test = all.subList(8_000, 12_000);
        double ap = Metrics.averagePrecision(d.scores(test), FraudDetector.labels(test));
        assertTrue(ap > 0.3, "average precision " + ap);
    }

    @Test
    void edgeCases() {
        assertTrue(Double.isNaN(Metrics.averagePrecision(new double[] {0.2, 0.8}, new boolean[] {false, false})));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.averagePrecision(new double[] {0.2}, new boolean[] {true, false}));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.averagePrecision(new double[] {Double.NaN, 0.4}, new boolean[] {true, false}));
    }
}
