package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

class ThresholdTest {

    @Test
    void findsCutOffBetweenCloseScores() {
        // The only perfect split lies between 0.503 and 0.507, which a 0.01 grid can't reach.
        double[] scores = {0.2, 0.501, 0.503, 0.507, 0.509, 0.9};
        boolean[] y = {false, false, false, true, true, true};
        double t = Metrics.bestF1Threshold(scores, y);
        assertEquals(0.505, t, 1e-12);
        assertEquals(1.0, Metrics.confusion(scores, y, t).f1(), 1e-12);
    }

    @Test
    void tiedScoresAreFlaggedTogether() {
        double[] scores = {0.3, 0.6, 0.6, 0.6, 0.8};
        boolean[] y = {false, true, false, true, true};
        double t = Metrics.bestF1Threshold(scores, y);
        // Taking all three 0.6s gives F1 6/7, better than 0.8 alone (F1 0.5).
        assertEquals(0.45, t, 1e-12);
        assertEquals(6.0 / 7, Metrics.confusion(scores, y, t).f1(), 1e-12);
    }

    @Test
    void neverWorseThanAnyGridThreshold() {
        Random r = new Random(11);
        for (int run = 0; run < 50; run++) {
            int n = 200;
            double[] scores = new double[n];
            boolean[] y = new boolean[n];
            for (int i = 0; i < n; i++) {
                y[i] = r.nextDouble() < 0.1;
                scores[i] = Math.min(1, Math.max(0, (y[i] ? 0.6 : 0.4) + r.nextGaussian() * 0.2));
            }
            double best = Metrics.confusion(scores, y, Metrics.bestF1Threshold(scores, y)).f1();
            for (int step = 1; step < 100; step++) {
                double f1 = Metrics.confusion(scores, y, step / 100.0).f1();
                assertTrue(best >= f1 - 1e-12, "grid threshold " + step / 100.0 + " beat the search");
            }
        }
    }

    @Test
    void edgeCases() {
        assertEquals(0.5, Metrics.bestF1Threshold(new double[] {0.1, 0.9}, new boolean[] {false, false}), 0);
        assertEquals(0.7, Metrics.bestF1Threshold(new double[] {0.7}, new boolean[] {true}), 0);
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.bestF1Threshold(new double[] {0.2, Double.NaN}, new boolean[] {true, false}));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.bestF1Threshold(new double[] {0.2}, new boolean[] {true, false}));
    }
}
