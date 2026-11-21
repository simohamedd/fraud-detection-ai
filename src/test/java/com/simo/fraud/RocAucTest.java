package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class RocAucTest {

    @Test
    void perfectAndReversedRankings() {
        boolean[] y = {false, true, false, true};
        assertEquals(1.0, Metrics.rocAuc(new double[] {0.1, 0.9, 0.2, 0.8}, y), 1e-12);
        assertEquals(0.0, Metrics.rocAuc(new double[] {0.9, 0.1, 0.8, 0.2}, y), 1e-12);
    }

    @Test
    void countsPairsAndHalvesTies() {
        // Pairs (fraud, genuine): (0.7, 0.4) win, (0.7, 0.8) lose, (0.4, 0.4) tie, (0.4, 0.8) lose.
        double[] scores = {0.7, 0.4, 0.4, 0.8};
        boolean[] y = {true, true, false, false};
        assertEquals(1.5 / 4, Metrics.rocAuc(scores, y), 1e-12);
    }

    @Test
    void allScoresTiedIsChance() {
        assertEquals(0.5, Metrics.rocAuc(new double[] {0.3, 0.3, 0.3}, new boolean[] {true, false, false}), 1e-12);
    }

    @Test
    void missingClassGivesNaN() {
        assertTrue(Double.isNaN(Metrics.rocAuc(new double[] {0.2, 0.8}, new boolean[] {false, false})));
        assertTrue(Double.isNaN(Metrics.rocAuc(new double[] {0.2, 0.8}, new boolean[] {true, true})));
        assertTrue(Double.isNaN(Metrics.rocAuc(new double[0], new boolean[0])));
    }

    @Test
    void nanScoreIsRejectedRatherThanRankedFirst() {
        // Sorting puts NaN above every real score, which used to make this look like a perfect model.
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.rocAuc(new double[] {Double.NaN, 0.9, 0.1}, new boolean[] {true, false, false}));
    }

    @Test
    void mismatchedLengthsAreRejected() {
        // A longer label array used to be silently cut short; a shorter one crashed with an index error.
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.rocAuc(new double[] {0.2, 0.8}, new boolean[] {false, true, true}));
        assertThrows(IllegalArgumentException.class,
                () -> Metrics.rocAuc(new double[] {0.2, 0.8, 0.5}, new boolean[] {false, true}));
    }
}
