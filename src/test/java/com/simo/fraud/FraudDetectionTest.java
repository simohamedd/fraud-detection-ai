package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class FraudDetectionTest {

    @Test
    void syntheticDataIsImbalancedAndReproducible() {
        List<Transaction> a = SyntheticData.generate(20_000, 1);
        long fraud = a.stream().filter(Transaction::fraud).count();
        assertTrue(fraud > 400 && fraud < 800, "about 3% fraud, got " + fraud);
        assertEquals(a, SyntheticData.generate(20_000, 1));
    }

    @Test
    void scalerGivesZeroMeanUnitVariance() {
        double[][] x = {{1, 10}, {2, 20}, {3, 30}};
        double[][] s = new StandardScaler().fit(x).transform(x);
        assertEquals(0, s[0][0] + s[1][0] + s[2][0], 1e-9);
        assertEquals(-1.2247, s[0][1], 1e-3);
    }

    @Test
    void metricsOnKnownExample() {
        double[] scores = {0.9, 0.8, 0.3, 0.2, 0.7};
        boolean[] y = {true, false, true, false, true};
        Metrics.Confusion c = Metrics.confusion(scores, y, 0.5);
        assertEquals(2, c.truePositives());
        assertEquals(1, c.falsePositives());
        assertEquals(1, c.falseNegatives());
        assertEquals(2.0 / 3, c.precision(), 1e-9);
        assertEquals(2.0 / 3, c.recall(), 1e-9);
        assertEquals(4.0 / 6, Metrics.rocAuc(scores, y), 1e-9);
        assertEquals(1.0, Metrics.rocAuc(new double[] {0.1, 0.9}, new boolean[] {false, true}), 1e-9);
    }

    @Test
    void modelCatchesFraudOnUnseenData() {
        List<Transaction> all = SyntheticData.generate(20_000, 3);
        FraudDetector d = FraudDetector.train(all.subList(0, 12_000), 3);
        d.calibrate(all.subList(12_000, 16_000));
        List<Transaction> test = all.subList(16_000, 20_000);
        double[] scores = d.scores(test);
        boolean[] y = FraudDetector.labels(test);

        assertTrue(Metrics.rocAuc(scores, y) > 0.95, "AUC should be high");
        Metrics.Confusion c = Metrics.confusion(scores, y, d.threshold());
        assertTrue(c.recall() > 0.6, "recall " + c.recall());
        assertTrue(c.precision() > 0.5, "precision " + c.precision());
    }

    @Test
    void obviousCasesAreScoredSensibly() {
        FraudDetector d = FraudDetector.train(SyntheticData.generate(15_000, 5), 5);
        assertFalse(d.isFraud(new Transaction(12.5, 12, 2, 0.1, false, 0, false)));
        assertTrue(d.isFraud(new Transaction(1500, 3, 900, 0.9, true, 5, false)));
    }

    @Test
    void explanationNamesTheSignalsBehindAFlag() {
        FraudDetector d = FraudDetector.train(SyntheticData.generate(15_000, 8), 8);
        var reasons = d.explain(new Transaction(1500, 3, 900, 0.9, true, 5, false));
        assertEquals(Transaction.FEATURES.length, reasons.size());
        assertTrue(reasons.get(0).logOdds() > 0, "top reason should raise the risk");
        assertTrue(Math.abs(reasons.get(0).logOdds()) >= Math.abs(reasons.get(5).logOdds()));
    }
}
