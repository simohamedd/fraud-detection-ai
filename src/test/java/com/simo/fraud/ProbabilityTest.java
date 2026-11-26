package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class ProbabilityTest {

    @Test
    void classWeightingDoesNotShiftTheProbabilities() {
        // No signal at all, 10% positives: the only honest answer is 10% for everyone. Without the
        // correction, weighting positives 9x would push every prediction to 50%.
        int n = 1_000;
        double[][] x = new double[n][1];
        boolean[] y = new boolean[n];
        for (int i = 0; i < n; i += 10) y[i] = true;
        double weight = LogisticRegression.balancedWeight(y);
        assertEquals(9.0, weight, 1e-12);

        LogisticRegression m = new LogisticRegression(0.1, 0, 300, weight).fit(x, y, 1);
        assertEquals(0.1, m.predictProbability(new double[] {0}), 0.01);
    }

    @Test
    void unweightedTrainingIsUnchanged() {
        int n = 1_000;
        double[][] x = new double[n][1];
        boolean[] y = new boolean[n];
        for (int i = 0; i < n; i += 4) y[i] = true;
        LogisticRegression m = new LogisticRegression(0.1, 0, 300, 1).fit(x, y, 1);
        assertEquals(0.25, m.predictProbability(new double[] {0}), 0.01);
    }

    @Test
    void riskPercentagesMatchHowOftenFraudHappens() {
        List<Transaction> all = SyntheticData.generate(30_000, 7);
        FraudDetector d = FraudDetector.train(all.subList(0, 18_000), 7);
        List<Transaction> test = all.subList(18_000, 30_000);
        double[] s = d.scores(test);

        double meanScore = 0;
        int fraud = 0, high = 0, highFraud = 0, low = 0, lowFraud = 0;
        for (int i = 0; i < s.length; i++) {
            boolean f = test.get(i).fraud();
            meanScore += s[i];
            if (f) fraud++;
            if (s[i] >= 0.5) { high++; if (f) highFraud++; }
            if (s[i] < 0.1) { low++; if (f) lowFraud++; }
        }
        meanScore /= s.length;
        double rate = (double) fraud / s.length;

        assertEquals(rate, meanScore, 0.25 * rate, "mean risk " + meanScore + " vs fraud rate " + rate);
        assertTrue((double) highFraud / high > 0.8, "scored 50%+ but only " + highFraud + " of " + high + " were fraud");
        assertTrue((double) lowFraud / low < 0.01, "scored under 10% but " + lowFraud + " of " + low + " were fraud");
    }

    @Test
    void rejectsANonPositiveWeight() {
        assertThrows(IllegalArgumentException.class, () -> new LogisticRegression(0.1, 0, 1, 0));
        assertThrows(IllegalArgumentException.class, () -> new LogisticRegression(0.1, 0, 1, -2));
        assertThrows(IllegalArgumentException.class, () -> new LogisticRegression(0.1, 0, 1, Double.NaN));
    }
}
