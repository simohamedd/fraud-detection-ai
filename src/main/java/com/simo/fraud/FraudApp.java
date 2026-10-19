package com.simo.fraud;

import java.util.List;

/** Trains the fraud model on synthetic data and reports how well it does. Run with: mvn -q compile exec:java */
public class FraudApp {

    public static void main(String[] args) {
        List<Transaction> all = SyntheticData.generate(30_000, 7);
        List<Transaction> train = all.subList(0, 18_000);
        List<Transaction> validation = all.subList(18_000, 24_000);
        List<Transaction> test = all.subList(24_000, 30_000);

        FraudDetector detector = FraudDetector.train(train, 7);
        detector.calibrate(validation);

        double[] scores = detector.scores(test);
        boolean[] y = FraudDetector.labels(test);
        Metrics.Confusion c = Metrics.confusion(scores, y, detector.threshold());

        System.out.printf("Trained on %,d transactions, tested on %,d unseen ones%n", train.size(), test.size());
        System.out.printf("ROC AUC     %.3f%n", Metrics.rocAuc(scores, y));
        System.out.printf("Threshold   %.2f%n", detector.threshold());
        System.out.printf("Precision   %.1f%%  (of transactions flagged, how many were fraud)%n", 100 * c.precision());
        System.out.printf("Recall      %.1f%%  (of all fraud, how much was caught)%n", 100 * c.recall());
        System.out.printf("Confusion   caught %d, missed %d, false alarms %d, correctly passed %d%n",
                c.truePositives(), c.falseNegatives(), c.falsePositives(), c.trueNegatives());

        System.out.println("\nWhat the model learned (weights on scaled features, higher = more suspicious):");
        double[] w = detector.weights();
        for (int j = 0; j < w.length; j++) System.out.printf("  %-15s %+.2f%n", Transaction.FEATURES[j], w[j]);

        System.out.println("\nScoring two new purchases:");
        show(detector, new Transaction(18.40, 13, 3, 0.15, false, 0, false), "£18.40 lunch near home");
        show(detector, new Transaction(940.00, 3, 820, 0.8, true, 4, false), "£940 online at 3am, 820 km away");
    }

    private static void show(FraudDetector d, Transaction t, String label) {
        System.out.printf("  %-34s risk %5.1f%%  -> %s%n", label, 100 * d.score(t), d.isFraud(t) ? "BLOCK and alert" : "approve");
        d.explain(t).stream().limit(3).forEach(c ->
                System.out.printf("      %s %s the risk (%+.2f)%n", c.feature(), c.logOdds() >= 0 ? "raised" : "lowered", c.logOdds()));
    }
}
