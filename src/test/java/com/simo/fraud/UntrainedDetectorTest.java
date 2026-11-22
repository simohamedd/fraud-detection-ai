package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.function.Executable;
import org.junit.jupiter.api.io.TempDir;

class UntrainedDetectorTest {

    private static final Transaction PURCHASE = new Transaction(25, 12, 2, 0.1, false, 0, false);

    @Test
    void everyUseOfAnUntrainedDetectorExplainsWhatToDo(@TempDir Path dir) {
        FraudDetector d = new FraudDetector();
        List<Transaction> validation = SyntheticData.generate(500, 3);
        assertFalse(d.isTrained());
        assertUntrained(() -> d.score(PURCHASE), "score with");
        assertUntrained(() -> d.isFraud(PURCHASE), "score with");
        assertUntrained(() -> d.explain(PURCHASE), "explain");
        assertUntrained(d::weights, "weights");
        assertUntrained(() -> d.calibrate(validation), "calibrate");
        assertUntrained(() -> d.calibrateForRecall(validation, 0.9), "calibrate");
        Path file = dir.resolve("model.properties");
        assertUntrained(() -> d.save(file), "save");
        assertFalse(Files.exists(file), "no half-written model file should be left behind");
    }

    @Test
    void trainedAndLoadedDetectorsAreReady(@TempDir Path dir) throws Exception {
        FraudDetector trained = FraudDetector.train(SyntheticData.generate(2_000, 4), 4);
        assertTrue(trained.isTrained());
        Path file = dir.resolve("model.properties");
        trained.save(file);
        FraudDetector loaded = FraudDetector.load(file);
        assertTrue(loaded.isTrained());
        assertTrue(loaded.score(PURCHASE) >= 0 && loaded.score(PURCHASE) <= 1);
    }

    private static void assertUntrained(Executable call, String expected) {
        IllegalStateException e = assertThrows(IllegalStateException.class, call);
        assertTrue(e.getMessage().contains(expected), "expected '" + expected + "' in: " + e.getMessage());
        assertTrue(e.getMessage().contains("train() or load()"), e.getMessage());
    }
}
