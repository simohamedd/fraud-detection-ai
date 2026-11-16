package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelFileTest {

    private static List<String> saved;

    @BeforeAll
    static void saveAModel(@TempDir Path dir) throws IOException {
        FraudDetector d = FraudDetector.train(SyntheticData.generate(4_000, 2), 2);
        Path file = dir.resolve("model.properties");
        d.save(file);
        saved = Files.readAllLines(file, StandardCharsets.UTF_8);
    }

    @Test
    void untouchedFileStillLoads(@TempDir Path dir) throws IOException {
        Path file = write(dir, saved);
        assertEquals(Transaction.FEATURES.length, FraudDetector.load(file).weights().length);
    }

    @Test
    void missingValueIsNamed(@TempDir Path dir) {
        assertRejected(dir, without("bias"), "missing 'bias'");
        assertRejected(dir, without("threshold"), "missing 'threshold'");
        assertRejected(dir, without("weights"), "missing 'weights'");
    }

    @Test
    void badNumberIsNamed(@TempDir Path dir) {
        assertRejected(dir, replace("threshold", "0.4x"), "invalid number for threshold");
        assertRejected(dir, replace("weights", "1,2,oops,4,5,6"), "invalid number for weights[2]");
    }

    @Test
    void wrongNumberOfValuesIsCaughtAtLoad(@TempDir Path dir) {
        assertRejected(dir, replace("weights", "1,2,3"), "3 values for 'weights', expected 6");
        assertRejected(dir, replace("mean", "1,2,3,4,5,6,7"), "7 values for 'mean', expected 6");
    }

    @Test
    void valuesThatWouldApproveEverythingAreRejected(@TempDir Path dir) {
        assertRejected(dir, replace("weights", "1,NaN,1,1,1,1"), "non-finite value for weights[1]");
        assertRejected(dir, replace("bias", "Infinity"), "non-finite value for bias");
        assertRejected(dir, replace("std", "1,1,0,1,1,1"), "non-positive std for log distance");
        assertRejected(dir, replace("threshold", "1.5"), "threshold outside 0-1");
        assertRejected(dir, replace("threshold", "-0.1"), "threshold outside 0-1");
    }

    private static void assertRejected(Path dir, List<String> lines, String expected) {
        IOException e = assertThrows(IOException.class, () -> FraudDetector.load(write(dir, lines)));
        assertTrue(e.getMessage().contains(expected), "expected '" + expected + "' in: " + e.getMessage());
    }

    private static List<String> without(String key) {
        return saved.stream().filter(l -> !l.startsWith(key + "=")).toList();
    }

    private static List<String> replace(String key, String value) {
        return saved.stream().map(l -> l.startsWith(key + "=") ? key + "=" + value : l).toList();
    }

    private static Path write(Path dir, List<String> lines) throws IOException {
        Path file = Files.createTempFile(dir, "model", ".properties");
        Files.write(file, lines, StandardCharsets.UTF_8);
        return file;
    }
}
