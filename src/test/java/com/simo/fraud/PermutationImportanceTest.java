package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class PermutationImportanceTest {

    private static final List<Transaction> DATA = SyntheticData.generate(8_000, 11);
    private static final List<Transaction> TRAIN = DATA.subList(0, 5_000);
    private static final List<Transaction> TEST = DATA.subList(5_000, 8_000);

    @Test
    void reportsEveryFeatureOnceMostImportantFirst() {
        List<FraudDetector.Importance> imp = FraudDetector.train(TRAIN, 11).permutationImportance(TEST, 3, 1);
        assertEquals(Transaction.FEATURES.length, imp.size());
        assertEquals(Arrays.stream(Transaction.FEATURES).sorted().toList(),
                imp.stream().map(FraudDetector.Importance::feature).sorted().toList());
        for (int i = 1; i < imp.size(); i++) assertTrue(imp.get(i - 1).drop() >= imp.get(i).drop());
        assertTrue(imp.get(0).drop() > 0.01, "the top signal should matter, got " + imp.get(0));
    }

    @Test
    void sameSeedGivesTheSameResult() {
        FraudDetector d = FraudDetector.train(TRAIN, 11);
        assertEquals(d.permutationImportance(TEST, 2, 5), d.permutationImportance(TEST, 2, 5));
    }

    @Test
    void featuresTheModelIgnoresHaveNoImportance(@TempDir Path dir) throws IOException {
        Path file = dir.resolve("model.properties");
        FraudDetector.train(TRAIN, 11).save(file);
        List<String> lines = Files.readAllLines(file, StandardCharsets.UTF_8).stream()
                .map(l -> l.startsWith("weights=") ? "weights=0,0,0,3,0,0" : l).toList();
        Files.write(file, lines, StandardCharsets.UTF_8);

        Map<String, Double> drop = FraudDetector.load(file).permutationImportance(TEST, 2, 3).stream()
                .collect(Collectors.toMap(FraudDetector.Importance::feature, FraudDetector.Importance::drop));
        assertTrue(drop.get("merchant risk") > 0.01, "merchant risk drop was " + drop.get("merchant risk"));
        for (String f : Transaction.FEATURES) {
            if (!f.equals("merchant risk")) assertEquals(0, drop.get(f), 1e-12, f);
        }
    }

    @Test
    void rejectsBadArguments() {
        FraudDetector d = FraudDetector.train(TRAIN, 11);
        assertThrows(IllegalArgumentException.class, () -> d.permutationImportance(TEST, 0, 1));
        List<Transaction> genuineOnly = TEST.stream().filter(t -> !t.fraud()).toList();
        assertThrows(IllegalArgumentException.class, () -> d.permutationImportance(genuineOnly, 2, 1));
        assertThrows(IllegalStateException.class, () -> new FraudDetector().permutationImportance(TEST, 2, 1));
    }
}
