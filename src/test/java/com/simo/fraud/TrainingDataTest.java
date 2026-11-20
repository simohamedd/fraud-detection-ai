package com.simo.fraud;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class TrainingDataTest {

    private static Transaction genuine() {
        return new Transaction(20, 13, 2, 0.1, false, 0, false);
    }

    private static Transaction fraud() {
        return new Transaction(900, 3, 800, 0.9, true, 5, true);
    }

    @Test
    void emptyTrainingDataIsRejected() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> FraudDetector.train(List.of(), 1));
        assertEquals("Training data is empty", e.getMessage());
    }

    @Test
    void trainingNeedsBothClasses() {
        IllegalArgumentException noFraud = assertThrows(IllegalArgumentException.class,
                () -> FraudDetector.train(List.of(genuine(), genuine()), 1));
        assertTrue(noFraud.getMessage().contains("no fraud"), noFraud.getMessage());
        IllegalArgumentException noGenuine = assertThrows(IllegalArgumentException.class,
                () -> FraudDetector.train(List.of(fraud(), fraud()), 1));
        assertTrue(noGenuine.getMessage().contains("no genuine"), noGenuine.getMessage());
    }

    @Test
    void oneOfEachIsEnoughToTrain() {
        FraudDetector d = FraudDetector.train(List.of(genuine(), fraud()), 1);
        assertTrue(d.score(fraud()) > d.score(genuine()));
    }

    @Test
    void failedCalibrationKeepsThePreviousThreshold() {
        List<Transaction> all = SyntheticData.generate(6_000, 9);
        FraudDetector d = FraudDetector.train(all.subList(0, 4_000), 9);
        d.calibrate(all.subList(4_000, 6_000));
        double calibrated = d.threshold();

        assertThrows(IllegalArgumentException.class, () -> d.calibrate(List.of()));
        assertThrows(IllegalArgumentException.class, () -> d.calibrate(List.of(genuine(), genuine())));
        assertEquals(calibrated, d.threshold(), 0);
    }
}
