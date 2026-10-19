package com.simo.fraud;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Generates realistic, imbalanced card data (about 3% fraud). Fraudulent purchases tend to be larger,
 * at night, far from home, online, at risky merchants and in quick bursts, but with plenty of overlap,
 * so the model has to learn rather than read off a single rule.
 */
public final class SyntheticData {
    private SyntheticData() {
    }

    public static List<Transaction> generate(int n, long seed) {
        Random r = new Random(seed);
        List<Transaction> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            boolean fraud = r.nextDouble() < 0.03;
            out.add(fraud ? fraudulent(r) : genuine(r));
        }
        return out;
    }

    private static Transaction genuine(Random r) {
        double amount = Math.exp(3.0 + 0.9 * r.nextGaussian());
        int hour = clampHour((int) Math.round(14 + 4 * r.nextGaussian()));
        double distance = Math.abs(8 * r.nextGaussian()) + (r.nextDouble() < 0.05 ? 300 * r.nextDouble() : 0);
        double risk = clamp01(0.2 + 0.15 * r.nextGaussian());
        boolean online = r.nextDouble() < 0.35;
        int burst = poisson(r, 0.4);
        return new Transaction(round2(amount), hour, round2(distance), round2(risk), online, burst, false);
    }

    private static Transaction fraudulent(Random r) {
        double amount = Math.exp(4.6 + 1.0 * r.nextGaussian());
        int hour = r.nextDouble() < 0.55 ? r.nextInt(6) : clampHour((int) Math.round(15 + 5 * r.nextGaussian()));
        double distance = r.nextDouble() < 0.6 ? 100 + 900 * r.nextDouble() : Math.abs(15 * r.nextGaussian());
        double risk = clamp01(0.55 + 0.2 * r.nextGaussian());
        boolean online = r.nextDouble() < 0.75;
        int burst = poisson(r, 2.5);
        return new Transaction(round2(amount), hour, round2(distance), round2(risk), online, burst, true);
    }

    private static int poisson(Random r, double lambda) {
        double l = Math.exp(-lambda), p = 1;
        int k = 0;
        do { k++; p *= r.nextDouble(); } while (p > l);
        return k - 1;
    }

    private static int clampHour(int h) { return Math.floorMod(h, 24); }
    private static double clamp01(double x) { return Math.max(0, Math.min(1, x)); }
    private static double round2(double x) { return Math.round(x * 100) / 100.0; }
}
