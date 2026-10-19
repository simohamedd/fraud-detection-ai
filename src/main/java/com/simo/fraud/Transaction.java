package com.simo.fraud;

/**
 * A card transaction described by the signals a fraud team would look at.
 *
 * @param amount            value in pounds
 * @param hourOfDay         0-23, local time of the purchase
 * @param distanceFromHomeKm how far the purchase happened from the cardholder's home
 * @param merchantRisk      0 (trusted) to 1 (high-risk merchant category)
 * @param online            card-not-present purchase
 * @param txLastHour        number of purchases on the card in the previous hour
 * @param fraud             known label (used for training and evaluation)
 */
public record Transaction(double amount, int hourOfDay, double distanceFromHomeKm, double merchantRisk,
                          boolean online, int txLastHour, boolean fraud) {

    public static final String[] FEATURES = {
        "log amount", "night time", "log distance", "merchant risk", "online", "tx last hour"
    };

    /** Turns the raw transaction into the numeric features the model uses. */
    public double[] features() {
        boolean night = hourOfDay < 6 || hourOfDay >= 23;
        return new double[] {
            Math.log1p(amount),
            night ? 1 : 0,
            Math.log1p(distanceFromHomeKm),
            merchantRisk,
            online ? 1 : 0,
            txLastHour
        };
    }
}
