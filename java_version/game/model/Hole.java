package game.model;

import java.util.EnumMap;
import java.util.Map;

public class Hole {

    private int holeNumber;
    private Map<Color, Integer> seedCounts;

    public Hole(int holeNumber) {
        this.holeNumber = holeNumber;
        seedCounts = new EnumMap<>(Color.class);
        seedCounts.put(Color.RED, 2);
        seedCounts.put(Color.BLUE, 2);
    }

    public int getHoleNumber() {
        return holeNumber;
    }

    public int getSeedCount(Color color) {
        return seedCounts.getOrDefault(color, 0);
    }

    public void addSeeds(Color color, int count) {
        seedCounts.put(color, getSeedCount(color) + count);
    }

    public void removeSeeds(Color color, int count) {
        seedCounts.put(color, getSeedCount(color) - count);
    }

    public void removeAllSeeds() {
        seedCounts.put(Color.RED, 0);
        seedCounts.put(Color.BLUE, 0);
    }

    public boolean isEmpty() {
        return getTotalSeedCount() == 0;
    }

    public int getTotalSeedCount() {
        return getSeedCount(Color.RED) + getSeedCount(Color.BLUE);
    }

    public Map<Color, Integer> getSeedCounts() {
        return new EnumMap<>(seedCounts);
    }
}
