package game.player;

public abstract class Player {

    protected String name;
    protected int[] controlledHoles;
    protected Map<Color, Integer> capturedSeeds;

    public Player(String name, int[] controlledHoles) {
        this.name = name;
        this.controlledHoles = controlledHoles;
        capturedSeeds = new EnumMap<>(Color.class);
        capturedSeeds.put(Color.RED, 0);
        capturedSeeds.put(Color.BLUE, 0);
    }

    public abstract void makeMove(Board board);

    public void captureSeeds(Map<Color, Integer> seeds) {
        for (Color color : Color.values()) {
            int count = seeds.getOrDefault(color, 0);
            capturedSeeds.put(color, capturedSeeds.get(color) + count);
        }
    }

    public int getTotalCapturedSeeds() {
        return capturedSeeds.get(Color.RED) + capturedSeeds.get(Color.BLUE);
    }

    public String getName() {
        return name;
    }

    public int[] getControlledHoles() {
        return controlledHoles;
    }
}
