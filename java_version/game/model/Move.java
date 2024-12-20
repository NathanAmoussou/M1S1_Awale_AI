package game.model;

public class Move {

    private int holeNumber;
    private Color seedColor;

    public Move(int holeNumber, Color seedColor) {
        this.holeNumber = holeNumber;
        this.seedColor = seedColor;
    }

    public int getHoleNumber() {
        return holeNumber;
    }

    public Color getSeedColor() {
        return seedColor;
    }
}
