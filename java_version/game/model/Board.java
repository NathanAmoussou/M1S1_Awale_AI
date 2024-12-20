package game.model;

public class Board {

    private Hole[] holes;

    public Board() {
        holes = new Hole[16];
        for (int i = 0; i < 16; i++) {
            holes[i] = new Hole(i + 1);
        }
    }

    public Hole getHole(int holeNumber) {
        return holes[(holeNumber - 1) % 16];
    }

    public void displayBoard() {
        // Implement a method to display the board state in the console.
    }
}
