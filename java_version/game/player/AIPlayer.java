package game.player;

import game.model.Board; // Add this import statement

public class AIPlayer extends Player {

    public AIPlayer(String name, int[] controlledHoles) {
        super(name, controlledHoles);
    }

    @Override
    public void makeMove(Board board) {
        // Implement AI logic to select the best move.
    }
}
