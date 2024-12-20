package game.player;

public class HumanPlayer extends Player {

    public HumanPlayer(String name, int[] controlledHoles) {
        super(name, controlledHoles);
    }

    @Override
    public void makeMove(Board board) {
        // Prompt the user for input and validate the move.
        // Example:
        // - Ask for hole number and color.
        // - Validate that the hole is controlled by the player and not empty.
    }
}
