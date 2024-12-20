package game;

public class Game {

    private Board board;
    private Player player1;
    private Player player2;
    private Player currentPlayer;

    public Game() {
        board = new Board();
        // Initialize players with their respective controlled holes.
        player1 = new HumanPlayer(
            "Player 1",
            new int[] { 1, 3, 5, 7, 9, 11, 13, 15 }
        );
        player2 = new AIPlayer(
            "Player 2",
            new int[] { 2, 4, 6, 8, 10, 12, 14, 16 }
        );
        currentPlayer = player1;
    }

    public void startGame() {
        while (!isGameOver()) {
            board.displayBoard();
            System.out.println(currentPlayer.getName() + "'s turn.");
            currentPlayer.makeMove(board);
            switchCurrentPlayer();
        }
        declareWinner();
    }

    private void switchCurrentPlayer() {
        currentPlayer = (currentPlayer == player1) ? player2 : player1;
    }

    private boolean isGameOver() {
        // Implement logic to check if the game is over according to the rules.
        return false;
    }

    private void declareWinner() {
        // Compare captured seeds and declare the winner.
    }
}
