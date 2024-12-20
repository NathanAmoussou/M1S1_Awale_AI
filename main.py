import math
import time

class AwaleGame:
    def __init__(self, player1_is_human=True, player2_is_human=True):
        # Initial board: each hole has [2 red, 2 blue]
        self.board = [[2, 2] for _ in range(16)]
        # Scores for players [player1_score, player2_score]
        self.scores = [0, 0]
        # Player 1 controls even indices (0,2,4,...), Player 2 controls odd indices (1,3,5,...)
        self.player_holes = {
            1: [i for i in range(0, 16, 2)],
            2: [i for i in range(1, 16, 2)]
        }
        self.current_player = 1

        # Set whether each player is human or AI
        self.player1_is_human = player1_is_human
        self.player2_is_human = player2_is_human

    def display_board(self):
        """Display the board in a simple linear fashion."""
        print("\nPlateau (dans l'ordre horaire) :")
        print(" " + "     ".join(f"{i+1:2}" for i in range(16)))
        print(" " + " ".join(f"{hole}" for hole in self.board))
        print(f"\nScores: Joueur 1 = {self.scores[0]}, Joueur 2 = {self.scores[1]}")

    def is_valid_move(self, hole, color):
        if hole not in self.player_holes[self.current_player]:
            return False
        if color not in [0, 1]:
            return False
        if self.board[hole][color] == 0:
            return False
        return True

    def play_move(self, hole, color):
        if not self.is_valid_move(hole, color):
            raise ValueError("Mouvement invalide !")

        seeds_to_sow = self.board[hole][color]
        self.board[hole][color] = 0

        initial_hole = hole
        current_index = hole

        while seeds_to_sow > 0:
            current_index = (current_index + 1) % 16
            if current_index == initial_hole:
                continue

            if color == 0:
                if current_index in self.player_holes[self.current_player]:
                    continue
                self.board[current_index][color] += 1
                seeds_to_sow -= 1
            else:
                self.board[current_index][color] += 1
                seeds_to_sow -= 1

        self.apply_capture(current_index)
        self.current_player = 3 - self.current_player

    def apply_capture(self, start_hole):
        current_index = start_hole
        while True:
            total_seeds = sum(self.board[current_index])
            if total_seeds in [2, 3]:
                self.scores[self.current_player - 1] += total_seeds
                self.board[current_index] = [0, 0]
                current_index = (current_index - 1) % 16
            else:
                break

    def game_over(self):
        total_seeds = sum(sum(hole) for hole in self.board)
        if total_seeds < 8:
            return True
        if self.scores[0] >= 33 or self.scores[1] >= 33:
            return True
        if self.scores[0] == 32 and self.scores[1] == 32:
            return True
        return False

    def get_winner(self):
        if self.scores[0] > self.scores[1]:
            return "Joueur 1"
        elif self.scores[1] > self.scores[0]:
            return "Joueur 2"
        else:
            return "Égalité"

    def clone(self):
        new_game = AwaleGame(self.player1_is_human, self.player2_is_human)
        new_game.board = [h[:] for h in self.board]
        new_game.scores = self.scores[:]
        new_game.current_player = self.current_player
        return new_game

    def get_valid_moves(self):
        moves = []
        for hole in self.player_holes[self.current_player]:
            for color in [0, 1]:
                if self.is_valid_move(hole, color):
                    moves.append((hole, color))
        return moves

    def evaluate(self):
        if self.current_player == 1:
            return self.scores[0] - self.scores[1]
        else:
            return self.scores[1] - self.scores[0]

    def minimax(self, depth, alpha, beta, maximizing_player):
        if self.game_over() or depth == 0:
            return self.evaluate(), None

        moves = self.get_valid_moves()
        if not moves:
            return self.evaluate(), None

        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                clone_state = self.clone()
                clone_state.play_move(move[0], move[1])
                eval_val, _ = clone_state.minimax(depth - 1, alpha, beta, False)
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                clone_state = self.clone()
                clone_state.play_move(move[0], move[1])
                eval_val, _ = clone_state.minimax(depth - 1, alpha, beta, True)
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def best_move(self, max_time=2):
        """Find the best move using iterative deepening within max_time seconds."""
        start_time = time.time()
        depth = 1
        best_move = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                break

            try:
                _, move = self.minimax(depth, -math.inf, math.inf, True)
                if move is not None:
                    best_move = move
            except Exception:
                break  # If something goes wrong, return the last best move

            depth += 1

        print(f"Temps de calcul : {time.time() - start_time:.2f}s, profondeur atteinte : {depth - 1}")
        return best_move


if __name__ == "__main__":
    player1_is_human = True
    player2_is_human = False

    game = AwaleGame(player1_is_human=player1_is_human, player2_is_human=player2_is_human)
    game.display_board()

    while not game.game_over():
        print(f"\nTour du Joueur {game.current_player}")

        if (game.current_player == 1 and game.player1_is_human) or (game.current_player == 2 and game.player2_is_human):
            hole = int(input("Choisissez un trou (1-16) : ")) - 1
            color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
            try:
                game.play_move(hole, color)
            except ValueError as e:
                print(e)
                continue
        else:
            move = game.best_move(max_time=2)
            if move is None:
                break
            print(f"L'IA (Joueur {game.current_player}) joue le trou {move[0]+1}, couleur {'Rouge' if move[1] == 0 else 'Bleu'}.")
            game.play_move(move[0], move[1])

        game.display_board()

    print(f"\nPartie terminée ! Le gagnant est : {game.get_winner()}")
