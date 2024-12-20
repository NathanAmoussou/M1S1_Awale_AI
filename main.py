import math
import time
import random

class AwaleGame:
    def __init__(self, player1_type="human", player2_type="ai"):
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
        self.turn_number = 1

        # Set player types: "human", "ai", or "random"
        self.player1_type = player1_type
        self.player2_type = player2_type

    def display_state(self, move=None, ai_time=None, depth=None):
        """Compact display of game state."""
        board_state = " ".join(f"{hole}" for hole in self.board)
        info = f"Tour {self.turn_number} | Plateau: {board_state} | Scores: J1={self.scores[0]} J2={self.scores[1]} | Joueur {self.current_player}"
        if move:
            info += f" joue {move}"
        if ai_time is not None and depth is not None:
            info += f" | Temps IA: {ai_time:.2f}s | Profondeur: {depth}"
        print(info)

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
        self.turn_number += 1

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
        new_game = AwaleGame(self.player1_type, self.player2_type)
        new_game.board = [h[:] for h in self.board]
        new_game.scores = self.scores[:]
        new_game.current_player = self.current_player
        new_game.turn_number = self.turn_number
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
                break

            depth += 1

        elapsed_time = time.time() - start_time
        return best_move, elapsed_time, depth

    def random_move(self):
        moves = self.get_valid_moves()
        return random.choice(moves) if moves else None


if __name__ == "__main__":
    game = AwaleGame(player1_type="random", player2_type="ai")

    while not game.game_over():
        if game.current_player == 1 and game.player1_type == "human":
            game.display_state()
            hole = int(input("Choisissez un trou (1-16) : ")) - 1
            color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
            try:
                game.play_move(hole, color)
            except ValueError as e:
                print(e)
                continue
        elif game.current_player == 1 and game.player1_type == "random":
            move = game.random_move()
            game.play_move(move[0], move[1])
            game.display_state(f"Trou {move[0]+1}, Couleur {'Rouge' if move[1] == 0 else 'Bleu'}")
        elif game.current_player == 2 and game.player2_type == "human":
            game.display_state()
            hole = int(input("Choisissez un trou (1-16) : ")) - 1
            color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
            try:
                game.play_move(hole, color)
            except ValueError as e:
                print(e)
                continue
        elif game.current_player == 2 and game.player2_type == "random":
            move = game.random_move()
            game.play_move(move[0], move[1])
            game.display_state(f"Trou {move[0]+1}, Couleur {'Rouge' if move[1] == 0 else 'Bleu'}")
        else:
            move, ai_time, depth = game.best_move(max_time=2)
            game.play_move(move[0], move[1])
            game.display_state(f"Trou {move[0]+1}, Couleur {'Rouge' if move[1] == 0 else 'Bleu'}", ai_time, depth)

    print(f"\nPartie terminée ! Le gagnant est : {game.get_winner()}")
