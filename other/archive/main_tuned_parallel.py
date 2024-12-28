import torch
import math
import time
import random
from typing import List, Tuple

class AwaleState:
    def __init__(self, board=None, scores=None, current_player=1):
        self.board = board if board else [[2, 2] for _ in range(16)]
        self.scores = scores if scores else [0, 0]
        self.current_player = current_player

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        moves = []
        holes = range(1, 16, 2) if self.current_player == 1 else range(0, 16, 2)
        for hole in holes:
            for color in [0, 1]:
                if self.board[hole][color] > 0:
                    moves.append((hole, color))
        return moves

    def make_move(self, move: Tuple[int, int]) -> 'AwaleState':
        hole, color = move
        seeds = self.board[hole][color]
        self.board[hole][color] = 0
        index = hole

        while seeds > 0:
            index = (index + 1) % 16
            if index != hole:
                if color == 0 and index % 2 == self.current_player - 1:
                    continue
                self.board[index][color] += 1
                seeds -= 1

        self.capture(index)
        self.current_player = 3 - self.current_player
        return self

    def capture(self, index):
        while sum(self.board[index]) in [2, 3]:
            self.scores[self.current_player - 1] += sum(self.board[index])
            self.board[index] = [0, 0]
            index = (index - 1) % 16

    def clone(self):
        return AwaleState([hole[:] for hole in self.board], self.scores[:], self.current_player)

@torch.jit.script
def evaluate_batch(state_tensor: torch.Tensor, scores_tensor: torch.Tensor, current_player: int) -> torch.Tensor:
    batch_size = state_tensor.size(0)
    device = state_tensor.device

    if current_player == 1:
        base_score = (scores_tensor[:, 0] - scores_tensor[:, 1]) * 10
    else:
        base_score = (scores_tensor[:, 1] - scores_tensor[:, 0]) * 10

    bonus = torch.zeros(batch_size, device=device)
    opponent_holes = [1, 3, 5, 7, 9, 11, 13, 15] if current_player == 1 else [0, 2, 4, 6, 8, 10, 12, 14]

    for hole in opponent_holes:
        hole_sum = state_tensor[:, hole, 0] + state_tensor[:, hole, 1]
        vulnerable = ((hole_sum == 1) | (hole_sum == 2)).float() * 2
        bonus += vulnerable

    return base_score + bonus

@torch.jit.script
def order_moves_batch(
    state_tensor: torch.Tensor,
    scores_tensor: torch.Tensor,
    moves: List[Tuple[int, int]],
    current_player: int
) -> List[Tuple[int, int]]:
    batch_size = state_tensor.size(0)
    device = state_tensor.device
    move_scores = torch.zeros(len(moves), device=device)

    for i, move in enumerate(moves):
        new_states = make_move_batch(state_tensor, move, current_player)
        new_scores = scores_tensor.clone()

        if current_player == 1:
            score_gain = new_scores[:, 0] - scores_tensor[:, 0]
        else:
            score_gain = new_scores[:, 1] - scores_tensor[:, 1]

        move_scores[i] = torch.mean(score_gain)

    sorted_indices = torch.argsort(move_scores, descending=True)
    indices_list: List[int] = sorted_indices.tolist()
    return [moves[i] for i in indices_list]

@torch.jit.script
def get_valid_moves_batch(state_tensor: torch.Tensor, current_player: int) -> List[Tuple[int, int]]:
    moves: List[Tuple[int, int]] = []

    # Define player-controlled holes
    player_1_holes = [0, 2, 4, 6, 8, 10, 12, 14]  # Odd holes for Player 1
    player_2_holes = [1, 3, 5, 7, 9, 11, 13, 15]  # Even holes for Player 2

    # Select the appropriate holes based on the current player
    controlled_holes = player_1_holes if current_player == 1 else player_2_holes

    # Iterate through holes and colors
    for hole in controlled_holes:
        for color in range(2):  # Colors: 0 (red), 1 (blue)
            if state_tensor[0, hole, color] > 0:  # Check seed count
                moves.append((hole, color))

    return moves

@torch.jit.script
def make_move_batch(state_tensor: torch.Tensor, move: Tuple[int, int], player: int) -> torch.Tensor:
    new_states = state_tensor.clone()
    batch_size = state_tensor.size(0)
    for b in range(batch_size):
        hole, color = move
        seeds = new_states[b, hole, color]
        new_states[b, hole, color] = 0
        index = hole
        while seeds > 0:
            index = (index + 1) % 16
            if index != hole:
                new_states[b, index, color] += 1
                seeds -= 1
    return new_states

@torch.jit.script
def parallel_minimax_layer(
    state_tensor: torch.Tensor,
    scores_tensor: torch.Tensor,
    depth: int,
    maximizing_player: bool
) -> torch.Tensor:
    batch_size = state_tensor.size(0)
    device = state_tensor.device

    if depth == 0:
        return evaluate_batch(state_tensor, scores_tensor, 1 if maximizing_player else 2)

    valid_moves = get_valid_moves_batch(state_tensor, 1 if maximizing_player else 2)
    if len(valid_moves) == 0:
        return evaluate_batch(state_tensor, scores_tensor, 1 if maximizing_player else 2)

    ordered_moves = order_moves_batch(state_tensor, scores_tensor, valid_moves, 1 if maximizing_player else 2)

    if maximizing_player:
        value = torch.full((batch_size,), float('-inf'), device=device)
        for move in ordered_moves:
            new_states = make_move_batch(state_tensor, move, 1)
            move_value = evaluate_batch(new_states, scores_tensor, 2)  # Evaluate from opponent's perspective
            value = torch.max(value, move_value)
    else:
        value = torch.full((batch_size,), float('inf'), device=device)
        for move in ordered_moves:
            new_states = make_move_batch(state_tensor, move, 2)
            move_value = evaluate_batch(new_states, scores_tensor, 1)  # Evaluate from opponent's perspective
            value = torch.min(value, move_value)

    return value

@torch.jit.script
def parallel_minimax(state_tensor: torch.Tensor, scores_tensor: torch.Tensor, max_depth: int) -> torch.Tensor:
    batch_size = state_tensor.size(0)
    device = state_tensor.device

    best_value = torch.zeros(batch_size, device=device)

    # Evaluate each depth level iteratively
    for depth in range(max_depth):
        value = parallel_minimax_layer(state_tensor, scores_tensor, depth, True)
        best_value = torch.max(best_value, value)

    return best_value

def benchmark_awale(num_games: int = 100, max_depth: int = 4):
    print(f"Benchmarking Awale with {num_games} games at depth {max_depth}")

    games = []
    for _ in range(num_games):
        state = AwaleState()
        games.append(state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_states = torch.tensor([[[hole[color] for color in range(2)] for hole in game.board] for game in games],
                                 dtype=torch.float32, device=device)
    tensor_scores = torch.tensor([game.scores for game in games], dtype=torch.float32, device=device)

    start_time = time.time()
    results = parallel_minimax(tensor_states, tensor_scores, max_depth)
    parallel_time = time.time() - start_time

    print(f"Parallel Time: {parallel_time:.4f} seconds")
    return results

class AwaleGame:
    def __init__(self, player1_type="human", player2_type="human"):
        self.board = [[2, 2] for _ in range(16)]
        self.scores = [0, 0]
        self.player_holes = {
            1: [i for i in range(0, 16, 2)],
            2: [i for i in range(1, 16, 2)]
        }
        self.current_player = 1
        self.player_types = {
            1: player1_type,
            2: player2_type
        }

    def display_board(self):
        print("\nPlateau (dans l'ordre horaire) :")
        print(" " + "     ".join(f"{i+1:2}" for i in range(16)))
        print(" " + " ".join(f"{hole}" for hole in self.board))
        print(f"\nScores: Joueur 1 = {self.scores[0]}, Joueur 2 = {self.scores[1]}")

    def is_valid_move(self, hole, color):
        if hole is None or color is None:
            return False
        if hole not in self.player_holes[self.current_player]:
            return False
        if color not in [0, 1]:
            return False
        if self.board[hole][color] == 0:
            return False
        return True

    def get_valid_moves(self):
        moves = []
        for hole in self.player_holes[self.current_player]:
            for color in [0, 1]:
                if self.board[hole][color] > 0:
                    moves.append((hole, color))
        return moves

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

    def get_move_for_current_player(self):
        ptype = self.player_types[self.current_player]

        if ptype == "human":
            hole = int(input("Choisissez un trou (1-16) : ")) - 1
            color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
            return hole, color

        elif ptype == "ai_minimax":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor_state = torch.tensor([[[hole[color] for color in range(2)] for hole in self.board]],
                                        dtype=torch.float32, device=device)
            tensor_scores = torch.tensor([self.scores], dtype=torch.float32, device=device)
            max_time = 2  # seconds
            start_time = time.time()

            depth = 1
            best_move = None
            while time.time() - start_time < max_time:
                try:
                    values = parallel_minimax(tensor_state, tensor_scores, depth)
                    valid_moves = get_valid_moves_batch(tensor_state, self.current_player)
                    if valid_moves:
                        best_move = valid_moves[torch.argmax(values).item()]
                    depth += 1
                except:
                    break

            print(f"Profondeur atteinte : {depth - 1}, Temps de calcul : {time.time() - start_time:.2f}s")
            return best_move

        elif ptype == "ai_random":
            valid_moves = self.get_valid_moves()
            return random.choice(valid_moves) if valid_moves else None

    def run_game(self):
        turn_counter = 0
        self.display_board()

        while not self.game_over():
            turn_counter += 1
            print(f"\nTour n°{turn_counter}, Joueur {self.current_player}")

            move = self.get_move_for_current_player()

            if move is None:
                break

            hole, color = move
            try:
                self.play_move(hole, color)
            except ValueError as e:
                print(e)
                turn_counter -= 1
                continue

            self.display_board()

        print(f"\nPartie terminée en {turn_counter} tours ! Le gagnant est : {self.get_winner()}")

if __name__ == "__main__":
    player1_type = "ai_minimax"
    player2_type = "ai_random"

    game = AwaleGame(player1_type=player1_type, player2_type=player2_type)
    game.run_game()
