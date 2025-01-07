from numba import njit, prange
import numpy as np
from typing import List, Tuple
from board_rules_interface_np import AwaleGame
import agents_np


@njit
def jit_apply_capture(board, start_hole, current_player, scores):
    """
    JIT-compiled version of apply_capture logic.
    board: a 2D int8 array of shape (16,2).
    scores: a 1D int16 array of length 2.
    current_player: int (1 or 2).
    """
    current_index = start_hole
    while True:
        total_seeds = board[current_index, 0] + board[current_index, 1]
        if total_seeds in (2, 3):
            # current_player - 1 => index 0 or 1
            scores[current_player - 1] += total_seeds
            board[current_index, 0] = 0
            board[current_index, 1] = 0
            current_index = (current_index - 1) % 16
        else:
            break

@njit
def jit_play_move(
    board,
    scores,
    hole,
    color,
    current_player,
    player1_holes,
    player2_holes
):
    """
    JIT-compiled function that performs the sowing logic for a single move.

    board:  (16,2) int8 array
    scores: (2,)   int16 array
    hole: int (hole index to sow from)
    color: int (0 or 1)
    current_player: int (1 or 2)
    player1_holes, player2_holes: arrays of hole indices for each player
    """

    # Identify which holes belong to the current player
    # We'll do a quick trick: if current_player == 1 => use player1_holes,
    # else use player2_holes
    if current_player == 1:
        my_holes = player1_holes
        opp_holes = player2_holes
    else:
        my_holes = player2_holes
        opp_holes = player1_holes

    seeds_to_sow = board[hole, color]
    board[hole, color] = 0

    # Create a boolean mask for opponent's holes
    # We'll do it manually since we can’t store dynamic masks easily in Numba
    # 16 holes => create a bool array of length 16
    opponent_mask = np.zeros(16, dtype=np.bool_)
    for h in opp_holes:
        opponent_mask[h] = True

    current_index = hole
    while seeds_to_sow > 0:
        current_index = (current_index + 1) % 16
        # Skip the original hole
        if current_index == hole:
            continue

        if color == 0:  # Red seeds
            if opponent_mask[current_index]:
                board[current_index, color] += 1
                seeds_to_sow -= 1
        else:  # Blue seeds
            board[current_index, color] += 1
            seeds_to_sow -= 1

    # Now apply capture
    jit_apply_capture(board, current_index, current_player, scores)

    # Switch player
    return 3 - current_player

@njit
def jit_clone(board, scores, current_player):
    """
    JIT-compiled function to clone the board + scores + current player.
    Returns a new (board_copy, scores_copy, current_player_copy).
    """
    board_copy = board.copy()       # Numba supports .copy() on NumPy arrays
    scores_copy = scores.copy()     # This is a small 2-element array
    current_player_copy = current_player
    return board_copy, scores_copy, current_player_copy

class AwaleGameJIT:
    """
    A drop-in replacement for AwaleGame that uses Numba-accelerated
    play_move(...) and clone().
    For brevity, we only implement the parts needed by the agent:
      - board, scores, current_player
      - get_valid_moves, is_valid_move
      - play_move (JIT)
      - clone (JIT)
      - game_over, etc.
    NOTE: We omit display methods or other logic to keep it short.
    """

    def __init__(self, player1_agent, player2_agent, game_id=None):
        self.board = np.zeros((16, 2), dtype=np.int8)
        self.board.fill(2)
        self.scores = np.zeros(2, dtype=np.int16)

        # Precompute holes for each player
        self.player1_holes = np.arange(0, 16, 2, dtype=np.int8)
        self.player2_holes = np.arange(1, 16, 2, dtype=np.int8)

        self.current_player = 1
        self.player_agents = {1: player1_agent, 2: player2_agent}
        self.game_id = game_id
        self.turn_number = 0

    def is_valid_move(self, hole: int, color: int) -> bool:
        if hole is None or color is None:
            return False
        if self.current_player == 1:
            if hole not in self.player1_holes:
                return False
        else:
            if hole not in self.player2_holes:
                return False

        if color not in (0, 1):
            return False
        if self.board[hole, color] <= 0:
            return False
        return True

    def get_valid_moves(self) -> List[Tuple[int, int]]:
        if self.current_player == 1:
            holes = self.player1_holes
        else:
            holes = self.player2_holes

        moves = []
        # Red
        red_holes = holes[self.board[holes, 0] > 0]
        # Blue
        blue_holes = holes[self.board[holes, 1] > 0]
        for h in red_holes:
            moves.append((h, 0))
        for h in blue_holes:
            moves.append((h, 1))

        return moves

    def play_move(self, hole: int, color: int):
        if not self.is_valid_move(hole, color):
            raise ValueError("Invalid move!")

        # JIT function returns the new current player
        new_player = jit_play_move(
            self.board,
            self.scores,
            hole,
            color,
            self.current_player,
            self.player1_holes,
            self.player2_holes
        )
        self.current_player = new_player

    def clone(self):
        board_copy, scores_copy, cp_copy = jit_clone(self.board, self.scores, self.current_player)
        clone_game = AwaleGameJIT(
            player1_agent=self.player_agents[1],
            player2_agent=self.player_agents[2],
            game_id=self.game_id
        )
        clone_game.board = board_copy
        clone_game.scores = scores_copy
        clone_game.current_player = cp_copy
        clone_game.turn_number = self.turn_number
        return clone_game

    def game_over(self) -> bool:
        total_seeds = np.sum(self.board)
        return (
            total_seeds < 8
            or np.max(self.scores) >= 33
            or (self.scores[0] == 32 and self.scores[1] == 32)
            or self.turn_number >= 150
        )

    # For evaluation, you can reuse your same logic from the old AwaleGame
    # or define one directly here. The agent calls game_state.evaluate(...), so:
    def GPT_evaluate_V2(self) -> int:
        # Example or your old logic
        # minimal example:
        return 0  # or your custom

    def claude_evaluate_V1(self) -> int:
        return 0

    def run_game(self):
        # Minimal or skip if you want
        pass

    # etc. (You can add any other needed methods from your original AwaleGame)

class MinimaxAgent6_4_JIT(agents_np.MinimaxAgent6_4):
    """
    Identical to MinimaxAgent6_4, but we assume the game_state is an AwaleGameJIT
    so that clone() and play_move() are JIT-compiled under the hood.
    """
    def __init__(self, max_time=2):
        super().__init__(max_time)
        # Optionally do anything extra if needed

agent1 = MinimaxAgent6_4_JIT(max_time=2)
agent2 = MinimaxAgent6_4_JIT(max_time=2)

game = AwaleGameJIT(player1_agent=agent1, player2_agent=agent2)
game.run_game()
