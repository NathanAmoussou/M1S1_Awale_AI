# numba_func.py

from numba import njit
import numpy as np

@njit
def get_valid_moves_numba(board, current_player):
    moves = []
    player_holes = np.arange(0, 16, 2) if current_player == 1 else np.arange(1, 16, 2)
    for hole in player_holes:
        if board[hole, 0] > 0:
            moves.append((hole, 0))
        if board[hole, 1] > 0:
            moves.append((hole, 1))
    return moves

@njit
def play_move_numba(board, scores, current_player, hole, color):
    seeds_to_sow = board[hole, color]
    board[hole, color] = 0
    current_index = hole
    while seeds_to_sow > 0:
        current_index = (current_index + 1) % 16
        if current_index == hole:
            continue
        if color == 0:  # Red seeds
            if (current_player == 2 and current_index % 2 == 1) or (current_player == 1 and current_index % 2 == 0):
                board[current_index, color] += 1
                seeds_to_sow -= 1
        else:  # Blue seeds
            board[current_index, color] += 1
            seeds_to_sow -= 1
    # Apply capture
    current_index = (current_index - 1) % 16
    while True:
        total_seeds = board[current_index, 0] + board[current_index, 1]
        if total_seeds in [2, 3]:
            scores[current_player - 1] += total_seeds
            board[current_index, 0] = 0
            board[current_index, 1] = 0
            current_index = (current_index - 1) % 16
        else:
            break

@njit
def evaluate_position_numba(board, scores, current_player, SCORE_WEIGHT, CONTROL_WEIGHT, CAPTURE_WEIGHT, MOBILITY_WEIGHT):
    my_index = current_player - 1
    opp_index = 1 - my_index

    # Score difference
    score_diff = scores[my_index] - scores[opp_index]

    # Board control
    my_holes = np.arange(0, 16, 2) if current_player == 1 else np.arange(1, 16, 2)
    opp_holes = np.arange(1, 16, 2) if current_player == 1 else np.arange(0, 16, 2)

    my_seeds = np.sum(board[my_holes])
    opp_seeds = np.sum(board[opp_holes])

    # Capture potential
    capture_positions = np.zeros(16, dtype=np.int8)
    for i in range(16):
        total_seeds = board[i, 0] + board[i, 1]
        if total_seeds == 1 or total_seeds == 4:
            capture_positions[i] = 1

    my_capture_potential = np.sum(capture_positions[my_holes])
    opp_capture_potential = np.sum(capture_positions[opp_holes])

    # Mobility
    my_mobility = np.sum(board[my_holes] > 0)
    opp_mobility = np.sum(board[opp_holes] > 0)

    return (SCORE_WEIGHT * score_diff +
            CONTROL_WEIGHT * (my_seeds - opp_seeds) +
            CAPTURE_WEIGHT * (my_capture_potential - opp_capture_potential) +
            MOBILITY_WEIGHT * (my_mobility - opp_mobility))
