from numba import njit
import numpy as np

@njit
def fast_claude_evaluate_V1(board, scores, current_player, player_holes):
    if current_player == 1:
        my_holes = player1_holes
        opp_holes = player2_holes
    else:
        my_holes = player2_holes
        opp_holes = player1_holes

    my_index = current_player - 1
    opp_index = 1 - my_index

    # 1. Score difference
    score_diff = scores[my_index] - scores[opp_index]

    # 2. Control of the board
    my_holes = player_holes[current_player]
    opp_holes = player_holes[2 - current_player]

    my_seeds = 0
    for h in my_holes:
        my_seeds += board[h][0] + board[h][1]

    opp_seeds = 0
    for h in opp_holes:
        opp_seeds += board[h][0] + board[h][1]

    # 3. Capture opportunities
    capture_potential = 0
    for h in opp_holes:
        total = board[h][0] + board[h][1]
        if 1 <= total <= 3:
            capture_potential -= 2

    # 4. Mobility score
    my_mobility = 0
    for h in my_holes:
        if board[h][0] > 0 or board[h][1] > 0:
            my_mobility += 1

    opp_mobility = 0
    for h in opp_holes:
        if board[h][0] > 0 or board[h][1] > 0:
            opp_mobility += 1

    mobility_score = my_mobility - opp_mobility

    # 5. Seed distribution
    my_distribution = 0
    for h in my_holes:
        if board[h][0] > 0 or board[h][1] > 0:
            my_distribution += 1

    opp_distribution = 0
    for h in opp_holes:
        if board[h][0] > 0 or board[h][1] > 0:
            opp_distribution += 1

    distribution_score = my_distribution - opp_distribution

    # 6. End game considerations
    total_seeds = 0
    for h in range(16):
        total_seeds += board[h][0] + board[h][1]

    if total_seeds < 16:
        score_diff_weight = 100
    else:
        score_diff_weight = 50

    # Weighted sum
    evaluation = (
        score_diff_weight * score_diff +
        30 * (my_seeds - opp_seeds) +
        20 * capture_potential +
        15 * mobility_score +
        10 * distribution_score
    )
    return evaluation

@njit
def minimax_numba(board, scores, current_player, player_holes, depth, alpha, beta, maximizing_player):
    # Base case
    if depth == 0 or game_over_numba(board, scores):
        return fast_claude_evaluate_V1(board, scores, current_player, player_holes), -1

    moves = get_valid_moves_numba(board, current_player, player_holes)
    if moves.shape[0] == 0:
        return fast_claude_evaluate_V1(board, scores, current_player, player_holes), -1

    best_move = -1
    if maximizing_player:
        max_eval = -np.inf
        for i in range(moves.shape[0]):
            move = moves[i]
            new_board, new_scores = play_move_numba(board, scores, move, current_player, player_holes)
            eval_val, _ = minimax_numba(new_board, new_scores, 3 - current_player, player_holes, depth - 1, alpha, beta, False)
            if eval_val > max_eval:
                max_eval = eval_val
                best_move = move
            alpha = max(alpha, eval_val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = np.inf
        for i in range(moves.shape[0]):
            move = moves[i]
            new_board, new_scores = play_move_numba(board, scores, move, current_player, player_holes)
            eval_val, _ = minimax_numba(new_board, new_scores, 3 - current_player, player_holes, depth - 1, alpha, beta, True)
            if eval_val < min_eval:
                min_eval = eval_val
                best_move = move
            beta = min(beta, eval_val)
            if beta <= alpha:
                break
        return min_eval, best_move

@njit
def game_over_numba(board, scores):
    total_seeds = 0
    for h in range(16):
        total_seeds += board[h][0] + board[h][1]
    if total_seeds < 8:
        return True
    if scores[0] >= 33 or scores[1] >= 33:
        return True
    if scores[0] == 32 and scores[1] == 32:
        return True
    return False

@njit
def get_valid_moves_numba(board, current_player, player_holes):
    moves = []
    for h in player_holes[current_player]:
        for color in range(2):
            if board[h][color] > 0:
                moves.append((h, color))
    if len(moves) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return np.array(moves, dtype=np.int32)

@njit
def play_move_numba(board, scores, move, current_player, player_holes):
    hole, color = move
    seeds_to_sow = board[hole][color]
    board[hole][color] = 0
    current_index = hole

    while seeds_to_sow > 0:
        current_index = (current_index + 1) % 16
        if current_index == hole:
            continue  # Skip the starting hole
        if color == 0 and current_index in player_holes[current_player]:
            continue  # Red seeds skip own holes
        board[current_index][color] += 1
        seeds_to_sow -= 1

    # Apply capture
    scores = apply_capture_numba(board, scores, current_index, current_player)

    return board, scores

@njit
def apply_capture_numba(board, scores, start_hole, current_player):
    current_index = start_hole
    while True:
        total_seeds = board[current_index][0] + board[current_index][1]
        if total_seeds in [2, 3]:
            scores[current_player - 1] += total_seeds
            board[current_index][0] = 0
            board[current_index][1] = 0
            current_index = (current_index - 1) % 16
        else:
            break
    return scores
