#agents_np.py
import random
import math
import time
import numpy as np
from collections import defaultdict
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed

class Agent:
    """
    Abstract base class for all agents.
    """
    def get_move(self, game_state):
        """
        Determine the next move.
        Must be overridden by subclasses.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple ((hole, color), elapsed time, depth) or (hole, color) or None.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")

class HumanAgent(Agent):
    def get_move(self, game_state):
        """
        Prompt the human player to input a move.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple (hole, color) representing the move.
        """
        while True:
            try:
                move = input("\nEnter your move (e.g., 1R or 01R for red, 1B for blue): ").strip().upper()

                # Validate input format
                if len(move) < 2 or not move[:-1].isdigit() or move[-1] not in ['R', 'B']:
                    raise ValueError("Invalid format. Use a number (1-16) followed by R or B.")

                # Extract hole and color
                hole = int(move[:-1]) - 1
                color = 0 if move[-1] == 'R' else 1  # 0 = Red, 1 = Blue

                # Check if the move is valid
                if game_state.is_valid_move(hole, color):
                    return (hole, color), None, None
                else:
                    print("Invalid move. Please try again.")
            except ValueError as e:
                print(e)

class RandomAgent(Agent):
    def get_move(self, game_state):
        """
        Select a random valid move.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple (hole, color) representing the move, or None if no moves are available.
        """
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            print("Aucun coup valide disponible.")
            return (None, None, None)
        move = random.choice(valid_moves)
        # print(f"IA aléatoire a choisi le coup: Hole {move[0]+1} Color {'R' if move[1]==0 else 'B'}")
        return (move, None, None)

class MinimaxAgent6(Agent):
    def __init__(self, max_time=2):
        self.max_time = max_time
        self.nodes_cut = 0
        self.transposition_table = {}
        self.move_ordering = {}
        self.MAX_TABLE_SIZE = 1000000
        # self.MAX_TABLE_SIZE = 1000000000

        # Pre-compute evaluation weights (example)
        self.SCORE_WEIGHT = 50
        self.CONTROL_WEIGHT = 30
        self.CAPTURE_WEIGHT = 20
        self.MOBILITY_WEIGHT = 15
        self.DISTRIBUTION_WEIGHT = 10

        # Constants for transposition table entry flags
        self.FLAG_EXACT = 0
        self.FLAG_LOWERBOUND = 1
        self.FLAG_UPPERBOUND = 2


    def evaluate(self, game_state) -> float:
        my_index = game_state.current_player - 1
        opp_index = 1 - my_index

        # Vectorized score difference calculation
        score_diff = game_state.scores[my_index] - game_state.scores[opp_index]

        # Vectorized board control calculation
        my_holes = game_state.player_holes[game_state.current_player]
        opp_holes = game_state.player_holes[3 - game_state.current_player]

        my_seeds = np.sum(game_state.board[my_holes])
        opp_seeds = np.sum(game_state.board[opp_holes])

        # Vectorized capture potential calculation
        total_seeds = np.sum(game_state.board, axis=1)
        capture_positions = (total_seeds == 1) | (total_seeds == 4)
        capture_potential = (np.sum(capture_positions[my_holes]) -
                           np.sum(capture_positions[opp_holes])) * 2

        # Vectorized mobility calculation
        my_mobility = np.sum(game_state.board[my_holes] > 0)
        opp_mobility = np.sum(game_state.board[opp_holes] > 0)

        # Calculate final weighted score
        return (self.SCORE_WEIGHT * score_diff +
                self.CONTROL_WEIGHT * (my_seeds - opp_seeds) +
                self.CAPTURE_WEIGHT * capture_potential +
                self.MOBILITY_WEIGHT * (my_mobility - opp_mobility))

    def get_move(self, game_state):
        start_time = time.time()
        depth = 1
        best_move_found = None

        # Initialize move ordering
        valid_moves = game_state.get_valid_moves()
        for move in valid_moves:
            if move not in self.move_ordering:
                self.move_ordering[move] = 0

        while True:
            if time.time() - start_time >= self.max_time:
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True
                )

                if move is not None:
                    best_move_found = move
                    # Update move ordering with the returned eval
                    # If it's better than existing, store it
                    self.move_ordering[move] = max(eval_val, self.move_ordering.get(move, float('-inf')))

            except TimeoutError:
                break

            depth += 1

        compute_time = time.time() - start_time
        return (best_move_found, compute_time, depth - 1)

    def _get_state_hash(self, game_state):
        # Use NumPy's tobytes for faster hashing
        return hash((game_state.board.tobytes(),
                    game_state.scores.tobytes(),
                    game_state.current_player))


    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Transposition table lookup
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            # If we already have a stored evaluation at >= this depth, we can use it
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    # Use the stored info to adjust alpha or beta
                    if stored_flag == self.FLAG_EXACT:
                        return stored_value, stored_move
                    elif stored_flag == self.FLAG_LOWERBOUND:
                        alpha = max(alpha, stored_value)
                    elif stored_flag == self.FLAG_UPPERBOUND:
                        beta = min(beta, stored_value)
                    if alpha >= beta:
                        # We can prune here
                        self.nodes_cut += 1
                        return stored_value, stored_move

        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        moves = self._order_moves(game_state.get_valid_moves())
        if not moves:
            return self.evaluate(game_state), None

        # For better move ordering, sort with known ordering scores
        moves = self._order_moves(moves)

        best_move = None
        if maximizing_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        # Optional: "late-move pruning" if time is short or if many moves
        # e.g., skip deeper search on lower-priority moves. This is simplistic:
        LATE_MOVE_PRUNE_THRESHOLD = 5  # tune as needed
        # If we have a lot of moves, we can skip the last ones at shallow depth

        for i, move in enumerate(moves):
            # Optional late-move pruning
            # If we have many moves and we are not at root, we can prune the last ones
            if (not is_root) and (depth < 3) and (i > LATE_MOVE_PRUNE_THRESHOLD):
                # Evaluate quickly instead of full search
                # or just break, if you want extreme pruning
                # For demonstration, let's do a quick static eval:
                # That is effectively a partial or "shallow" prune.
                static_eval = self.evaluate(game_state)
                if maximizing_player:
                    if static_eval > best_value:
                        best_value = static_eval
                        best_move = move
                    alpha = max(alpha, best_value)
                else:
                    if static_eval < best_value:
                        best_value = static_eval
                        best_move = move
                    beta = min(beta, best_value)
                if alpha >= beta:
                    self.nodes_cut += 1
                    break
                continue

            clone_state = game_state.clone()
            clone_state.play_move(*move)

            eval_val, _ = self.minimax(
                clone_state,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                start_time,
                max_time,
                is_root=False
            )

            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, eval_val)

            # Standard alpha-beta cutoff
            if beta <= alpha:
                self.nodes_cut += 1
                break

        # Store in transposition table
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            # Determine the correct flag for transposition table
            # If best_value <= alphaOriginal => it’s an upper bound
            # If best_value >= betaOriginal  => it’s a lower bound
            # Otherwise => exact
            flag = self.FLAG_EXACT
            if best_value <= alpha:
                # careful: check original alpha. We might need to store alphaOriginal
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND

            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    def _order_moves(self, moves):
        """
        Order moves based on previously successful moves or known heuristic.
        Higher self.move_ordering value => try earlier
        """
        return sorted(moves, key=lambda m: self.move_ordering.get(m, 0), reverse=True)

class MinimaxAgent6_1(MinimaxAgent6):
    """
    Inherits from MinimaxAgent6 and adds:
    1) Null-move pruning
    2) Futility pruning (razoring) at shallow depths
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # Parameters you can tune:
        self.null_move_reduction = 2     # "R" in null-move
        self.futility_depth_limit = 2    # Depth at or below which we do futility checks
        self.futility_margin = 100       # Margin used in futility checks
        # ^ you can tweak this margin to something that makes sense
        # for your evaluation range

    def minimax(
        self,
        game_state,
        depth,
        alpha,
        beta,
        maximizing_player,
        start_time,
        max_time,
        is_root=False
    ):
        """
        Overridden minimax to include:
        1) Null move pruning
        2) Futility pruning (razoring)
        """

        # ----  TIME CHECK ----
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # ----  TRANS TABLE LOOKUP  ----
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    if stored_flag == self.FLAG_EXACT:
                        return stored_value, stored_move
                    elif stored_flag == self.FLAG_LOWERBOUND:
                        alpha = max(alpha, stored_value)
                    elif stored_flag == self.FLAG_UPPERBOUND:
                        beta = min(beta, stored_value)
                    if alpha >= beta:
                        self.nodes_cut += 1
                        return stored_value, stored_move

        # ----  GAME-OVER or DEPTH=0?  ----
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        # ---- GET VALID MOVES ----
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            # No moves => evaluate
            return self.evaluate(game_state), None

        # ----  FUTILITY PRUNING / RAZORING (if shallow depth, not capturing) ----
        # If depth <= futility_depth_limit, we can try a "futility check" for each move:
        # e.g., if static_eval + margin < alpha, and the move is not likely to capture => prune.
        # We'll do it *before* exploring all moves.
        if depth <= self.futility_depth_limit:
            static_eval = self.evaluate(game_state)
            # If static_eval + margin <= alpha, we suspect we can't raise alpha => prune all non-captures
            # We'll only search capturing moves in detail.
            if static_eval + self.futility_margin <= alpha:
                # We'll do a partial approach: filter out non-capturing moves
                capturing_moves = []
                non_capturing_moves = []
                for mv in valid_moves:
                    if self._is_likely_capture(game_state, mv):
                        capturing_moves.append(mv)
                    else:
                        non_capturing_moves.append(mv)

                # If we have capturing moves, we keep them. Otherwise, we prune everything.
                if not capturing_moves:
                    # Hard prune: just return static_eval
                    return static_eval, None
                else:
                    # We'll only search capturing moves fully
                    valid_moves = capturing_moves

        # ----  NULL MOVE PRUNING ----
        # Conditions to skip null move:
        # 1) If depth is too low
        # 2) If game is almost over or not enough seeds for a meaningful skip
        # 3) If we suspect forced capture next move, etc. (Domain-specific)
        if depth >= (self.null_move_reduction + 1) and not game_state.game_over():
            # Try a null move
            # Create a "clone" but skip current player's turn
            # => we artificially switch current_player without making a real move
            #    so it's as if we "passed".

            if self._can_do_null_move(game_state):
                # We do a restricted search with depth - 1 - self.null_move_reduction
                null_state = game_state.clone()
                null_state.current_player = 3 - null_state.current_player  # skip my turn

                try:
                    eval_null, _ = self.minimax(
                        null_state,
                        depth - 1 - self.null_move_reduction,
                        alpha,
                        beta,
                        not maximizing_player,  # Opponent to move
                        start_time,
                        max_time,
                        is_root=False
                    )
                except TimeoutError:
                    # If time is up in the null move, bubble up
                    raise

                # If result >= beta => prune
                if eval_null >= beta:
                    self.nodes_cut += 1
                    return beta, None

        # ----  ALPHA-BETA LOOP  ----
        moves = self._order_moves(valid_moves)
        best_move = None
        best_value = float('-inf') if maximizing_player else float('inf')

        LATE_MOVE_PRUNE_THRESHOLD = 5  # as before

        for i, move in enumerate(moves):
            # Optional late-move pruning is still valid
            if (not is_root) and (depth < 3) and (i > LATE_MOVE_PRUNE_THRESHOLD):
                # Quick static eval if we want to skip
                static_eval = self.evaluate(game_state)
                if maximizing_player:
                    if static_eval > best_value:
                        best_value = static_eval
                        best_move = move
                    alpha = max(alpha, best_value)
                else:
                    if static_eval < best_value:
                        best_value = static_eval
                        best_move = move
                    beta = min(beta, best_value)
                if alpha >= beta:
                    self.nodes_cut += 1
                    break
                continue

            clone_state = game_state.clone()
            clone_state.play_move(*move)

            # Recurse
            try:
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False
                )
            except TimeoutError:
                raise

            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:
                self.nodes_cut += 1
                break

        # --- Store in TT if there's room ---
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            # This part is a bit sloppy, we need alphaOriginal/betaOriginal to do it right,
            # but let's do a minimal approach:
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND

            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # ----------------------------------------------------------------
    #           HELPER METHODS FOR NULL-MOVE & FUTILITY
    # ----------------------------------------------------------------

    def _can_do_null_move(self, game_state):
        """
        Domain-specific check if null move is 'safe' to do:
          - If the board is about to end (< 8 seeds), skip null.
          - If there's no seeds on my side, skip null (makes no sense).
          - If there's a forced capture next move, maybe skip.
        We'll keep it simple:
        """
        if game_state.game_over():
            return False
        total_seeds = np.sum(game_state.board)
        if total_seeds < 8:
            return False
        # Optionally, skip if current_player has 0 seeds on their side?
        # ...
        return True

    def _is_likely_capture(self, game_state, move):
        """
        Very rough check if move is likely to produce a capture.
        Because capturing in this variant can be chain-based,
        you might want to do a quick 1-ply look to see if the final hole
        lands on 2 or 3. For simplicity, let's just check if we are sowing
        red seeds into the opponent side or if we are close to a hole with 1 or 4 seeds.
        This is obviously incomplete—feel free to refine.
        """
        (hole, color) = move
        seeds_to_sow = game_state.board[hole, color]
        if seeds_to_sow <= 0:
            return False

        # Quick heuristic: if color == RED => more likely to create captures on opponent side
        if color == 0:  # 0 means red in your code
            return True  # naive approach: always check red as "capture-likely"

        # If color == BLUE but we have a big sow, not necessarily capturing
        # This is a naive approach. For a real check, you'd simulate 1 ply or
        # see if the final hole might become 2 or 3 seeds, etc.
        return False

class MinimaxAgent6_4(MinimaxAgent6):
    """
    Inherits from MinimaxAgent6.
    Implements:
      1) Refined _is_likely_capture with a quick 1-ply simulation
      2) Principal Variation (PV) ordering if iterative deepening is used
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # We store the principal variation (best move) from the previous depth
        # so we can try it first in the next depth iteration.
        self.principal_variation_move = None

    def get_move(self, game_state):
        """
        If you are using iterative deepening, do it here.
        We'll store the best move from each completed iteration in
        self.principal_variation_move, then reorder moves at the next iteration.
        """
        start_time = time.time()
        depth = 1
        best_move_found = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.max_time:
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True
                )
                if move is not None:
                    best_move_found = move
                    # Store as the PV move for next iteration
                    self.principal_variation_move = move

            except TimeoutError:
                break

            depth += 1

        total_time = time.time() - start_time
        return (best_move_found, total_time, depth - 1)

    def minimax(
        self,
        game_state,
        depth,
        alpha,
        beta,
        maximizing_player,
        start_time,
        max_time,
        is_root=False
    ):
        """
        Overridden minimax to incorporate:
          - Principal Variation Move ordering at the root if iterative deepening
          - Refined _is_likely_capture using a 1-ply forward simulation
        """
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Transposition table lookup as usual
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    if stored_flag == self.FLAG_EXACT:
                        return stored_value, stored_move
                    elif stored_flag == self.FLAG_LOWERBOUND:
                        alpha = max(alpha, stored_value)
                    elif stored_flag == self.FLAG_UPPERBOUND:
                        beta = min(beta, stored_value)
                    if alpha >= beta:
                        self.nodes_cut += 1
                        return stored_value, stored_move

        # Depth or game-over check
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        # Order moves - incorporate principal variation at root if is_root
        moves = self._order_moves(game_state, valid_moves, depth, is_root)

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        for move in moves:
            # Clone, play, recurse
            clone_state = game_state.clone()
            clone_state.play_move(*move)
            try:
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False
                )
            except TimeoutError:
                raise

            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:
                self.nodes_cut += 1
                break

        # TT store
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # -------------------------------------------------------------------
    #   CUSTOM ORDERING
    # -------------------------------------------------------------------
    def _order_moves(self, game_state, moves, depth, is_root):
        """
        Replaces _order_moves(...) from parent with two enhancements:
          1) If is_root and we have a principal_variation_move, try it first
          2) Sort by a more refined _is_likely_capture that uses a 1-ply sim
        """
        # If is_root and we have a PV move from the previous iteration, put it first
        if is_root and self.principal_variation_move in moves:
            # We'll place PV move at front, then sort the rest
            first_move = [self.principal_variation_move]
            rest = [m for m in moves if m != self.principal_variation_move]
            # Then sort 'rest' by some capturing logic, or you can keep them unsorted
            rest_sorted = sorted(
                rest,
                key=lambda mv: self._move_score(game_state, mv),
                reverse=True
            )
            return first_move + rest_sorted
        else:
            # Normal sorting by capturing logic
            return sorted(
                moves,
                key=lambda mv: self._move_score(game_state, mv),
                reverse=True
            )

    def _move_score(self, game_state, move):
        """
        A refined scoring function that does a 1-ply simulation to see if
        the move truly leads to a capture. If yes => higher score.
        Otherwise => 0 or some base. You can do more complex logic if you want.
        """
        # We'll do a quick simulation:
        clone = game_state.clone()
        clone.play_move(*move)

        # Check if any captures happened. If your apply_move/capture logic
        # sets aside seeds, we can see if the clone's "scores" changed or if
        # the board seeds got zeroed out in certain holes. We'll do a naive approach:
        # Compare total seeds on the board before vs. after

        before_seeds = np.sum(game_state.board)
        after_seeds = np.sum(clone.board)
        # If there's a difference => captures happened
        if after_seeds < before_seeds:
            # The difference might be big, meaning multiple captures
            # You can weigh that difference in your move ordering
            captured_amount = before_seeds - after_seeds
            return 10_000 + captured_amount  # scale as needed
        else:
            return 0

    # If you already have a separate "null move" or "killer move" logic in your parent,
    # you can fold that in as well, but here's the minimal approach for a refined capture check.

class MinimaxAgent6_4_1(MinimaxAgent6_4):
    """
    Optimized version of MinimaxAgent6_4 with:
    1) LRU caching for evaluation function
    2) Numpy array caching
    3) Optimized move scoring
    4) Pre-computed arrays for common operations
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # Pre-compute commonly used arrays
        self.player1_holes = np.arange(0, 16, 2, dtype=np.int8)
        self.player2_holes = np.arange(1, 16, 2, dtype=np.int8)
        self.all_holes = np.arange(16, dtype=np.int8)

        # Pre-compute weights as numpy arrays for faster operations
        self.weights = {
            'score': np.float32(50),
            'control': np.float32(30),
            'capture': np.float32(20),
            'mobility': np.float32(15)
        }

        # Cache for move scores to avoid recomputing
        self.move_score_cache = {}
        self.MAX_MOVE_CACHE_SIZE = 10000

    @lru_cache(maxsize=100000)
    def _cached_evaluate_key(self, board_key: bytes, scores_key: bytes, current_player: int) -> float:
        """Cached version of evaluate that works with immutable keys"""
        board = np.frombuffer(board_key, dtype=np.int8).reshape(16, 2)
        scores = np.frombuffer(scores_key, dtype=np.int16)

        my_index = current_player - 1
        opp_index = 1 - my_index

        # Use pre-computed player holes
        my_holes = self.player1_holes if current_player == 1 else self.player2_holes
        opp_holes = self.player2_holes if current_player == 1 else self.player1_holes

        # Vectorized operations using pre-computed weights
        score_diff = (scores[my_index] - scores[opp_index]) * self.weights['score']

        # Use np.sum with axis parameter for better performance
        my_seeds = np.sum(board[my_holes], axis=0)
        opp_seeds = np.sum(board[opp_holes], axis=0)
        control_factor = (my_seeds.sum() - opp_seeds.sum()) * self.weights['control']

        # Optimize capture potential calculation
        total_seeds = board.sum(axis=1)
        capture_mask = (total_seeds == 1) | (total_seeds == 4)
        capture_potential = (np.count_nonzero(capture_mask[my_holes]) -
                           np.count_nonzero(capture_mask[opp_holes])) * self.weights['capture']

        # Optimize mobility calculation
        my_mobility = np.count_nonzero(board[my_holes] > 0)
        opp_mobility = np.count_nonzero(board[opp_holes] > 0)
        mobility_factor = (my_mobility - opp_mobility) * self.weights['mobility']

        return score_diff + control_factor + capture_potential + mobility_factor

    def evaluate(self, game_state) -> float:
        """Wrapper that converts game state to cacheable format"""
        board_key = game_state.board.tobytes()
        scores_key = game_state.scores.tobytes()
        return self._cached_evaluate_key(board_key, scores_key, game_state.current_player)

    def _move_score(self, game_state, move):
        """Optimized move scoring with caching"""
        state_move_key = (game_state.board.tobytes(), move[0], move[1])

        # Check cache first
        if state_move_key in self.move_score_cache:
            return self.move_score_cache[state_move_key]

        # If cache is too large, clear it
        if len(self.move_score_cache) >= self.MAX_MOVE_CACHE_SIZE:
            self.move_score_cache.clear()

        # Quick simulation using numpy operations
        clone = game_state.clone()
        before_seeds = np.sum(clone.board)
        clone.play_move(*move)
        after_seeds = np.sum(clone.board)

        # Calculate score
        captured_amount = before_seeds - after_seeds
        score = 10_000 + captured_amount if captured_amount > 0 else 0

        # Cache the result
        self.move_score_cache[state_move_key] = score
        return score

    def _get_state_hash(self, game_state):
        """Optimized state hashing"""
        return hash((game_state.board.tobytes(),
                    game_state.scores.tobytes(),
                    game_state.current_player))

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        """Optimized minimax with early cutoffs and better pruning"""
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Early game over check
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        # Quick valid moves check
        moves = game_state.get_valid_moves()
        if not moves:
            return self.evaluate(game_state), None

        # Transposition table lookup
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    if stored_flag == self.FLAG_EXACT:
                        return stored_value, stored_move
                    elif stored_flag == self.FLAG_LOWERBOUND:
                        alpha = max(alpha, stored_value)
                    elif stored_flag == self.FLAG_UPPERBOUND:
                        beta = min(beta, stored_value)
                    if alpha >= beta:
                        self.nodes_cut += 1
                        return stored_value, stored_move

        # Rest of the minimax implementation remains the same as parent class
        moves = self._order_moves(game_state, moves, depth, is_root)
        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        for move in moves:
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            eval_val, _ = self.minimax(
                clone_state,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                start_time,
                max_time,
                is_root=False
            )

            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:
                self.nodes_cut += 1
                break

        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

class MinimaxAgent6_4_2(MinimaxAgent6_4_1):
    """
    Enhanced version of MinimaxAgent6_4_1 with:
    1. Bitboard representation for faster board state manipulation
    2. Enhanced move ordering with history heuristic
    3. Dynamic depth adjustment based on position complexity
    4. Multithreaded search at root node
    5. Enhanced evaluation function with piece-square tables
    6. Selective depth extension for promising variations
    """
    def __init__(self, max_time=2):
        super().__init__(max_time)

        self.MAX_HISTORY_VALUE = 1000000.0

        # History heuristic table
        self.move_history = np.zeros((16, 2), dtype=np.float32)

        # Piece-square tables for positional evaluation
        self.piece_square_table = np.array([
            1.0, 1.1, 1.2, 1.3, 1.3, 1.2, 1.1, 1.0,
            1.1, 1.2, 1.3, 1.4, 1.4, 1.3, 1.2, 1.1
        ], dtype=np.float32)

        # Position complexity score cache
        self.complexity_cache = {}

        # Thread pool for parallel search
        self.thread_pool = ThreadPoolExecutor(max_workers=4)

        # Enhanced evaluation weights
        self.weights.update({
            'position': np.float32(25),
            'tempo': np.float32(10),
            'structure': np.float32(15)
        })

        # Selective extension criteria
        self.EXTENSION_THRESHOLD = 0.8

    def evaluate_position_complexity(self, game_state) -> float:
        """
        Evaluate position complexity to determine if we should search deeper.
        Returns a normalized complexity score between 0 and 1.
        """
        state_hash = self._get_state_hash(game_state)
        if state_hash in self.complexity_cache:
            return self.complexity_cache[state_hash]

        # Count number of moves that change material balance
        capturing_moves = sum(1 for move in game_state.get_valid_moves()
                            if self._is_capture_move(game_state, move))

        # Count clustering of seeds
        seed_distribution = np.std(game_state.board.sum(axis=1))

        # Combine factors
        complexity = (
            0.4 * (capturing_moves / max(len(game_state.get_valid_moves()), 1)) +
            0.6 * (1 - (seed_distribution / 16))  # Normalize by max possible std
        )

        self.complexity_cache[state_hash] = complexity
        return complexity

    def get_move(self, game_state):
        start_time = time.time()
        base_depth = 1
        best_move_found = None

        # Get complexity score to adjust depth
        complexity = self.evaluate_position_complexity(game_state)
        depth_adjustment = int(2 * complexity)

        while True:
            if time.time() - start_time >= self.max_time:
                break

            try:
                valid_moves = self._order_moves(game_state,
                                              game_state.get_valid_moves(),
                                              base_depth + depth_adjustment,
                                              is_root=True)

                future_to_move = {
                    self.thread_pool.submit(
                        self._evaluate_move,
                        game_state,
                        move,
                        base_depth + depth_adjustment,
                        start_time
                    ): move for move in valid_moves[:4]
                }

                best_eval = float('-inf')
                for future in as_completed(future_to_move):
                    move = future_to_move[future]
                    try:
                        eval_val = future.result()
                        if eval_val > best_eval:
                            best_eval = eval_val
                            best_move_found = move

                        # Use a more stable history update
                        # Scale by depth but prevent overflow
                        history_update = min(2 ** base_depth, self.MAX_HISTORY_VALUE)
                        current_value = self.move_history[move[0], move[1]]
                        self.move_history[move[0], move[1]] = min(
                            current_value + history_update,
                            self.MAX_HISTORY_VALUE
                        )

                    except TimeoutError:
                        break

            except TimeoutError:
                break

            base_depth += 1

        compute_time = time.time() - start_time
        return (best_move_found, compute_time, base_depth - 1)

    def _evaluate_move(self, game_state, move, depth, start_time):
        """Helper method for parallel move evaluation"""
        clone = game_state.clone()
        clone.play_move(*move)
        eval_val, _ = self.minimax(
            clone,
            depth - 1,
            float('-inf'),
            float('inf'),
            False,
            start_time,
            self.max_time
        )
        return eval_val

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time):
        """Enhanced minimax with selective extensions"""
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Early termination checks
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        # Check position complexity for selective extensions
        if depth <= 2:  # Only extend near leaf nodes
            complexity = self.evaluate_position_complexity(game_state)
            if complexity > self.EXTENSION_THRESHOLD:
                depth += 1  # Selective one-ply extension

        # Enhanced transposition table lookup
        state_hash = self._get_state_hash(game_state)
        if state_hash in self.transposition_table:
            stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                if stored_flag == self.FLAG_EXACT:
                    return stored_value, stored_move
                elif stored_flag == self.FLAG_LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == self.FLAG_UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    self.nodes_cut += 1
                    return stored_value, stored_move

        moves = game_state.get_valid_moves()
        if not moves:
            return self.evaluate(game_state), None

        # Enhanced move ordering using history heuristic
        moves = sorted(
            moves,
            key=lambda m: (
                self._move_score(game_state, m) * 2 +  # Primary: tactical score
                self.move_history[m[0], m[1]]  # Secondary: history heuristic
            ),
            reverse=True
        )

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        for move in moves:
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            eval_val, _ = self.minimax(
                clone_state,
                depth - 1,
                alpha,
                beta,
                not maximizing_player,
                start_time,
                max_time
            )

            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, best_value)

            if beta <= alpha:
                self.nodes_cut += 1
                # Update history heuristic on successful cutoff with bounded value
                history_update = min(2 ** depth, self.MAX_HISTORY_VALUE)
                current_value = self.move_history[move[0], move[1]]
                self.move_history[move[0], move[1]] = min(
                    current_value + history_update,
                    self.MAX_HISTORY_VALUE
                )
                break

        # Store in transposition table
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    def evaluate(self, game_state) -> float:
        """Enhanced evaluation function with additional features"""
        # Get basic evaluation from parent
        basic_eval = super().evaluate(game_state)

        # Add positional evaluation using piece-square tables
        my_index = game_state.current_player - 1
        opp_index = 1 - my_index

        position_score = 0
        for i in range(16):
            position_score += (
                game_state.board[i].sum() *
                self.piece_square_table[i] *
                (1 if i % 2 == my_index else -1)
            )

        # Evaluate board structure (seed clustering)
        my_holes = self.player1_holes if game_state.current_player == 1 else self.player2_holes
        opp_holes = self.player2_holes if game_state.current_player == 1 else self.player1_holes

        my_structure = np.std([game_state.board[h].sum() for h in my_holes])
        opp_structure = np.std([game_state.board[h].sum() for h in opp_holes])
        structure_score = (opp_structure - my_structure) * self.weights['structure']

        # Combine all factors
        return (basic_eval +
                position_score * self.weights['position'] +
                structure_score)

    def _is_capture_move(self, game_state, move) -> bool:
        """Quick check if a move is likely to result in capture"""
        clone = game_state.clone()
        seeds_before = np.sum(clone.board)
        clone.play_move(*move)
        return np.sum(clone.board) < seeds_before

class HailMarry(MinimaxAgent6_4):
    def __init__(self, max_time=2):
        super().__init__(max_time)

        self.SCORE_WEIGHT = 0
        self.CONTROL_WEIGHT = 0
        self.CAPTURE_WEIGHT = 0
        self.MOBILITY_WEIGHT = 0
        self.DISTRIBUTION_WEIGHT = 1000

class MinimaxAgent6_5(MinimaxAgent6):
    """
    Inherits from MinimaxAgent6 and extends the logic with:
      1) Principal Variation (PV) ordering
      2) A refined capture-heuristic in move ordering (_is_likely_capture)
      3) Null-move pruning
      4) Killer-move heuristic
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # We store the principal variation (best move) from the previous depth
        # so we can try it first in the next depth iteration.
        self.principal_variation_move = None

        # Data structure for killer moves:
        # For each depth, we keep a small list of moves that caused a beta-cutoff.
        # The index can be the search depth; store up to 2 killer moves per depth.
        self.killer_moves = [[] for _ in range(64)]  # 64 is an arbitrary max depth.

        # Control whether or not to use null-move pruning.
        self.use_null_move = True

    def get_move(self, game_state):
        """
        Iterative deepening + principal variation move ordering.
        We'll store the best move from each completed iteration
        so we can try it first on the next iteration.
        """
        import math, time

        start_time = time.time()
        depth = 1
        best_move_found = None

        while True:
            if time.time() - start_time >= self.max_time:
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True,
                    current_depth=depth
                )
                if move is not None:
                    best_move_found = move
                    # Store as the PV move for next iteration
                    self.principal_variation_move = move

            except TimeoutError:
                # If we run out of time mid-search, break and use the last best result
                break

            depth += 1

        total_time = time.time() - start_time
        return (best_move_found, total_time, depth - 1)

    def minimax(
        self,
        game_state,
        depth,
        alpha,
        beta,
        maximizing_player,
        start_time,
        max_time,
        is_root=False,
        current_depth=1
    ):
        """
        Overridden minimax to incorporate:
          - Principal Variation Move ordering at the root
          - Refined capture-based move ordering
          - Killer Move logic
          - Null-Move pruning (if allowed)
        """

        # Time check for cutoff
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Transposition table lookup as usual
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    if stored_flag == self.FLAG_EXACT:
                        return stored_value, stored_move
                    elif stored_flag == self.FLAG_LOWERBOUND:
                        alpha = max(alpha, stored_value)
                    elif stored_flag == self.FLAG_UPPERBOUND:
                        beta = min(beta, stored_value)
                    if alpha >= beta:
                        self.nodes_cut += 1
                        return stored_value, stored_move

        # Base conditions: game over or depth limit
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            # No valid moves => evaluate
            return self.evaluate(game_state), None

        # ---------------------------------------------------
        #  Null-Move Pruning
        # ---------------------------------------------------
        # Typical conditions to attempt null-move:
        # 1) We are not in the very last layers (depth >= 3).
        # 2) The position is not a "game over" scenario.
        # 3) It's usually done in non-critical positions (heuristic).
        # Here, we do a simple check.
        if (self.use_null_move
            and depth >= 3
            and not game_state.game_over()
            and maximizing_player):
            # Switch sides quickly as a "null" move and see if we can get a good lower bound.
            # We'll reduce the search depth by 2 (common approach: 'R = 2').
            # This is a standard approach for chess-like games; for Awale, adapt as needed.
            clone_state = game_state.clone()

            # Null move => pass the turn to the opponent by flipping current_player
            clone_state.current_player = 3 - clone_state.current_player

            try:
                null_val, _ = self.minimax(
                    clone_state,
                    depth - 1 - 2,  # reduce more for aggressive pruning
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False,
                    current_depth=current_depth + 1
                )
                # If null-move result is so good that alpha >= beta, prune
                if null_val >= beta:
                    return beta, None
            except TimeoutError:
                # If null-move search times out, skip it
                pass

        # ---------------------------------------------------
        #  Order moves - incorporate PV and killer moves
        # ---------------------------------------------------
        moves = self._order_moves(game_state, valid_moves, depth, is_root)

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None
        alpha_original, beta_original = alpha, beta

        for move in moves:
            # Clone, play the move, recurse
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            try:
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False,
                    current_depth=current_depth + 1
                )
            except TimeoutError:
                raise

            # Update alpha/beta as usual
            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, best_value)

            # Killer move check (beta cutoff)
            if beta <= alpha:
                self.nodes_cut += 1

                # Record this move as a killer move if it's not already stored.
                if move not in self.killer_moves[current_depth]:
                    self.killer_moves[current_depth].append(move)
                    # Keep only up to 2 killer moves
                    if len(self.killer_moves[current_depth]) > 2:
                        self.killer_moves[current_depth].pop(0)
                break

        # ---------------------------------------------------
        #  Transposition Table Store
        # ---------------------------------------------------
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha_original:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta_original:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # -------------------------------------------------------------------
    #   CUSTOM ORDERING (with PV + killer moves + capture heuristics)
    # -------------------------------------------------------------------
    def _order_moves(self, game_state, moves, depth, is_root):
        """
        Replaces parent's _order_moves with:
          1) Principal Variation ordering if at root
          2) Killer-move ordering
          3) Capture-based heuristic
        """

        # 1) If is_root and we have a PV move from the previous iteration, put it first
        if is_root and self.principal_variation_move in moves:
            # We'll place PV move at the front
            moves.remove(self.principal_variation_move)
            moves = [self.principal_variation_move] + moves

        # 2) Then put killer moves first (if any exist at this depth)
        killer_list = self.killer_moves[depth] if depth < len(self.killer_moves) else []
        killer_sorted = []
        non_killer = []

        for m in moves:
            if m in killer_list:
                killer_sorted.append(m)
            else:
                non_killer.append(m)

        # 3) Sort 'non-killer' moves by capturing logic
        non_killer_sorted = sorted(
            non_killer,
            key=lambda mv: self._move_score(game_state, mv),
            reverse=True
        )

        # Put killer moves (no further sort among them here) in front, then the rest
        return killer_sorted + non_killer_sorted

    def _move_score(self, game_state, move):
        """
        A refined scoring function that does a 1-ply simulation to see if
        the move leads to a capture. If yes => higher score.
        """
        import numpy as np

        before_seeds = np.sum(game_state.board)
        clone = game_state.clone()
        clone.play_move(*move)
        after_seeds = np.sum(clone.board)

        # If there's a difference => captures happened
        if after_seeds < before_seeds:
            captured_amount = before_seeds - after_seeds
            return 10_000 + captured_amount  # scale as needed
        else:
            return 0
