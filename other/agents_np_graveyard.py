#agents_np.py
import random
import math
import time
import numpy as np

class MinimaxAgent1(Agent):
    def __init__(self, max_time=2):
        """
        Initialize the Minimax agent.

        Parameters:
            max_time (float): Maximum time allowed for move computation in seconds.
        """
        self.max_time = max_time

    def get_move(self, game_state):
        """
        Determine the best move using the Minimax algorithm with alpha-beta pruning.

        Parameters:
            game_state (AwaleGame): The current state of the game.

        Returns:
            tuple: A tuple (hole, color) representing the best move found.
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
                    -math.inf,
                    math.inf,
                    True,  # Maximizing player
                    start_time,
                    self.max_time
                )
                if move is not None:
                    best_move_found = move
            except Exception as e:
                print(f"Exception during minimax at depth {depth}: {e}")
                break

            depth += 1

        total_time = time.time() - start_time
        # print(f"Temps de calcul (Minimax) : {total_time:.2f}s, profondeur atteinte : {depth - 1}")
        return best_move_found, total_time, depth - 1

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time):
        """
        Recursive Minimax function with alpha-beta pruning.

        Parameters:
            game_state (AwaleGame): The current state of the game.
            depth (int): Current depth in the game tree.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximizing_player (bool): True if the current layer is maximizing, False otherwise.
            start_time (float): The start time of the computation.
            max_time (float): Maximum allowed computation time.

        Returns:
            tuple: (evaluation value, best move)
        """
        # Time check
        if time.time() - start_time >= max_time:
            return game_state.GPT_evaluate_V2(), None

        # Terminal condition
        if game_state.game_over() or depth == 0:
            return game_state.GPT_evaluate_V2(), None

        moves = game_state.get_valid_moves()
        if not moves:
            return game_state.GPT_evaluate_V2(), None

        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                # Time check within loop
                if time.time() - start_time >= max_time:
                    break

                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    False,  # Switch to minimizing
                    start_time,
                    max_time
                )
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break  # Beta cut-off
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                if time.time() - start_time >= max_time:
                    break

                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(
                    clone_state,
                    depth - 1,
                    alpha,
                    beta,
                    True,  # Switch to maximizing
                    start_time,
                    max_time
                )
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break  # Alpha cut-off
            return min_eval, best_move

class MinimaxAgent6_S(MinimaxAgent6):
    def __init__(self, max_time=2):
        super().__init__(max_time)
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics counters"""
        self.nodes_cut = 0
        self.total_nodes_visited = 0

    def get_move(self, game_state):
        # Reset stats at the start of each move
        self.reset_stats()

        # Get the move using parent class implementation
        move_result = super().get_move(game_state)

        # Print statistics after move computation
        pruning_percentage = (self.nodes_cut / self.total_nodes_visited * 100) if self.total_nodes_visited > 0 else 0
        print(f"\nPruning Statistics:")
        print(f"Total nodes visited: {self.total_nodes_visited}")
        print(f"Nodes cut by pruning: {self.nodes_cut}")
        print(f"Pruning percentage: {pruning_percentage:.2f}%")

        return move_result

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        # Increment total nodes counter
        self.total_nodes_visited += 1

        # Use parent class implementation
        return super().minimax(game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root)

class MinimaxAgent6_2(MinimaxAgent6):
    """
    Inherits from MinimaxAgent6 and adds:
      1) Killer moves and history heuristics for better move ordering.
      2) Late Move Reductions (LMR) rather than skipping late moves.
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)

        # We'll store killer moves in a small array/dict keyed by depth
        # e.g. killer_moves[depth] = [None, None] to store up to 2 killer moves
        self.killer_moves = {}

        # We'll store history scores in a dict keyed by (move, depth)
        # or simply keyed by move across all depths. For simplicity, let's do keyed by move only:
        self.history_scores = {}

        # For LMR: define a threshold for which moves get reduced
        self.lmr_move_threshold = 3  # If the move index i >= 3 => consider reducing
        # Typically set with some small integer in chess engines.

        # For LMR: define the reduction amount. Usually 1 ply.
        self.lmr_reduction = 1

    def get_move(self, game_state):
        """
        Same as parent, but we might also reset or adjust some killer/history data
        if we want it per-turn. Typically killer moves are stored per search, so you
        could clear them here if desired.
        """
        # Optionally clear killer_moves each new move:
        self.killer_moves.clear()
        # Or keep them. Some engines preserve them across iterative deepening.

        return super().get_move(game_state)

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
        Overridden to incorporate:
          1) Killer move / history-based ordering
          2) Late Move Reductions
        """
        # ---- TIME CHECK ----
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # ---- TRANS TABLE LOOKUP (same as parent, partial copy/paste) ----
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

        # ---- DEPTH/GAME OVER CHECK ----
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        # ---- GET VALID MOVES ----
        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        # ---- ORDER MOVES (Killer + History) ----
        # We'll apply a new method _order_moves_killer_history()
        moves = self._order_moves_killer_history(valid_moves, depth)

        best_move = None
        if maximizing_player:
            best_value = float('-inf')
        else:
            best_value = float('inf')

        # We'll still keep your original LATE_MOVE_PRUNE_THRESHOLD,
        # but we'll do LMR instead of skipping.
        LATE_MOVE_PRUNE_THRESHOLD = 5

        # We'll track the move index, so we can do LMR on "late" moves
        for i, move in enumerate(moves):
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            do_lmr = False
            new_depth = depth - 1

            # LMR Condition (example):
            # 1) depth >= 3 (some safe minimum)
            # 2) This is not the first few moves
            # 3) Move is likely not a capture or "killer"
            # 4) Not root node
            if (
                not is_root
                and depth >= 3
                and i >= self.lmr_move_threshold
                and (not self._is_killer_move(move, depth))
                and (not self._is_likely_capture(game_state, move))
            ):
                # Apply a depth reduction by self.lmr_reduction
                do_lmr = True
                reduced_depth = new_depth - self.lmr_reduction
                if reduced_depth < 1:
                    reduced_depth = 1  # clamp

                # We'll do a reduced-depth search first:
                eval_val, _ = self.minimax(
                    clone_state,
                    reduced_depth,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False
                )
                # If it "fails high" (for maximizing) or "fails low" (for minimizing),
                # we do a full re-search at the normal depth.
                if maximizing_player:
                    if eval_val > best_value:
                        # We update best_value, but let's check if we need a re-search:
                        if eval_val > alpha:
                            # do re-search at original depth
                            eval_val, _ = self.minimax(
                                clone_state,
                                new_depth,
                                alpha,
                                beta,
                                not maximizing_player,
                                start_time,
                                max_time,
                                is_root=False
                            )
                else:
                    # Minimizing
                    if eval_val < best_value:
                        if eval_val < beta:
                            eval_val, _ = self.minimax(
                                clone_state,
                                new_depth,
                                alpha,
                                beta,
                                not maximizing_player,
                                start_time,
                                max_time,
                                is_root=False
                            )
            else:
                # Normal search
                eval_val, _ = self.minimax(
                    clone_state,
                    new_depth,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False
                )

            # ---- Evaluate alpha/beta as usual ----
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

            # ---- Beta cutoff? ----
            if beta <= alpha:
                self.nodes_cut += 1

                # Mark the current move as a Killer move:
                self._add_killer_move(move, depth)

                # Also update history heuristic:
                self._update_history(move, depth)

                break

            # If no cutoff, still update history if it improves alpha:
            # For maximizing: if eval_val > alpha, we improved alpha
            # For minimizing: if eval_val < beta, we improved alpha in a sense
            # We'll do it in a simpler form:
            if maximizing_player and (eval_val > alpha):
                # alpha improved => update history
                self._update_history(move, depth)
            elif (not maximizing_player) and (eval_val < beta):
                self._update_history(move, depth)

        # ---- Store in TT if there's room ----
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            # We don't have alphaOriginal/betaOriginal, so do a guess:
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND

            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # -----------------------------------------------------------------
    #   KILLER / HISTORY ORDERING
    # -----------------------------------------------------------------

    def _order_moves_killer_history(self, moves, depth):
        """
        Return moves sorted by:
         1) If a move is a killer move at this depth, rank it high
         2) If not, use 'history_scores' as a tie-breaker
         3) Possibly also push 'likely captures' forward
        We'll keep it simple: each move gets a 'score'.
        """

        scored_moves = []
        for mv in moves:
            score = 0
            # Check if killer:
            if self._is_killer_move(mv, depth):
                score += 10_000_000  # big jump

            # Add history score:
            # If we have no entry, default is 0
            hist_score = self.history_scores.get(mv, 0)
            score += hist_score

            # Optionally, bump if it's likely capture:
            if self._is_likely_capture(None, mv):
                score += 5_000_000  # or some smaller/larger factor

            scored_moves.append((mv, score))

        # Sort descending by the combined score
        scored_moves.sort(key=lambda x: x[1], reverse=True)
        ordered = [mv[0] for mv in scored_moves]
        return ordered

    def _is_killer_move(self, move, depth):
        """Check if `move` is stored as a killer move for this depth."""
        if depth in self.killer_moves:
            if move in self.killer_moves[depth]:
                return True
        return False

    def _add_killer_move(self, move, depth):
        """Add this move to the killer moves at the given depth."""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [None, None]  # store up to 2

        # if the move is already there, no need
        if move in self.killer_moves[depth]:
            return

        # Insert in front and drop the last
        self.killer_moves[depth].insert(0, move)
        if len(self.killer_moves[depth]) > 2:
            self.killer_moves[depth].pop()

    def _update_history(self, move, depth):
        """If a move improves alpha, increment the history score."""
        old_score = self.history_scores.get(move, 0)
        # The deeper the improvement, the bigger the increment:
        # a common approach is e.g. + 2^(depth)
        # or a simpler +1 per improvement. We'll do something in between:
        bonus = depth * depth
        new_score = old_score + bonus
        self.history_scores[move] = new_score

    # -----------------------------------------------------------------
    #   LMR SUPPORT CHECKS
    # -----------------------------------------------------------------

    def _is_likely_capture(self, game_state, move):
        """
        For example, replicate the logic from NullMove example.
        If `game_state` is None, we skip or do something naive.
        We'll just do something naive: red move => likely capture.
        """
        if not game_state:
            # can't do a real check
            (hole, color) = move
            if color == 0: # 0 => red
                return True
            return False

        # Otherwise do a real check if you want. For brevity, we'll keep it naive:
        (hole, color) = move
        seeds_to_sow = game_state.board[hole, color]
        if color == 0 and seeds_to_sow > 0:
            return True
        return False

class MinimaxAgent6_3(MinimaxAgent6):
    """
    A single merged agent that implements:
      1) Null-move pruning
      2) Futility pruning (razoring) at shallow depths
      3) Killer moves + history heuristic for move ordering
      4) Late Move Reductions (LMR)
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # -- Parameters from "MinimaxAgent6_1" (Null/Futility):
        self.null_move_reduction = 2     # "R" for null move
        self.futility_depth_limit = 2    # Depth at or below which we do futility checks
        self.futility_margin = 100       # Margin for futility

        # -- From "MinimaxAgent6_2" (killer/history + LMR):
        self.killer_moves = {}
        self.history_scores = {}
        self.lmr_move_threshold = 3  # If move index >= 3 => consider LMR
        self.lmr_reduction = 1       # Usually reduce by 1 ply for LMR

    def get_move(self, game_state):
        """
        Optionally reset killer/history for each new top-level move.
        """
        self.killer_moves.clear()
        # You might or might not clear self.history_scores here.
        # Typically, you'd keep history across the iterative deepening loop, but it's up to you.
        return super().get_move(game_state)

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
        Overridden minimax combining:
          - Null Move
          - Futility (razoring)
          - Killer/History ordering
          - Late Move Reductions (LMR)
        """

        # ---- TIME CHECK ----
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # ---- TRANSPOSITION TABLE LOOKUP ----
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

        # ---- DEPTH / GAME OVER CHECK ----
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        # ---- FUTILITY PRUNING (RAZORING) ----
        # If depth <= self.futility_depth_limit, do a static eval; if it's well below alpha,
        # prune away all non-capturing moves.
        if depth <= self.futility_depth_limit:
            static_eval = self.evaluate(game_state)
            if static_eval + self.futility_margin <= alpha:
                # Filter out non-capturing
                capturing_moves = []
                for mv in valid_moves:
                    if self._is_likely_capture(game_state, mv):
                        capturing_moves.append(mv)
                if not capturing_moves:
                    # Hard prune all
                    return static_eval, None
                else:
                    # Only keep capturing moves
                    valid_moves = capturing_moves

        # ---- NULL MOVE PRUNING ----
        if depth >= (self.null_move_reduction + 1) and not game_state.game_over():
            if self._can_do_null_move(game_state):
                null_state = game_state.clone()
                null_state.current_player = 3 - null_state.current_player  # skip my turn
                try:
                    eval_null, _ = self.minimax(
                        null_state,
                        depth - 1 - self.null_move_reduction,
                        alpha,
                        beta,
                        not maximizing_player,
                        start_time,
                        max_time,
                        is_root=False
                    )
                except TimeoutError:
                    raise
                # If a null-move search is still >= beta => prune
                if eval_null >= beta:
                    self.nodes_cut += 1
                    return beta, None

        # ---- ORDER MOVES (KILLER + HISTORY + optional capturing first, etc.) ----
        moves = self._order_moves_killer_history(valid_moves, depth)

        best_move = None
        best_value = float('-inf') if maximizing_player else float('inf')
        LATE_MOVE_PRUNE_THRESHOLD = 5  # your existing threshold

        for i, move in enumerate(moves):
            # Optionally, "late-move pruning" at shallow depth => we do LMR instead of skipping.
            # We'll do LMR if conditions are met:
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            new_depth = depth - 1
            do_lmr = False

            if (
                not is_root
                and depth >= 3
                and i >= self.lmr_move_threshold
                and (not self._is_killer_move(move, depth))
                and (not self._is_likely_capture(game_state, move))
            ):
                do_lmr = True
                reduced_depth = max(1, new_depth - self.lmr_reduction)

                eval_val, _ = self.minimax(
                    clone_state,
                    reduced_depth,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False
                )

                if maximizing_player:
                    if eval_val > best_value:
                        if eval_val > alpha:
                            # Re-search at normal depth
                            eval_val, _ = self.minimax(
                                clone_state,
                                new_depth,
                                alpha,
                                beta,
                                not maximizing_player,
                                start_time,
                                max_time,
                                is_root=False
                            )
                else:
                    if eval_val < best_value:
                        if eval_val < beta:
                            eval_val, _ = self.minimax(
                                clone_state,
                                new_depth,
                                alpha,
                                beta,
                                not maximizing_player,
                                start_time,
                                max_time,
                                is_root=False
                            )
            else:
                # Normal alpha-beta
                eval_val, _ = self.minimax(
                    clone_state,
                    new_depth,
                    alpha,
                    beta,
                    not maximizing_player,
                    start_time,
                    max_time,
                    is_root=False
                )

            # Update alpha/beta
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

            # Beta cutoff => killer move + history update
            if beta <= alpha:
                self.nodes_cut += 1
                self._add_killer_move(move, depth)
                self._update_history(move, depth)
                break

            # If no cutoff, still update history if alpha improved:
            if maximizing_player and (eval_val > alpha):
                self._update_history(move, depth)
            elif (not maximizing_player) and (eval_val < beta):
                self._update_history(move, depth)

        # ---- Store in TT if there's room ----
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            # approximate guess for flag
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # ----------------------------------------------------------------
    #        HELPER METHODS (NULL MOVE, FUTILITY, LMR, ETC.)
    # ----------------------------------------------------------------

    def _can_do_null_move(self, game_state):
        """
        Check if null move is safe:
          - Not near game over
          - More than 7 seeds remain, etc.
        """
        if game_state.game_over():
            return False
        total_seeds = np.sum(game_state.board)
        if total_seeds < 8:
            return False
        return True

    def _is_likely_capture(self, game_state, move):
        """
        Quick check if move is likely to produce a capture.
        For red seeds -> likely capturing. Very naive logic, adapt as needed.
        """
        (hole, color) = move
        if game_state:
            seeds_to_sow = game_state.board[hole, color]
            if color == 0 and seeds_to_sow > 0:  # red sow
                return True
            return False
        else:
            # If no game_state passed, fallback
            if color == 0:
                return True
            return False

    # ----------------------------------------------------------------
    #   KILLER / HISTORY ORDERING
    # ----------------------------------------------------------------
    def _order_moves_killer_history(self, moves, depth):
        """
        Combine killer & history. Possibly also check captures to order them earlier.
        """
        scored_moves = []
        for mv in moves:
            score = 0
            # If it's a killer move for this depth => big bonus
            if self._is_killer_move(mv, depth):
                score += 10_000_000

            # Add history
            hist_score = self.history_scores.get(mv, 0)
            score += hist_score

            # Optionally bump if it's a likely capture
            if self._is_likely_capture(None, mv):
                score += 5_000_000

            scored_moves.append((mv, score))

        scored_moves.sort(key=lambda x: x[1], reverse=True)
        return [mv[0] for mv in scored_moves]

    def _is_killer_move(self, move, depth):
        if depth in self.killer_moves:
            if move in self.killer_moves[depth]:
                return True
        return False

    def _add_killer_move(self, move, depth):
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [None, None]
        if move in self.killer_moves[depth]:
            return
        self.killer_moves[depth].insert(0, move)
        if len(self.killer_moves[depth]) > 2:
            self.killer_moves[depth].pop()

    def _update_history(self, move, depth):
        old_score = self.history_scores.get(move, 0)
        # Increase by depth^2 or any function you like
        bonus = depth * depth
        self.history_scores[move] = old_score + bonus

class MinimaxAgent6_5(Agent):
    """
    Merged agent that features:
      1) Null-move pruning
      2) Futility pruning (razoring)
      3) Killer moves + history heuristic + LMR
      4) Principal Variation (PV) ordering in iterative deepening
      5) Refined 1-ply capture check in move ordering

    Inherits conceptually from MinimaxAgent6,
    but includes logic from 6_3 + 6_4.
    """

    def __init__(self, max_time=2):
        super().__init__()
        self.max_time = max_time

        # --- From 6_3 ---
        self.null_move_reduction = 2         # "R" for null move
        self.futility_depth_limit = 2        # Depth at or below which we do futility checks
        self.futility_margin = 100           # Margin for futility

        self.killer_moves = {}               # For killer heuristic
        self.history_scores = {}             # For history heuristic
        self.lmr_move_threshold = 3          # If move index >= 3 => consider LMR
        self.lmr_reduction = 1               # Usually reduce by 1 ply for LMR

        # --- From 6_4 ---
        # We'll store the principal variation move from the previous iteration
        # to order it first at the root in the next iteration.
        self.principal_variation_move = None

        # Common internal structures
        self.transposition_table = {}
        self.MAX_TABLE_SIZE = 1000000

        # Stats
        self.nodes_cut = 0

    def get_move(self, game_state):
        """
        Iterative deepening approach:
          - We'll keep diving deeper until we run out of time.
          - After each depth, we record the best move found as `principal_variation_move`.
        """
        start_time = time.time()
        depth = 1
        best_move_found = None

        # Optionally reset killer/history each new full turn:
        self.killer_moves.clear()
        # self.history_scores.clear()  # optional, you might keep it

        while True:
            elapsed = time.time() - start_time
            if elapsed >= self.max_time:
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True
                )

                if move is not None:
                    best_move_found = move
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
        A merged alpha-beta implementing:
         - Null move pruning
         - Futility (razoring)
         - Killer/history ordering
         - Late Move Reductions
         - Principal Variation reordering (at root)
         - 1-ply capture check in ordering
        """
        # Time check
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Transposition table lookup
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    if stored_flag == 0:  # FLAG_EXACT
                        return stored_value, stored_move
                    elif stored_flag == 1:  # FLAG_LOWERBOUND
                        alpha = max(alpha, stored_value)
                    elif stored_flag == 2:  # FLAG_UPPERBOUND
                        beta = min(beta, stored_value)
                    if alpha >= beta:
                        self.nodes_cut += 1
                        return stored_value, stored_move

        # Depth/game-over check
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        # Futility pruning (razoring) if depth <= futility_depth_limit
        if depth <= self.futility_depth_limit:
            static_eval = self.evaluate(game_state)
            if static_eval + self.futility_margin <= alpha:
                # keep only capturing moves
                capturing_moves = []
                for mv in valid_moves:
                    if self._is_likely_capture(game_state, mv):
                        capturing_moves.append(mv)
                if not capturing_moves:
                    return static_eval, None
                else:
                    valid_moves = capturing_moves

        # Null move pruning
        if depth >= (self.null_move_reduction + 1) and not game_state.game_over():
            if self._can_do_null_move(game_state):
                null_state = game_state.clone()
                null_state.current_player = 3 - null_state.current_player
                try:
                    eval_null, _ = self.minimax(
                        null_state,
                        depth - 1 - self.null_move_reduction,
                        alpha,
                        beta,
                        not maximizing_player,
                        start_time,
                        max_time,
                        is_root=False
                    )
                except TimeoutError:
                    raise
                if eval_null >= beta:
                    self.nodes_cut += 1
                    return beta, None

        # Sort moves (Killer/History + Principal Variation + 1-ply capture check)
        moves = self._order_moves(game_state, valid_moves, depth, is_root)

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        for i, move in enumerate(moves):
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            new_depth = depth - 1
            do_lmr = False

            # Late Move Reductions (LMR) check
            if (
                not is_root
                and depth >= 3
                and i >= self.lmr_move_threshold
                and (not self._is_killer_move(move, depth))
                and (self._move_score(game_state, move) < 10_000)  # means not a big capture
            ):
                # apply LMR
                do_lmr = True
                reduced_depth = max(1, new_depth - self.lmr_reduction)

                try:
                    eval_val, _ = self.minimax(
                        clone_state,
                        reduced_depth,
                        alpha,
                        beta,
                        not maximizing_player,
                        start_time,
                        max_time,
                        is_root=False
                    )
                except TimeoutError:
                    raise

                # If it "fails high/low," re-search at full depth
                if maximizing_player:
                    if eval_val > best_value and eval_val > alpha:
                        eval_val, _ = self.minimax(
                            clone_state,
                            new_depth,
                            alpha,
                            beta,
                            not maximizing_player,
                            start_time,
                            max_time,
                            is_root=False
                        )
                else:
                    if eval_val < best_value and eval_val < beta:
                        eval_val, _ = self.minimax(
                            clone_state,
                            new_depth,
                            alpha,
                            beta,
                            not maximizing_player,
                            start_time,
                            max_time,
                            is_root=False
                        )
            else:
                # normal alpha-beta
                try:
                    eval_val, _ = self.minimax(
                        clone_state,
                        new_depth,
                        alpha,
                        beta,
                        not maximizing_player,
                        start_time,
                        max_time,
                        is_root=False
                    )
                except TimeoutError:
                    raise

            # alpha/beta update
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
                # cutoff
                self.nodes_cut += 1
                self._add_killer_move(move, depth)
                self._update_history(move, depth)
                break

            # If no cutoff, update history if alpha improved
            if maximizing_player and (eval_val > alpha):
                self._update_history(move, depth)
            elif (not maximizing_player) and (eval_val < beta):
                self._update_history(move, depth)

        # Store in TT if there's room
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = 0  # FLAG_EXACT
            if best_value <= alpha:
                flag = 2  # FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = 1  # FLAG_LOWERBOUND

            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # ------------------------------------------------------------------
    #  ORDER MOVES => Combines principal variation, killer/history,
    #                 and refined 1-ply capture check
    # ------------------------------------------------------------------
    def _order_moves(self, game_state, moves, depth, is_root):
        """
        1) If is_root and we have principal_variation_move, try it first
        2) Then sort the rest by (killer, history, capture check).
        """

        # We'll build a list of (move, score).
        # Then we'll insert the principal variation move at front if is_root.

        # If we do have a PV move, let's separate it out
        pv_move = self.principal_variation_move if is_root else None

        # Moves without PV
        non_pv_moves = [m for m in moves if m != pv_move]

        scored_non_pv = []
        for mv in non_pv_moves:
            base_score = 0

            # Killer?
            if self._is_killer_move(mv, depth):
                base_score += 10_000_000

            # History:
            hist = self.history_scores.get(mv, 0)
            base_score += hist

            # 1-ply capture check
            capture_bonus = self._move_score(game_state, mv)
            base_score += capture_bonus

            scored_non_pv.append((mv, base_score))

        scored_non_pv.sort(key=lambda x: x[1], reverse=True)

        if pv_move and pv_move in moves:
            # Put PV move in front
            return [pv_move] + [m[0] for m in scored_non_pv]
        else:
            return [m[0] for m in scored_non_pv]

    # ------------------------------------------------------------------
    #  _move_score => 1-ply simulation to see if seeds were captured
    # ------------------------------------------------------------------
    def _move_score(self, game_state, move):
        """
        Quick clone + play => compare total seeds before vs after
        to see if there's a capture difference. We return
        10,000 + captured_amount if seeds were captured, else 0.

        You can adapt to see if it was your capture or the opponent's,
        or do more advanced checks.
        """
        before_seeds = np.sum(game_state.board)

        clone = game_state.clone()
        clone.play_move(*move)

        after_seeds = np.sum(clone.board)

        if after_seeds < before_seeds:
            captured_amount = before_seeds - after_seeds
            return 10_000 + captured_amount
        else:
            return 0

    # ------------------------------------------------------------------
    #  Null move & futility checks
    # ------------------------------------------------------------------
    def _can_do_null_move(self, game_state):
        # Check if safe to do null move
        if game_state.game_over():
            return False
        total_seeds = np.sum(game_state.board)
        if total_seeds < 8:
            return False
        return True

    def _is_likely_capture(self, game_state, move):
        """
        We keep the simpler approach from 6_3 for 'likely capture'.
        The true 'refined' check we do in _move_score anyway.
        This is used for quick 'futility' or to skip moves in LMR.
        """
        (hole, color) = move
        seeds_to_sow = game_state.board[hole, color]
        if seeds_to_sow <= 0:
            return False
        # naive approach => if it's red, say "likely capture"
        if color == 0:
            return True
        return False

    # ------------------------------------------------------------------
    #  TT hashing
    # ------------------------------------------------------------------
    def _get_state_hash(self, game_state):
        return hash((game_state.board.tobytes(),
                     game_state.scores.tobytes(),
                     game_state.current_player))

    # ------------------------------------------------------------------
    #  Evaluate
    # ------------------------------------------------------------------
    def evaluate(self, game_state):
        """
        You can keep your usual evaluate or anything else.
        For demonstration, we do something simple from 6, 6_1, etc.
        """
        return game_state.GPT_evaluate_V2()

    # ------------------------------------------------------------------
    #  KILLER / HISTORY
    # ------------------------------------------------------------------
    def _is_killer_move(self, move, depth):
        if depth not in self.killer_moves:
            return False
        return (move in self.killer_moves[depth])

    def _add_killer_move(self, move, depth):
        if depth not in self.killer_moves:
            self.killer_moves[depth] = [None, None]
        if move in self.killer_moves[depth]:
            return
        self.killer_moves[depth].insert(0, move)
        if len(self.killer_moves[depth]) > 2:
            self.killer_moves[depth].pop()

    def _update_history(self, move, depth):
        old_val = self.history_scores.get(move, 0)
        bonus = depth * depth
        self.history_scores[move] = old_val + bonus

class MinimaxAgent6_6(MinimaxAgent6):
    """
    Merges:
      - Null-move pruning
      - Futility pruning (razoring) from 6_1
      - Principal Variation (PV) ordering for iterative deepening
      - Refined _is_likely_capture using a 1-ply sim (from 6_4)
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)

        # --- Null-move + Futility (from 6_1) ---
        self.null_move_reduction = 2       # "R" in null move
        self.futility_depth_limit = 2      # Depth <= which we do futility
        self.futility_margin = 100         # Margin used in futility checks

        # --- For principal variation ordering (from 6_4) ---
        self.principal_variation_move = None

        # We keep the rest from the parent structure (TT, nodes_cut, etc.)

    def get_move(self, game_state):
        """
        Iterative deepening approach:
          We'll store the best move from each iteration in self.principal_variation_move
          so it can be ordered first at the next iteration's root.
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
                    depth=depth,
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
        Combined alpha-beta that includes:
          1) Null move pruning (from 6_1)
          2) Futility pruning (razoring) (from 6_1)
          3) Principal Variation ordering (from 6_4)
          4) Refined _is_likely_capture with a 1-ply simulation (from 6_4)

        We assume other standard alpha-beta logic is in the parent or here.
        """
        # Time check
        if time.time() - start_time >= max_time:
            raise TimeoutError()

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

        # Depth or game-over check
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        # --- Futility pruning (razoring) at shallow depths ---
        if depth <= self.futility_depth_limit:
            static_eval = self.evaluate(game_state)
            # If static_eval + margin <= alpha => prune non-captures
            if static_eval + self.futility_margin <= alpha:
                capturing_moves = []
                for mv in valid_moves:
                    if self._quick_capture_check(game_state, mv):
                        capturing_moves.append(mv)
                if not capturing_moves:
                    # Hard prune all
                    return static_eval, None
                else:
                    valid_moves = capturing_moves

        # --- Null move pruning ---
        if depth >= (self.null_move_reduction + 1) and not game_state.game_over():
            if self._can_do_null_move(game_state):
                null_state = game_state.clone()
                null_state.current_player = 3 - null_state.current_player  # skip
                try:
                    eval_null, _ = self.minimax(
                        null_state,
                        depth - 1 - self.null_move_reduction,
                        alpha,
                        beta,
                        not maximizing_player,
                        start_time,
                        max_time,
                        is_root=False
                    )
                except TimeoutError:
                    raise

                if eval_null >= beta:
                    self.nodes_cut += 1
                    return beta, None

        # --- Order moves with PV + refined capture check ---
        moves = self._order_moves(game_state, valid_moves, depth, is_root)

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

        # Transposition table store
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    # ---------------------------------------------------------
    #  ORDER MOVES => merges PV logic + refined 1-ply capture check
    # ---------------------------------------------------------
    def _order_moves(self, game_state, moves, depth, is_root):
        """
        1) If is_root & we have principal_variation_move => put it first
        2) Then sort the rest by 1-ply capture check
        """
        if is_root and (self.principal_variation_move in moves):
            pv = self.principal_variation_move
            rest = [m for m in moves if m != pv]
            scored_rest = sorted(
                rest,
                key=lambda mv: self._move_score(game_state, mv),
                reverse=True
            )
            return [pv] + scored_rest
        else:
            return sorted(
                moves,
                key=lambda mv: self._move_score(game_state, mv),
                reverse=True
            )

    def _move_score(self, game_state, move):
        """
        1-ply simulation approach from 6_4 to see if seeds were actually captured.
        Return a big bonus if so, or 0 otherwise.
        """
        before_seeds = np.sum(game_state.board)
        clone = game_state.clone()
        clone.play_move(*move)
        after_seeds = np.sum(clone.board)

        if after_seeds < before_seeds:
            captured_amount = before_seeds - after_seeds
            return 10_000 + captured_amount
        return 0

    # ---------------------------------------------------------
    #  quick check for futility capturing
    # ---------------------------------------------------------
    def _quick_capture_check(self, game_state, move):
        """
        A simpler check than _move_score() for futility pruning.
        Could be the same approach, but we can do a short-circuit if we only
        need to see if capturing is at all likely.
        """
        before_seeds = np.sum(game_state.board)
        clone = game_state.clone()
        clone.play_move(*move)
        after_seeds = np.sum(clone.board)
        return (after_seeds < before_seeds)

    # ---------------------------------------------------------
    #  Null Move Check
    # ---------------------------------------------------------
    def _can_do_null_move(self, game_state):
        if game_state.game_over():
            return False
        total_seeds = np.sum(game_state.board)
        if total_seeds < 8:
            return False
        return True

class MinimaxAgent6_6_1(MinimaxAgent6):
    """
    Enhanced Minimax agent implementing:
    - Null-move pruning
    - Futility pruning (razoring)
    - Principal Variation (PV) move ordering
    - Enhanced capture detection
    - Transposition table
    - Late move reduction
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)

        # Null-move pruning parameters
        self.null_move_reduction = 3  # Increased from 2 for more aggressive pruning
        self.null_move_eval_margin = 50  # Safety margin for null-move pruning

        # Futility pruning parameters
        self.futility_depth_limit = 3  # Increased from 2 to apply futility more aggressively
        self.futility_margins = {  # Different margins for different depths
            1: 100,
            2: 300,
            3: 500
        }

        # Principal variation tracking
        self.pv_table = {}  # Store PV moves for each depth
        self.pv_length = [0] * 64  # Track length of PV at each depth

        # Late move reduction parameters
        self.lmr_depth_threshold = 3  # Minimum depth for LMR
        self.lmr_move_threshold = 4   # Number of moves before applying LMR

        # History heuristic table
        self.history_table = {}

    def get_move(self, game_state):
        """
        Iterative deepening with principal variation tracking.
        """
        start_time = time.time()
        depth = 1
        best_move_found = None

        # Clear PV table for new search
        self.pv_table.clear()

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.max_time:
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True,
                    ply=0  # Add ply tracking for PV
                )

                if move is not None:
                    best_move_found = move
                    # Update history heuristic for the best move
                    self._update_history(move, depth)

            except TimeoutError:
                break

            depth += 1

        total_time = time.time() - start_time
        return (best_move_found, total_time, depth - 1)

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False, ply=0):
        """
        Enhanced minimax with all pruning techniques.
        """
        # Original alpha/beta for TT flag determination
        original_alpha = alpha
        original_beta = beta

        # Time check
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Transposition table lookup with enhanced conditions
        if not is_root:
            tt_entry = self._tt_lookup(game_state, depth, alpha, beta)
            if tt_entry is not None:
                return tt_entry

        # Base cases
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        # Null move pruning with verification
        if self._should_try_null_move(game_state, depth, beta):
            eval_null = self._try_null_move(game_state, depth, beta, maximizing_player, start_time, max_time, ply)
            if eval_null is not None and eval_null >= beta:
                return beta, None

        # Futility pruning with dynamic margins
        if self._should_apply_futility(depth, maximizing_player):
            static_eval = self.evaluate(game_state)
            if self._is_futile(static_eval, alpha, depth):
                return self._handle_futility(game_state, valid_moves, static_eval)

        # Move ordering with multiple heuristics
        moves = self._order_moves(game_state, valid_moves, depth, ply)

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        # Main move loop with LMR
        for i, move in enumerate(moves):
            # Late Move Reduction
            current_depth = self._get_move_depth(depth, i, is_root)

            clone_state = game_state.clone()
            clone_state.play_move(*move)

            # Recursive search with appropriate depth
            eval_val, _ = self.minimax(
                clone_state,
                current_depth,
                alpha,
                beta,
                not maximizing_player,
                start_time,
                max_time,
                is_root=False,
                ply=ply + 1
            )

            # Update best value and move
            if self._is_better_move(eval_val, best_value, maximizing_player):
                best_value = eval_val
                best_move = move

                # Update PV table
                if is_root:
                    self._update_pv(move, ply)

            # Alpha-beta update
            if maximizing_player:
                alpha = max(alpha, best_value)
            else:
                beta = min(beta, best_value)

            if beta <= alpha:
                self._update_history(move, depth)  # Killer move found
                break

        # Store position in transposition table
        self._store_tt_entry(game_state, depth, best_value, best_move, original_alpha, original_beta)

        return best_value, best_move

    def _should_try_null_move(self, game_state, depth, beta):
        """Enhanced null-move pruning conditions."""
        if depth < self.null_move_reduction + 1:
            return False
        if game_state.game_over():
            return False

        # Don't try null move if in potential zugzwang
        total_seeds = np.sum(game_state.board)
        if total_seeds < 12:  # Endgame position
            return False

        static_eval = self.evaluate(game_state)
        if static_eval < beta - self.null_move_eval_margin:
            return False

        return True

    def _try_null_move(self, game_state, depth, beta, maximizing_player, start_time, max_time, ply):
        """Execute null move search."""
        null_state = game_state.clone()
        null_state.current_player = 3 - null_state.current_player

        try:
            eval_null, _ = self.minimax(
                null_state,
                depth - 1 - self.null_move_reduction,
                beta - 1,
                beta,
                not maximizing_player,
                start_time,
                max_time,
                is_root=False,
                ply=ply + 1
            )
            return eval_null
        except TimeoutError:
            raise

    def _update_history(self, move, depth):
        """Update history heuristic table."""
        if move not in self.history_table:
            self.history_table[move] = 0
        self.history_table[move] += 2 ** depth

    def _update_pv(self, move, ply):
        """Update principal variation table."""
        self.pv_table[ply] = move
        self.pv_length[ply] = self.pv_length[ply + 1] + 1

    def _get_move_depth(self, depth, move_index, is_root):
        """Determine search depth with LMR."""
        if (not is_root and
            depth >= self.lmr_depth_threshold and
            move_index >= self.lmr_move_threshold):
            return depth - 1
        return depth

    def _order_moves(self, game_state, moves, depth, ply):
        """
        Enhanced move ordering using multiple heuristics:
        1. PV moves
        2. Capturing moves (verified through simulation)
        3. History heuristic
        4. Killer moves
        """
        move_scores = []

        for move in moves:
            score = 0

            # PV move gets highest priority
            if ply in self.pv_table and self.pv_table[ply] == move:
                score += 10000000

            # Capture detection through simulation
            if self._is_capturing_move(game_state, move):
                score += 1000000

            # History heuristic
            score += self.history_table.get(move, 0)

            move_scores.append((move, score))

        return [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]

    def _is_capturing_move(self, game_state, move):
        """Simulate move to check for capture."""
        before_seeds = np.sum(game_state.board)
        clone = game_state.clone()
        clone.play_move(*move)
        after_seeds = np.sum(clone.board)
        return after_seeds < before_seeds

class MinimaxAgent7(MinimaxAgent6_4):
    """
    Inherits from MinimaxAgent6_4, adds specialized domain-specific evaluation:
      - Chain-capture heuristic
      - Starving / forced endgame logic
      - Avoid having 2/3 seeds on our side (optional penalty)
      - Possibly reward red-seed usage when leading, etc.
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # You can define separate weights/bonuses here
        # for the domain-specific heuristics:
        self.CHAIN_CAPTURE_BONUS = 8     # how much to reward each potential 2-or-3 on opponent side
        self.STARVING_BONUS       = 50   # how much to reward leading with <8 seeds left
        self.HOLE_DANGER_PENALTY  = 2    # penalty per hole with 2 or 3 seeds on our side

    def evaluate(self, game_state) -> float:
        """
        Override the evaluate function to incorporate chain-capture, starvation, etc.
        We'll still call super().evaluate() to keep the old logic,
        then add or subtract domain-specific heuristics.
        """
        base_eval = super().evaluate(game_state)

        # Identify basic references
        my_index = game_state.current_player - 1
        opp_index = 1 - my_index
        my_holes = game_state.player_holes[game_state.current_player]
        opp_holes = game_state.player_holes[3 - game_state.current_player]

        # Score difference
        score_diff = game_state.scores[my_index] - game_state.scores[opp_index]

        # Count seeds on board
        board_total = np.sum(game_state.board)
        # Mobility checks
        opp_moves = 0
        # The game_state's get_valid_moves is always for current_player, so we do a quick hack:
        # Temporarily swap current_player to check the opponent's possible moves.
        current_player_backup = game_state.current_player
        game_state.current_player = 3 - current_player_backup
        opp_valid_moves = game_state.get_valid_moves()
        opp_moves = len(opp_valid_moves)
        # Restore current_player
        game_state.current_player = current_player_backup

        # ----------------------------------------------------------------------
        # CHAIN-CAPTURE POTENTIAL
        # A simpler (static) approach is to count how many holes on the opponent's side
        # are exactly 1 or 4 seeds => they can be turned into 2 or 3 on your next sow.
        # You can also count how many holes on your side are 2 or 3 (which is dangerous).
        # This is just a static measure; you might do more advanced checks if you like.
        # ----------------------------------------------------------------------
        total_seeds_per_hole = np.sum(game_state.board, axis=1)

        # Count how many opp holes are at 1 or 4 => potential to become 2/3
        opp_1_or_4 = np.sum((total_seeds_per_hole[opp_holes] == 1) |
                            (total_seeds_per_hole[opp_holes] == 4))

        # We'll give a small bonus for each such hole,
        # representing potential chain capture on next move
        chain_capture_score = self.CHAIN_CAPTURE_BONUS * opp_1_or_4

        # ----------------------------------------------------------------------
        # STARVING / ENDGAME LOGIC
        # If we are leading and board_total < 8 => big bonus (we're about to lock in a win)
        # If opponent has no moves => also a big bonus.
        # ----------------------------------------------------------------------
        starving_score = 0
        if board_total < 8:
            # If I'm leading, add a big bonus to reflect nearly guaranteed win
            if score_diff > 0:
                starving_score += self.STARVING_BONUS * (1 + score_diff / 2.0)
                # factor in score_diff a bit

        # If the opponent literally has no valid moves => also a bonus
        if opp_moves == 0:
            # Usually that means they can’t move, so you can starve out or capture more seeds
            # If you're already ahead, that likely means you'll soon end the game winning
            starving_score += self.STARVING_BONUS * (1 + max(score_diff, 1) / 2.0)

        # ----------------------------------------------------------------------
        # HOLE DANGER: Having your own holes at exactly 2 or 3 seeds is risky
        # Because the opponent can sow a single red seed there to capture them.
        # We'll apply a small penalty for each hole on your side that has 2 or 3 seeds.
        # ----------------------------------------------------------------------
        my_2_or_3 = np.sum((total_seeds_per_hole[my_holes] == 2) |
                           (total_seeds_per_hole[my_holes] == 3))
        hole_danger_penalty = -1 * self.HOLE_DANGER_PENALTY * my_2_or_3

        # ----------------------------------------------------------------------
        # Combine everything
        # ----------------------------------------------------------------------
        domain_specific_eval = chain_capture_score + starving_score + hole_danger_penalty

        return base_eval + domain_specific_eval

class MinimaxAgent8(MinimaxAgent6_4):
    """
    A 'greedy' variant that:
      1) Strongly rewards immediate captures (looks at all possible moves this turn).
      2) Penalizes having holes with 2 or 3 seeds on our side (easy for opponent to capture).
      3) Gives a bonus for opponent holes at 0 seeds (starving them).
    Inherits all the search logic from MinimaxAgent6_4, only overrides evaluate().
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        # You can tweak these constants:
        self.IMMEDIATE_CAPTURE_WEIGHT = 25    # weight per seed captured immediately
        self.HOLE_2_3_PENALTY         = 3     # penalty per hole with 2 or 3 seeds on our side
        self.OPP_HOLE_EMPTY_BONUS     = 2     # bonus per empty hole on opponent's side

    def evaluate(self, game_state) -> float:
        """
        Overrides the MinimaxAgent6_4 evaluation to add a greedy, immediate-capture twist
        plus some defensive and starvation logic.
        """
        # 1) Start with the base evaluate from 6_4 (which already includes your standard
        #    scoring, mobility, etc., plus any 1-ply capture ordering at move_score).
        base_eval = super().evaluate(game_state)

        # 2) Grab references: which holes are ours vs. opponent's
        my_index = game_state.current_player - 1
        opp_index = 1 - my_index
        my_holes = game_state.player_holes[game_state.current_player]
        opp_holes = game_state.player_holes[3 - game_state.current_player]

        # 3) Check immediate captures: simulate *all* valid moves for the current player
        #    and see the maximum number of seeds we can capture this turn.
        current_valid_moves = game_state.get_valid_moves()
        max_immediate_capture = 0
        if current_valid_moves:
            for move in current_valid_moves:
                clone_state = game_state.clone()
                before_seeds = np.sum(clone_state.board)
                clone_state.play_move(*move)
                after_seeds = np.sum(clone_state.board)
                captured = before_seeds - after_seeds
                if captured > max_immediate_capture:
                    max_immediate_capture = captured

        # 4) Penalize having holes with 2 or 3 seeds on our side
        total_seeds_per_hole = np.sum(game_state.board, axis=1)
        my_2_or_3 = np.sum((total_seeds_per_hole[my_holes] == 2) |
                           (total_seeds_per_hole[my_holes] == 3))
        hole_2_3_penalty = - self.HOLE_2_3_PENALTY * my_2_or_3

        # 5) Reward each 0-seed hole on the opponent's side
        opp_empty_holes = np.sum(total_seeds_per_hole[opp_holes] == 0)
        opp_empty_bonus = self.OPP_HOLE_EMPTY_BONUS * opp_empty_holes

        # 6) Combine everything into a single domain-specific score
        #    - immediate_capture_score: big plus for large immediate captures
        immediate_capture_score = self.IMMEDIATE_CAPTURE_WEIGHT * max_immediate_capture

        domain_specific_eval = immediate_capture_score + hole_2_3_penalty + opp_empty_bonus

        return base_eval + domain_specific_eval

class MinimaxAgent6_4_2(MinimaxAgent6_4):
    """
    Inherits from MinimaxAgent6_4:
      1) Adds killer/history move ordering.
      2) Stores a full principal variation at each node (via PV table) for deeper iterative deepening.
    """

    def __init__(self, max_time=2, max_depth=50):
        super().__init__(max_time)

        # The maximum search depth we might handle (arbitrary upper bound).
        self.MAX_DEPTH = max_depth

        # 1) Killer moves: for each depth, we keep a small dictionary or map
        #    from move -> "killer score". We’ll track the top 1 or 2 killer moves per depth.
        self.killer_moves = [defaultdict(int) for _ in range(self.MAX_DEPTH + 1)]

        # 2) History table: move -> integer. The higher the integer, the more we order that move first.
        self.history_table = defaultdict(int)

        # 3) PV table: store a best line for each (state_hash, depth).
        #    Key: (hash, depth), Value: [list of moves].
        self.pv_table = {}

    def get_move(self, game_state):
        """
        Similar iterative deepening as in MinimaxAgent6_4, but we’ll store a full PV each iteration.
        """
        start_time = time.time()
        depth = 1
        best_move_found = None
        best_line_found = []

        while True:
            if (time.time() - start_time) >= self.max_time:
                break

            try:
                # We also retrieve a "best_line" from the search
                eval_val, move, line = self.minimax(
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
                    best_line_found = line

                # We store the entire PV for the root node
                # so next iteration can reorder moves.
                state_hash = self._get_state_hash(game_state)
                self.pv_table[(state_hash, depth)] = line

            except TimeoutError:
                break

            depth += 1

        compute_time = time.time() - start_time
        return (best_move_found, compute_time, depth - 1)

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
        Overridden to:
          1) Return (best_eval, best_move, best_line).
          2) Update killer/history tables on beta-cutoffs.
          3) Check our PV table for deeper reordering, not just root.
        """
        if (time.time() - start_time) >= max_time:
            raise TimeoutError()

        state_hash = self._get_state_hash(game_state)
        # Transposition Table check (same logic)
        if not is_root and state_hash in self.transposition_table:
            stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                if stored_flag == self.FLAG_EXACT:
                    return stored_value, stored_move, [stored_move]
                elif stored_flag == self.FLAG_LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == self.FLAG_UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    self.nodes_cut += 1
                    # Return a trivial line with just the stored_move
                    return stored_value, stored_move, [stored_move]

        # Terminal or depth-0
        if game_state.game_over() or depth == 0:
            static_eval = self.evaluate(game_state)
            return static_eval, None, []

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            # No moves => evaluate
            static_eval = self.evaluate(game_state)
            return static_eval, None, []

        # -------------- ORDER MOVES --------------
        moves = self._order_moves(game_state, valid_moves, depth, is_root)

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None
        best_line = []

        for mv in moves:
            clone_state = game_state.clone()
            clone_state.play_move(*mv)

            try:
                eval_val, _, child_line = self.minimax(
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
                    best_move = mv
                    best_line = [mv] + child_line
                alpha = max(alpha, best_value)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = mv
                    best_line = [mv] + child_line
                beta = min(beta, best_value)

            # Beta cutoff
            if beta <= alpha:
                self.nodes_cut += 1

                # -------------- KILLER / HISTORY UPDATE --------------
                # If not is_root, record the move in killer & history
                if depth <= self.MAX_DEPTH:
                    self.killer_moves[depth][mv] += 1
                    self.history_table[mv] += depth * depth

                break

        # Store in TT
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        # Also store this node’s best line in the PV table
        self.pv_table[(state_hash, depth)] = best_line

        return best_value, best_move, best_line

    # ----------------------------------------------------------------------
    #        Overridden _order_moves with PV + Killer/History
    # ----------------------------------------------------------------------
    def _order_moves(self, game_state, moves, depth, is_root):
        """
        Now we incorporate:
          1) Full PV ordering if we have a stored line in self.pv_table[(state_hash, depth)].
          2) Killer/History heuristics in the final sorting.
        """
        state_hash = self._get_state_hash(game_state)

        # 1) If we have a known PV line for (state_hash, depth), put that move first
        #    (like a multi-move approach: we only reorder the 1st move in that line).
        pv_line = self.pv_table.get((state_hash, depth), [])
        pv_move = pv_line[0] if len(pv_line) > 0 else None

        # 2) We'll define an internal scoring function that includes:
        #    - existing 1-ply capture check (via self._move_score)
        #    - killer moves
        #    - history table
        def move_order_score(mv):
            base_score = self._move_score(game_state, mv)  # your capture-based scoring

            # Killer bonus
            killer_score = self.killer_moves[depth].get(mv, 0)
            killer_bonus = 1000 * killer_score

            # History bonus
            hist_bonus = self.history_table[mv] * 0.1

            return base_score + killer_bonus + hist_bonus

        # If the PV move is in the list, put it up front
        if pv_move in moves:
            first = [pv_move]
            rest = [m for m in moves if m != pv_move]
            # Sort the rest by move_order_score
            rest_sorted = sorted(rest, key=move_order_score, reverse=True)
            return first + rest_sorted
        else:
            # Sort everything by move_order_score
            return sorted(moves, key=move_order_score, reverse=True)

class MinimaxAgent6_6_2(MinimaxAgent6):
    """
    Enhanced Minimax agent implementing:
    - Null-move pruning
    - Futility pruning (razoring)
    - Principal Variation (PV) move ordering
    - Enhanced capture detection
    - Transposition table
    - Late move reduction
    """

    def __init__(self, max_time=2):
        super().__init__(max_time)
        self.null_move_reduction = 3
        self.null_move_eval_margin = 50
        self.futility_depth_limit = 3
        self.futility_margins = {1: 100, 2: 300, 3: 500}
        self.pv_table = {}
        self.pv_length = [0] * 64
        self.lmr_depth_threshold = 3
        self.lmr_move_threshold = 4
        self.history_table = {}
        self.max_depth = 50  # Increased max depth
        self.max_recursion_count = 1000

    def _should_apply_futility(self, depth: int, maximizing_player: bool) -> bool:
        return (depth <= self.futility_depth_limit and
                not maximizing_player)  # Only apply to minimizing nodes

    def _is_futile(self, static_eval: float, alpha: float, depth: int) -> bool:
        margin = self.futility_margins.get(depth, 100)
        return static_eval + margin <= alpha

    def _handle_futility(self, game_state, valid_moves, static_eval):
        """
        Handle futility pruning by limiting search to capturing moves if any.
        """
        # Filter capturing moves
        captures = [move for move in valid_moves if self._is_capturing_move(game_state, move)]

        if not captures:
            # No captures, prune and return static evaluation
            return static_eval, None

        # Continue with captures if any are present
        return None, captures

    def _tt_lookup(self, game_state, depth: int, alpha: float, beta: float):
        state_hash = self._get_state_hash(game_state)
        if state_hash in self.transposition_table:
            stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                if stored_flag == self.FLAG_EXACT:
                    return stored_value, stored_move
                if stored_flag == self.FLAG_LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == self.FLAG_UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value, stored_move
        return None

    def _store_tt_entry(self, game_state, depth: int, value: float, move, alpha: float, beta: float):
        if len(self.transposition_table) >= self.MAX_TABLE_SIZE:
            return

        state_hash = self._get_state_hash(game_state)
        flag = self.FLAG_EXACT
        if value <= alpha:
            flag = self.FLAG_UPPERBOUND
        elif value >= beta:
            flag = self.FLAG_LOWERBOUND

        self.transposition_table[state_hash] = (depth, value, flag, move)

    def _is_better_move(self, eval_val: float, best_value: float, maximizing_player: bool) -> bool:
        return ((maximizing_player and eval_val > best_value) or
                (not maximizing_player and eval_val < best_value))

    def get_move(self, game_state):
        start_time = time.time()
        best_move_found = None
        self.pv_table.clear()

        for depth in range(1, 10):  # Iterative deepening with reasonable limit
            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth=depth,
                    alpha=-math.inf,
                    beta=math.inf,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True,
                    ply=0
                )

                if move is not None:
                    best_move_found = move

            except TimeoutError:
                break

        return best_move_found, time.time() - start_time, depth - 1

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False, ply=0):
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        if ply >= self.max_depth or game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return self.evaluate(game_state), None

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        # Simplified move ordering for stability
        moves = valid_moves
        if depth > 2:  # Only order moves at deeper depths
            try:
                moves = self._order_moves(game_state, valid_moves, depth, ply)
            except:
                pass

        for move in moves:
            try:
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
                    is_root=False,
                    ply=ply + 1
                )

                if self._is_better_move(eval_val, best_value, maximizing_player):
                    best_value = eval_val
                    best_move = move

                if maximizing_player:
                    alpha = max(alpha, best_value)
                else:
                    beta = min(beta, best_value)

                if beta <= alpha:
                    break

            except RecursionError:
                return self.evaluate(game_state), move
            except:
                continue

        return best_value, best_move

    def _should_try_null_move(self, game_state, depth, beta):
        """Enhanced null-move pruning conditions."""
        if depth < self.null_move_reduction + 1:
            return False
        if game_state.game_over():
            return False

        # Don't try null move if in potential zugzwang
        total_seeds = np.sum(game_state.board)
        if total_seeds < 12:  # Endgame position
            return False

        static_eval = self.evaluate(game_state)
        if static_eval < beta - self.null_move_eval_margin:
            return False

        return True

    def _try_null_move(self, game_state, depth, beta, maximizing_player, start_time, max_time, ply):
        """Execute null move search."""
        null_state = game_state.clone()
        null_state.current_player = 3 - null_state.current_player

        try:
            eval_null, _ = self.minimax(
                null_state,
                depth - 1 - self.null_move_reduction,
                beta - 1,
                beta,
                not maximizing_player,
                start_time,
                max_time,
                is_root=False,
                ply=ply + 1
            )
            return eval_null
        except TimeoutError:
            raise

    def _update_history(self, move, depth):
        """Update history heuristic table."""
        if move not in self.history_table:
            self.history_table[move] = 0
        self.history_table[move] += 2 ** depth

    def _update_pv(self, move, ply):
        """Update principal variation table."""
        self.pv_table[ply] = move
        self.pv_length[ply] = self.pv_length[ply + 1] + 1

    def _get_move_depth(self, depth, move_index, is_root):
        """Determine search depth with LMR."""
        if (not is_root and
            depth >= self.lmr_depth_threshold and
            move_index >= self.lmr_move_threshold):
            return depth - 1
        return depth

    def _order_moves(self, game_state, moves, depth, ply):
        """
        Enhanced move ordering using multiple heuristics:
        1. PV moves
        2. Capturing moves (verified through simulation)
        3. History heuristic
        4. Killer moves
        """
        move_scores = []

        for move in moves:
            score = 0

            # PV move gets highest priority
            if ply in self.pv_table and self.pv_table[ply] == move:
                score += 10000000

            # Capture detection through simulation
            if self._is_capturing_move(game_state, move):
                score += 1000000

            # History heuristic
            score += self.history_table.get(move, 0)

            move_scores.append((move, score))

        return [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]

    def _is_capturing_move(self, game_state, move):
        clone = game_state.clone()
        initial_score = clone.scores[clone.current_player - 1]

        try:
            clone.play_move(*move)
            return clone.scores[clone.current_player - 1] > initial_score
        except:
            return False

class MinimaxAgent6_4_3(MinimaxAgent6_4):
    def __init__(self, max_time=2):
        super().__init__(max_time)
        self.history_table = {}  # Store move history scores
        self.killer_moves = [[] for _ in range(64)]  # Killer moves per depth
        self.MAX_KILLER_MOVES = 2

        # Aspiration windows for iterative deepening
        self.ASPIRATION_WINDOW = 50

        # Enhanced evaluation weights
        self.ENDGAME_THRESHOLD = 16  # Total seeds threshold for endgame
        self.TEMPO_BONUS = 5  # Bonus for maintaining initiative

        # Progressive history scaling
        self.HISTORY_MAX = 10000
        self.HISTORY_DECAY = 0.9

    def get_move(self, game_state):
        start_time = time.time()
        depth = 1
        best_move_found = None

        # Aspiration windows
        alpha = float('-inf')
        beta = float('inf')
        window = self.ASPIRATION_WINDOW

        while True:
            if time.time() - start_time >= self.max_time:
                break

            try:
                # Use aspiration windows after first iteration
                if depth > 1:
                    alpha = max(prev_eval - window, float('-inf'))
                    beta = min(prev_eval + window, float('inf'))

                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth,
                    alpha,
                    beta,
                    maximizing_player=True,
                    start_time=start_time,
                    max_time=self.max_time,
                    is_root=True
                )

                # Store previous evaluation for aspiration windows
                prev_eval = eval_val

                if move:
                    best_move_found = move
                    self.principal_variation_move = move
                    # Update history table with depth-based score
                    self._update_history(move, depth)

            except TimeoutError:
                break
            except AspirationFailure:
                # Window too narrow - retry with full window
                window *= 2
                continue

            depth += 1
            window = self.ASPIRATION_WINDOW  # Reset window

        return (best_move_found, time.time() - start_time, depth - 1)

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        alpha_orig = alpha

        # Enhanced transposition table lookup
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            tt_entry = self.transposition_table.get(state_hash)
            if tt_entry and tt_entry[0] >= depth:
                stored_depth, stored_value, stored_flag, stored_move = tt_entry
                if stored_flag == self.FLAG_EXACT:
                    return stored_value, stored_move
                elif stored_flag == self.FLAG_LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == self.FLAG_UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    return stored_value, stored_move

        if game_state.game_over() or depth == 0:
            return self._evaluate_enhanced(game_state, depth), None

        moves = self._order_moves_enhanced(game_state, depth)
        if not moves:
            return self._evaluate_enhanced(game_state, depth), None

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        # Internal iterative deepening
        if depth >= 4 and not self.principal_variation_move and not is_root:
            try:
                _, iid_move = self.minimax(
                    game_state,
                    depth - 2,
                    alpha,
                    beta,
                    maximizing_player,
                    start_time,
                    max_time
                )
                if iid_move:
                    self.principal_variation_move = iid_move
            except TimeoutError:
                raise

        # Late move reduction
        for i, move in enumerate(moves):
            # Full-depth search for first few moves
            reduction = 0
            if depth >= 3 and i >= 3 and not self._is_capture_move(game_state, move):
                reduction = 1 if depth < 6 else 2

            clone_state = game_state.clone()
            clone_state.play_move(*move)

            try:
                if reduction == 0:
                    eval_val, _ = self.minimax(
                        clone_state,
                        depth - 1,
                        -beta,
                        -alpha,
                        not maximizing_player,
                        start_time,
                        max_time
                    )
                    eval_val = -eval_val
                else:
                    # Reduced depth search
                    eval_val, _ = self.minimax(
                        clone_state,
                        depth - 1 - reduction,
                        -alpha - 1,
                        -alpha,
                        not maximizing_player,
                        start_time,
                        max_time
                    )
                    eval_val = -eval_val
                    # Re-search if promising
                    if eval_val > alpha:
                        eval_val, _ = self.minimax(
                            clone_state,
                            depth - 1,
                            -beta,
                            -alpha,
                            not maximizing_player,
                            start_time,
                            max_time
                        )
                        eval_val = -eval_val

            except TimeoutError:
                raise

            if maximizing_player:
                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                    if depth == 1:
                        self._update_killer_moves(move, depth)
                alpha = max(alpha, eval_val)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                    if depth == 1:
                        self._update_killer_moves(move, depth)
                beta = min(beta, eval_val)

            if beta <= alpha:
                self._update_killer_moves(move, depth)
                break

        # Update transposition table
        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            flag = self.FLAG_EXACT
            if best_value <= alpha_orig:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return best_value, best_move

    def _evaluate_enhanced(self, game_state, depth):
        base_eval = self.evaluate(game_state)
        """
        # Endgame adjustments
        total_seeds = np.sum(game_state.board)
        if total_seeds <= self.ENDGAME_THRESHOLD:
            base_eval *= 1.2  # Increase importance of material in endgame

        # Tempo bonus
        if game_state.current_player == 1:
            base_eval += self.TEMPO_BONUS
        """
        return base_eval

    def _order_moves_enhanced(self, game_state, depth):
        moves = game_state.get_valid_moves()

        # Score moves based on multiple criteria
        move_scores = []
        for move in moves:
            score = 0

            # PV move gets highest priority
            if move == self.principal_variation_move:
                score += 10000000

            # Killer moves get high priority
            if move in self.killer_moves[depth]:
                score += 1000000 * (self.MAX_KILLER_MOVES -
                                  self.killer_moves[depth].index(move))

            # History heuristic
            score += self.history_table.get(move, 0)

            # Capture potential
            if self._is_capture_move(game_state, move):
                score += 500000

            move_scores.append((move, score))

        return [move for move, _ in sorted(move_scores, key=lambda x: x[1], reverse=True)]

    def _update_history(self, move, depth):
        current_score = self.history_table.get(move, 0)
        depth_bonus = depth * depth
        self.history_table[move] = min(
            self.HISTORY_MAX,
            int(current_score * self.HISTORY_DECAY + depth_bonus)
        )

    def _update_killer_moves(self, move, depth):
        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].insert(0, move)
            if len(self.killer_moves[depth]) > self.MAX_KILLER_MOVES:
                self.killer_moves[depth].pop()

    def _is_capture_move(self, game_state, move):
        clone = game_state.clone()
        pre_score = clone.scores[clone.current_player - 1]
        clone.play_move(*move)
        post_score = clone.scores[clone.current_player - 1]
        return post_score > pre_score

class AspirationFailure(Exception):
    pass

class HailMarry(MinimaxAgent6):
    """
    A MinimaxAgent variant focused purely on defense, avoiding captures at all costs
    while ignoring offensive opportunities.
    """
    def __init__(self, max_time=2):
        super().__init__(max_time)

        # Override weights to focus entirely on defensive metrics
        self.VULNERABILITY_WEIGHT = 100  # Weight for avoiding vulnerable positions
        self.DISTANCE_WEIGHT = 50       # Weight for keeping seeds away from opponent
        self.DISTRIBUTION_WEIGHT = 30   # Weight for maintaining good seed distribution
        self.MOBILITY_WEIGHT = 20       # Weight for maintaining move options

        # Precompute opponent holes for faster checks
        self.opponent_holes = {
            1: np.array([1, 3, 5, 7, 9, 11, 13, 15]),
            2: np.array([0, 2, 4, 6, 8, 10, 12, 14])
        }

        # Cache for vulnerability checks
        self.vulnerability_cache = {}
        self.MAX_VULNERABILITY_CACHE = 100000

    @lru_cache(maxsize=10000)
    def evaluate(self, game_state) -> float:
        """
        Purely defensive evaluation function that only cares about:
        1. Avoiding vulnerable positions (2-3 seeds)
        2. Keeping seeds away from opponent's side
        3. Maintaining good seed distribution
        4. Maintaining mobility
        """
        my_index = game_state.current_player - 1
        my_holes = self.player_holes[game_state.current_player]
        opp_holes = self.opponent_holes[game_state.current_player]

        # 1. Vulnerability score (avoid positions that could be captured)
        vulnerability_score = self._calculate_vulnerability(game_state, my_holes)

        # 2. Distance score (prefer keeping seeds away from opponent's capture range)
        distance_score = self._calculate_distance_safety(game_state, my_holes, opp_holes)

        # 3. Distribution score (prefer evenly distributed seeds)
        distribution_score = self._calculate_distribution(game_state, my_holes)

        # 4. Mobility score (maintain move options)
        mobility_score = self._calculate_mobility(game_state, my_holes)

        # Combine scores with weights
        return (
            self.VULNERABILITY_WEIGHT * vulnerability_score +
            self.DISTANCE_WEIGHT * distance_score +
            self.DISTRIBUTION_WEIGHT * distribution_score +
            self.MOBILITY_WEIGHT * mobility_score
        )

    def _calculate_vulnerability(self, game_state, my_holes):
        """Calculate how vulnerable our positions are to capture"""
        state_key = (game_state.board.tobytes(), tuple(my_holes))

        if state_key in self.vulnerability_cache:
            return self.vulnerability_cache[state_key]

        # Count holes with 2 or 3 seeds (highly vulnerable)
        total_seeds = game_state.board.sum(axis=1)
        vulnerable_positions = np.sum((total_seeds == 2) | (total_seeds == 3))

        # Count holes with 1 or 4 seeds (potentially vulnerable)
        semi_vulnerable = np.sum((total_seeds == 1) | (total_seeds == 4))

        # Penalize vulnerable positions heavily
        vulnerability_score = -(vulnerable_positions * 2 + semi_vulnerable)

        # Cache the result
        if len(self.vulnerability_cache) < self.MAX_VULNERABILITY_CACHE:
            self.vulnerability_cache[state_key] = vulnerability_score

        return vulnerability_score

    def _calculate_distance_safety(self, game_state, my_holes, opp_holes):
        """Calculate how safe our seeds are based on distance from opponent"""
        total_distance = 0
        my_seeds = game_state.board[my_holes]

        # Calculate weighted distance of our seeds from opponent's holes
        for hole in my_holes:
            seeds = game_state.board[hole].sum()
            if seeds > 0:
                # Find minimum distance to any opponent hole
                distances = np.abs(hole - opp_holes)
                min_distance = np.min(distances)
                # Larger distance is better
                total_distance += seeds * min_distance

        return total_distance

    def _calculate_distribution(self, game_state, my_holes):
        """Calculate how well distributed our seeds are"""
        my_seeds = game_state.board[my_holes].sum(axis=1)

        # Prefer more even distribution (lower standard deviation)
        std_dev = np.std(my_seeds)

        # Also prefer having seeds spread across more holes
        holes_with_seeds = np.sum(my_seeds > 0)

        return holes_with_seeds - std_dev

    def _calculate_mobility(self, game_state, my_holes):
        """Calculate our movement options"""
        # Count how many different moves we have available
        red_moves = np.sum(game_state.board[my_holes, 0] > 0)
        blue_moves = np.sum(game_state.board[my_holes, 1] > 0)

        return red_moves + blue_moves

    def _move_score(self, game_state, move):
        """
        Override move scoring to prioritize defensive moves.
        Only care about moves that improve our defensive position.
        """
        # Simulate the move
        clone = game_state.clone()
        clone.play_move(*move)

        # Calculate vulnerability before and after move
        before_vuln = self._calculate_vulnerability(
            game_state,
            self.player_holes[game_state.current_player]
        )
        after_vuln = self._calculate_vulnerability(
            clone,
            self.player_holes[clone.current_player]
        )

        # Prioritize moves that reduce vulnerability
        vulnerability_improvement = after_vuln - before_vuln

        # Check if move leads to opponent having capture opportunities
        opponent_captures = self._count_opponent_capture_opportunities(clone)

        # Heavily penalize moves that give opponent capture opportunities
        if opponent_captures > 0:
            return float('-inf')

        return vulnerability_improvement

    def _count_opponent_capture_opportunities(self, game_state):
        """Count how many capture opportunities the opponent would have"""
        opp_holes = self.opponent_holes[game_state.current_player]
        total_seeds = game_state.board.sum(axis=1)
        return np.sum((total_seeds == 2) | (total_seeds == 3))

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        """
        Override minimax to focus on defensive play.
        Prune branches that lead to highly vulnerable positions early.
        """
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        # Early cutoff if position is too vulnerable
        if not is_root:
            vulnerability = self._calculate_vulnerability(
                game_state,
                self.player_holes[game_state.current_player]
            )
            if vulnerability < -3:  # More than 3 vulnerable positions
                return float('-inf'), None

        # Rest of minimax implementation remains similar to parent class
        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        moves = self._order_moves(game_state, game_state.get_valid_moves(), depth, is_root)
        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None

        for move in moves:
            # Skip moves that immediately lead to captures
            clone_state = game_state.clone()
            clone_state.play_move(*move)
            if self._count_opponent_capture_opportunities(clone_state) > 0:
                continue

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
                break

        return best_value, best_move

class DefensiveAgent:
    def __init__(self, max_depth=100, max_time=2):
        self.max_depth = max_depth
        self.max_time = max_time
        self.VULNERABILITY_WEIGHT = -100  # Heavy penalty for vulnerabilities
        self.DISTRIBUTION_WEIGHT = 10    # Bonus for better distribution
        self.MOBILITY_WEIGHT = 5         # Bonus for more available moves

    def evaluate(self, game_state):
        """
        Defensive evaluation function prioritizing avoiding vulnerabilities.
        """
        my_holes = self._get_my_holes(game_state)
        total_seeds = game_state.board.sum(axis=1)

        # Vulnerability: Penalize holes with 2 or 3 seeds
        vulnerabilities = np.sum((total_seeds == 2) | (total_seeds == 3))

        # Distribution: Prefer an even spread of seeds
        distribution = -np.std([game_state.board[hole].sum() for hole in my_holes])

        # Mobility: Count available moves
        mobility = np.sum(game_state.board[my_holes] > 0)

        return (self.VULNERABILITY_WEIGHT * vulnerabilities +
                self.DISTRIBUTION_WEIGHT * distribution +
                self.MOBILITY_WEIGHT * mobility)

    def get_move(self, game_state):
        """
        Determines the best defensive move using a bounded depth minimax.
        """
        start_time = time.time()
        best_move = None
        best_score = float('-inf')

        for move in game_state.get_valid_moves():
            clone = game_state.clone()
            clone.play_move(*move)
            score = self._minimax(clone, self.max_depth, False, start_time)
            if score > best_score:
                best_score = score
                best_move = move

            # Stop if time is up
            if time.time() - start_time >= self.max_time:
                break

        compute_time = time.time() - start_time
        return best_move, compute_time, self.max_depth


    def _minimax(self, game_state, depth, maximizing_player, start_time):
        """
        Minimax algorithm with early cutoff for defensive evaluation.
        """
        if time.time() - start_time >= self.max_time or depth == 0 or game_state.game_over():
            return self.evaluate(game_state)

        if maximizing_player:
            best_score = float('-inf')
            for move in game_state.get_valid_moves():
                clone = game_state.clone()
                clone.play_move(*move)
                score = self._minimax(clone, depth - 1, False, start_time)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in game_state.get_valid_moves():
                clone = game_state.clone()
                clone.play_move(*move)
                score = self._minimax(clone, depth - 1, True, start_time)
                best_score = min(best_score, score)
            return best_score

    def _get_my_holes(self, game_state):
        """
        Helper function to get the indices of the player's holes.
        """
        if game_state.current_player == 1:
            return np.arange(0, 16, 2)  # Odd-numbered holes
        else:
            return np.arange(1, 16, 2)  # Even-numbered holes

class DefensiveMinimaxAgent(Agent):
    def __init__(self, max_time=2):
        self.max_time = max_time
        self.transposition_table = {}
        self.move_ordering = {}
        self.MAX_TABLE_SIZE = 1000000

        # Extreme defensive weights
        self.IMMEDIATE_THREAT_WEIGHT = 10000  # Extreme penalty for immediate capture threats
        self.SETUP_THREAT_WEIGHT = 5000       # Heavy penalty for opponent capture setups
        self.SAFE_POSITION_WEIGHT = 100       # Reward for maintaining safe positions

    def evaluate(self, game_state) -> float:
        """
        Focused defensive evaluation that looks for specific threatening patterns
        and immediate capture possibilities.
        """
        my_index = game_state.current_player - 1
        my_holes = game_state.player_holes[game_state.current_player]
        opp_holes = game_state.player_holes[3 - game_state.current_player]

        score = 0
        board_sums = np.sum(game_state.board, axis=1)

        # Check for immediate capture threats (holes with 1 or 4 seeds)
        immediate_threats = np.sum((board_sums[my_holes] == 1) | (board_sums[my_holes] == 4))
        score -= immediate_threats * self.IMMEDIATE_THREAT_WEIGHT

        # Check for opponent's setup moves
        for opp_hole in opp_holes:
            next_hole = (opp_hole + 1) % 16
            if next_hole in my_holes:
                # Check if opponent can create a capture threat in one move
                if board_sums[opp_hole] > 0:  # If opponent has seeds to move
                    distance_to_my_hole = (next_hole - opp_hole) % 16
                    if board_sums[opp_hole] >= distance_to_my_hole:  # Can reach my hole
                        current_seeds = board_sums[next_hole]
                        if current_seeds in [0, 1, 3]:  # Could become 1 or 4
                            score -= self.SETUP_THREAT_WEIGHT

        # Reward safe seed counts (0, 2, 3, 5 or more)
        safe_positions = np.sum((board_sums[my_holes] == 0) |
                              (board_sums[my_holes] == 2) |
                              (board_sums[my_holes] == 3) |
                              (board_sums[my_holes] >= 5))
        score += safe_positions * self.SAFE_POSITION_WEIGHT

        # Extra penalty for consecutive vulnerable holes
        for i in range(len(my_holes) - 1):
            current_hole = my_holes[i]
            next_hole = my_holes[i + 1]
            if ((board_sums[current_hole] == 1 or board_sums[current_hole] == 4) and
                (board_sums[next_hole] == 1 or board_sums[next_hole] == 4)):
                score -= self.IMMEDIATE_THREAT_WEIGHT * 2

        return score

    def get_move(self, game_state):
        """
        Prioritize checking immediate defensive moves before deeper search.
        """
        # First, check for immediate defensive needs
        urgent_move = self.find_urgent_defensive_move(game_state)
        if urgent_move:
            return urgent_move, 0, 1

        # If no urgent moves, proceed with regular minimax search
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
                    is_root=True
                )

                if move is not None:
                    best_move_found = move
                    self.move_ordering[move] = eval_val

            except TimeoutError:
                break

            depth += 1

        compute_time = time.time() - start_time
        return (best_move_found, compute_time, depth - 1)

    def find_urgent_defensive_move(self, game_state):
        """
        Check for immediate defensive moves needed to prevent captures.
        """
        board_sums = np.sum(game_state.board, axis=1)
        my_holes = game_state.player_holes[game_state.current_player]

        # First priority: Fix any holes with 1 or 4 seeds
        vulnerable_holes = [h for h in my_holes if board_sums[h] in [1, 4]]
        for hole in vulnerable_holes:
            # Try to move seeds from this hole if possible
            for color in [0, 1]:  # Try both red and blue
                if game_state.board[hole, color] > 0:
                    clone = game_state.clone()
                    clone.play_move(hole, color)
                    # Check if move improves situation
                    new_sums = np.sum(clone.board, axis=1)
                    if not any((new_sums[my_holes] == 1) | (new_sums[my_holes] == 4)):
                        return (hole, color)

        # Second priority: Prevent opponent from setting up captures
        opp_holes = game_state.player_holes[3 - game_state.current_player]
        for opp_hole in opp_holes:
            next_hole = (opp_hole + 1) % 16
            if next_hole in my_holes and board_sums[next_hole] in [0, 1, 3]:
                # Find defensive move to protect this hole
                for hole in my_holes:
                    for color in [0, 1]:
                        if game_state.board[hole, color] > 0:
                            clone = game_state.clone()
                            clone.play_move(hole, color)
                            new_sums = np.sum(clone.board, axis=1)
                            if new_sums[next_hole] not in [1, 4]:
                                return (hole, color)

        return None

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        if time.time() - start_time >= max_time:
            raise TimeoutError()

        state_hash = hash((game_state.board.tobytes(), game_state.current_player))

        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    return stored_value, None

        if game_state.game_over() or depth == 0:
            return self.evaluate(game_state), None

        moves = game_state.get_valid_moves()
        if not moves:
            return self.evaluate(game_state), None

        # Sort moves based on previous evaluations
        moves = sorted(moves, key=lambda m: self.move_ordering.get(m, float('-inf')), reverse=True)

        best_move = None
        if maximizing_player:
            best_value = float('-inf')
            for move in moves:
                clone_state = game_state.clone()
                clone_state.play_move(*move)

                # Quick check for obviously bad moves
                if not is_root:
                    board_sums = np.sum(clone_state.board, axis=1)
                    my_holes = clone_state.player_holes[game_state.current_player]
                    if any((board_sums[my_holes] == 1) | (board_sums[my_holes] == 4)):
                        continue

                eval_val, _ = self.minimax(clone_state, depth - 1, alpha, beta, False, start_time, max_time)

                if eval_val > best_value:
                    best_value = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
        else:
            best_value = float('inf')
            for move in moves:
                clone_state = game_state.clone()
                clone_state.play_move(*move)
                eval_val, _ = self.minimax(clone_state, depth - 1, alpha, beta, True, start_time, max_time)

                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break

        if len(self.transposition_table) < self.MAX_TABLE_SIZE:
            self.transposition_table[state_hash] = (depth, best_value, best_move)

        return best_value, best_move

class MinimaxAgentBizarreLeMec:
    """
    This agent extends MinimaxAgent6_4 with:
      - Iterative Deepening
      - Principal Variation Search (PVS)
      - Late Move Reductions (LMR)
      - History Heuristic
      - Refined Killer Move logic
      - (Null Move Pruning is carried over from parent)
    """

    def __init__(self, max_time=2):
        self.max_time = max_time
        self.nodes_cut = 0
        self.transposition_table = {}
        # track best move from last completed depth
        self.principal_variation_move = None

        # A dictionary counting how often moves cause beta-cutoffs
        # for move ordering: { move: score }
        self.history_scores = defaultdict(int)

        # Killer moves: store up to 2 killers for each search depth
        self.killer_moves = defaultdict(lambda: [])

        # Some constants
        self.FLAG_EXACT = 0
        self.FLAG_LOWERBOUND = 1
        self.FLAG_UPPERBOUND = 2

        # For demonstration: we keep your evaluation weights from original
        self.SCORE_WEIGHT = 50
        self.CONTROL_WEIGHT = 30
        self.CAPTURE_WEIGHT = 20
        self.MOBILITY_WEIGHT = 15

        # The internal iterative deepening search limit
        self.max_depth = 30  # or any large enough depth

    def evaluate(self, game_state):
        """
        Same evaluate() from your original code:
        an example of a heuristic combining multiple factors.
        """
        my_index = game_state.current_player - 1
        opp_index = 1 - my_index

        score_diff = game_state.scores[my_index] - game_state.scores[opp_index]

        my_holes = game_state.player_holes[game_state.current_player]
        opp_holes = game_state.player_holes[3 - game_state.current_player]

        my_seeds = np.sum(game_state.board[my_holes])
        opp_seeds = np.sum(game_state.board[opp_holes])

        total_seeds = np.sum(game_state.board, axis=1)
        capture_positions = (total_seeds == 1) | (total_seeds == 4)
        capture_potential = (np.sum(capture_positions[my_holes]) -
                             np.sum(capture_positions[opp_holes])) * 2

        my_mobility = np.sum(game_state.board[my_holes] > 0)
        opp_mobility = np.sum(game_state.board[opp_holes] > 0)

        return (self.SCORE_WEIGHT * score_diff
                + self.CONTROL_WEIGHT * (my_seeds - opp_seeds)
                + self.CAPTURE_WEIGHT * capture_potential
                + self.MOBILITY_WEIGHT * (my_mobility - opp_mobility))

    def get_move(self, game_state):
        """
        Iterative Deepening + PVS driver.
        We iterate depth = 1..max_depth until time is up or we've completed a depth.
        We store the best move found in self.principal_variation_move after each iteration.
        """
        start_time = time.time()
        best_move_found = None
        depth_completed = 0

        # We'll do iterative deepening from depth=1 upwards
        for depth in range(1, self.max_depth + 1):
            if time.time() - start_time >= self.max_time:
                break

            alpha = -math.inf
            beta = math.inf

            try:
                value, move = self._pvs(game_state.clone(), depth, alpha, beta, True, start_time)
                if move is not None:
                    best_move_found = move
                    self.principal_variation_move = move
                depth_completed = depth
            except TimeoutError:
                # out of time in middle of search
                break

        total_time = time.time() - start_time
        return (best_move_found, total_time, depth_completed)

    def _pvs(self, game_state, depth, alpha, beta, maximizing_player, start_time):
        """
        Principal Variation Search (PVS) with:
         - Transposition table lookups
         - Move ordering (PV move first, killer moves, history heuristic, LMR)
         - Null move pruning
        If time runs out, we raise TimeoutError to break search early.
        Returns (best_value, best_move).
        """

        # Time check
        if time.time() - start_time >= self.max_time:
            raise TimeoutError()

        # Depth or game-over check
        if depth == 0 or game_state.game_over():
            return (self.evaluate(game_state), None)

        # Transposition table lookup
        state_hash = self._get_state_hash(game_state)
        alpha_original = alpha
        beta_original = beta

        if state_hash in self.transposition_table:
            stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                if stored_flag == self.FLAG_EXACT:
                    return (stored_value, stored_move)
                elif stored_flag == self.FLAG_LOWERBOUND:
                    alpha = max(alpha, stored_value)
                elif stored_flag == self.FLAG_UPPERBOUND:
                    beta = min(beta, stored_value)
                if alpha >= beta:
                    self.nodes_cut += 1
                    return (stored_value, stored_move)

        # *** Null-Move Pruning *** (optional, if we allow it)
        # Avoid if near terminal or too shallow
        if maximizing_player and depth >= 3 and not game_state.game_over():
            # pass the turn to see if the evaluation is big enough to prune
            clone_null = game_state.clone()
            # "null move" => skip current player's move
            clone_null.current_player = 3 - clone_null.current_player

            try:
                null_val, _ = self._pvs(clone_null, depth - 1 - 2, alpha, beta,
                                        not maximizing_player, start_time)
                if null_val >= beta:
                    return (beta, None)  # fails high => prune
            except TimeoutError:
                raise

        valid_moves = game_state.get_valid_moves()
        if not valid_moves:
            return (self.evaluate(game_state), None)

        # Order moves:
        #   1) principal_variation_move (if is root or we have a stored PV)
        #   2) killer moves
        #   3) history heuristic
        #   4) capture or likely capture moves (optional refinement)
        #   5) everything else
        ordered_moves = self._order_moves(game_state, valid_moves, depth, maximizing_player)

        best_value = float('-inf') if maximizing_player else float('inf')
        best_move = None
        first_move = True

        for move_index, move in enumerate(ordered_moves):
            # Late Move Reduction (LMR) heuristic
            # If we are beyond some minimal depth and we've tried many moves
            # (move_index > say 3 or 4), reduce search by 1
            reduction = 0
            if depth >= 3 and move_index > 3:
                reduction = 1

            # Principal Variation Search
            clone_state = game_state.clone()
            clone_state.play_move(*move)

            # If it's the first move in the node, we search fully
            if first_move:
                try:
                    val, _ = self._pvs(clone_state, depth - 1, alpha, beta,
                                       not maximizing_player, start_time)
                except TimeoutError:
                    raise
                first_move = False
            else:
                # For subsequent moves, search with a *narrow window* first
                try:
                    # reduced depth if LMR
                    val, _ = self._pvs(clone_state, depth - 1 - reduction,
                                       alpha, alpha + 1,
                                       not maximizing_player, start_time)
                except TimeoutError:
                    raise

                # If that “fails high,” we re-search fully
                if val > alpha and val < beta and maximizing_player:
                    try:
                        val, _ = self._pvs(clone_state, depth - 1,
                                           alpha, beta,
                                           not maximizing_player, start_time)
                    except TimeoutError:
                        raise
                elif val < beta and val > alpha and not maximizing_player:
                    # min fails low
                    try:
                        val, _ = self._pvs(clone_state, depth - 1,
                                           alpha, beta,
                                           not maximizing_player, start_time)
                    except TimeoutError:
                        raise

            # Update alpha/beta
            if maximizing_player:
                if val > best_value:
                    best_value = val
                    best_move = move
                alpha = max(alpha, best_value)
            else:
                if (best_value == float('inf')) or val < best_value:
                    best_value = val
                    best_move = move
                beta = min(beta, best_value)

            # killer / history updates if we get a cutoff
            if alpha >= beta:
                self.nodes_cut += 1
                # killer move
                kmoves = self.killer_moves[depth]
                if move not in kmoves:
                    kmoves.append(move)
                    if len(kmoves) > 2:
                        kmoves.pop(0)
                # history heuristic: increment
                self.history_scores[move] += depth * depth
                break
            else:
                # If no cutoff, partially reward moves that improved alpha or beta
                if maximizing_player and val > alpha_original:
                    self.history_scores[move] += depth
                if not maximizing_player and val < beta_original:
                    self.history_scores[move] += depth

        # Store in transposition table
        if len(self.transposition_table) < 1_000_000:  # limit size
            flag = self.FLAG_EXACT
            if best_value <= alpha_original:
                flag = self.FLAG_UPPERBOUND
            elif best_value >= beta_original:
                flag = self.FLAG_LOWERBOUND
            self.transposition_table[state_hash] = (depth, best_value, flag, best_move)

        return (best_value, best_move)

    def _get_state_hash(self, game_state):
        """
        A compact representation of the game state for the transposition table.
        """
        return hash((game_state.board.tobytes(),
                     game_state.scores.tobytes(),
                     game_state.current_player))

    def _order_moves(self, game_state, moves, depth, maximizing_player):
        """
        Move ordering combining:
          1) Principal Variation Move
          2) Killer Moves
          3) History Heuristic
          4) Others
        """
        # If we have a known PV move at root or from TT
        # we can place it in front
        if self.principal_variation_move in moves and depth == self.max_depth:
            # put PV move in front
            moves.remove(self.principal_variation_move)
            moves = [self.principal_variation_move] + moves

        # Next, gather killer moves for this depth
        # so we can put them next
        killer_list = self.killer_moves[depth]
        killers_in_moves = []
        rest = []
        for m in moves:
            if m in killer_list and m not in killers_in_moves:
                killers_in_moves.append(m)
            else:
                rest.append(m)

        # Sort the remaining moves by history scores (descending)
        rest.sort(key=lambda mv: self.history_scores.get(mv, 0), reverse=True)

        # Combine them
        combined = killers_in_moves + rest
        return combined
