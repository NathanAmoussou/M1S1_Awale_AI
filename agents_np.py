import random
import math
import time
import numpy as np

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
        return (move, None, None)

class MinimaxAgent6(Agent):
    def __init__(self, max_time=2):
        self.max_time = max_time
        self.nodes_cut = 0
        self.transposition_table = {}
        self.move_ordering = {}
        self.MAX_TABLE_SIZE = 1000000

        # Pre-compute evaluation weights
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

        LATE_MOVE_PRUNE_THRESHOLD = 5
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
        Otherwise => 0 or some base.
        """
        # We'll do a quick simulation:
        clone = game_state.clone()
        clone.play_move(*move)

        # Check if any captures happened. If apply_move/capture logic
        # sets aside seeds, we can see if the clone's "scores" changed or if
        # the board seeds got zeroed out in certain holes. We'll do a naive approach:
        # Compare total seeds on the board before vs. after

        before_seeds = np.sum(game_state.board)
        after_seeds = np.sum(clone.board)
        # If there's a difference => captures happened
        if after_seeds < before_seeds:
            # The difference might be big, meaning multiple captures
            # We can weigh that difference in the move ordering
            captured_amount = before_seeds - after_seeds
            return 10_000 + captured_amount
        else:
            return 0
