#agents_np.py
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
                hole = int(input("Choisissez un trou (1-16) : ")) - 1
                color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
                if game_state.is_valid_move(hole, color):
                    return (hole, color), None, None
                else:
                    print("Coup invalide. Veuillez réessayer.")
            except ValueError:
                print("Entrée invalide. Veuillez entrer des nombres valides.")

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
        print(f"IA aléatoire a choisi le coup: Hole {move[0]+1} Color {'R' if move[1]==0 else 'B'}")
        return (move, None, None)

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

class MinimaxAgent6(Agent):
    def __init__(self, max_time=2):
        self.max_time = max_time
        self.nodes_cut = 0
        self.transposition_table = {}
        self.move_ordering = {}
        self.MAX_TABLE_SIZE = 1000000

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

    # claude get move
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

class MinimaxAgent6_1_S(MinimaxAgent6_1):
    def __init__(self, max_time=2):
        super().__init__(max_time)
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics counters"""
        # Basic statistics
        self.nodes_cut = 0  # From alpha-beta pruning
        self.total_nodes_visited = 0

        # Additional pruning statistics
        self.null_move_cuts = 0
        self.futility_cuts = 0
        self.transposition_cuts = 0
        self.late_move_cuts = 0

        # Move statistics
        self.capturing_moves_found = 0
        self.non_capturing_moves_found = 0

    def get_move(self, game_state):
        # Reset stats at the start of each move
        self.reset_stats()

        # Get the move using parent class implementation
        move_result = super().get_move(game_state)

        # Calculate pruning percentages
        total_cuts = (self.nodes_cut + self.null_move_cuts +
                     self.futility_cuts + self.transposition_cuts +
                     self.late_move_cuts)

        pruning_percentage = (total_cuts / self.total_nodes_visited * 100) if self.total_nodes_visited > 0 else 0

        # Print comprehensive statistics
        print(f"\nPruning Statistics:")
        print(f"Total nodes visited: {self.total_nodes_visited}")
        print(f"Pruning Breakdown:")
        print(f"  - Alpha-beta cuts: {self.nodes_cut}")
        print(f"  - Null-move cuts: {self.null_move_cuts}")
        print(f"  - Futility cuts: {self.futility_cuts}")
        print(f"  - Transposition cuts: {self.transposition_cuts}")
        print(f"  - Late-move cuts: {self.late_move_cuts}")
        print(f"Total nodes pruned: {total_cuts}")
        print(f"Overall pruning percentage: {pruning_percentage:.2f}%")

        print(f"\nMove Analysis:")
        total_moves = self.capturing_moves_found + self.non_capturing_moves_found
        if total_moves > 0:
            capture_percentage = (self.capturing_moves_found / total_moves * 100)
            print(f"Capturing moves: {self.capturing_moves_found} ({capture_percentage:.1f}%)")
            print(f"Non-capturing moves: {self.non_capturing_moves_found}")

        return move_result

    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        # Increment total nodes counter
        self.total_nodes_visited += 1

        # Transposition table lookup
        state_hash = self._get_state_hash(game_state)
        if not is_root:
            if state_hash in self.transposition_table:
                stored_depth, stored_value, stored_flag, stored_move = self.transposition_table[state_hash]
                if stored_depth >= depth:
                    if (stored_flag == self.FLAG_EXACT or
                        (stored_flag == self.FLAG_LOWERBOUND and stored_value >= beta) or
                        (stored_flag == self.FLAG_UPPERBOUND and stored_value <= alpha)):
                        self.transposition_cuts += 1
                        return stored_value, stored_move

        # Futility pruning statistics tracking
        if depth <= self.futility_depth_limit:
            static_eval = self.evaluate(game_state)
            if static_eval + self.futility_margin <= alpha:
                valid_moves = game_state.get_valid_moves()
                capturing_moves = []
                non_capturing_moves = []
                for mv in valid_moves:
                    if self._is_likely_capture(game_state, mv):
                        capturing_moves.append(mv)
                        self.capturing_moves_found += 1
                    else:
                        non_capturing_moves.append(mv)
                        self.non_capturing_moves_found += 1

                if not capturing_moves:
                    self.futility_cuts += 1
                    return static_eval, None

        # Null move pruning statistics tracking
        if depth >= (self.null_move_reduction + 1) and self._can_do_null_move(game_state):
            null_state = game_state.clone()
            null_state.current_player = 3 - null_state.current_player

            eval_null, _ = super().minimax(
                null_state,
                depth - 1 - self.null_move_reduction,
                beta - 1,
                beta,
                not maximizing_player,
                start_time,
                max_time,
                is_root=False
            )

            if eval_null >= beta:
                self.null_move_cuts += 1
                return beta, None

        # Use parent class implementation for the rest of minimax
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
