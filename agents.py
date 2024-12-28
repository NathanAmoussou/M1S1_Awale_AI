# agents.py
import random
import math
import time

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
            tuple: A tuple (hole, color) representing the move.
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
                    return (hole, color)
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
            return None
        move = random.choice(valid_moves)
        print(f"IA aléatoire a choisi le coup: Hole {move[0]+1} Color {'R' if move[1]==0 else 'B'}")
        return move

class GPTMinimaxAgentV2(Agent):
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
        print(f"Temps de calcul (Minimax) : {total_time:.2f}s, profondeur atteinte : {depth - 1}")
        return best_move_found

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

class ClaudeMinimaxAgentV1(Agent):
    def __init__(self, max_time=2):
        """
        Initialize the Minimax agent.

        Parameters:
            max_time (float): Maximum time allowed for move computation in seconds.
        """
        self.max_time = max_time
        self.nodes_cut = 0
        # flag can be: 'EXACT', 'LOWERBOUND', or 'UPPERBOUND'
        self.transposition_table = {}
        # Move ordering table stores historical success of moves
        self.move_ordering = {}

        # Constants for transposition table
        self.EXACT = 0
        self.LOWERBOUND = 1
        self.UPPERBOUND = 2

        # Maximum size for transposition table (adjust based on available memory)
        self.MAX_TABLE_SIZE = 1000000

    # claude get move
    def get_move(self, game_state):
        start_time = time.time()
        depth = 1
        best_move_found = None

        # Initialize move ordering table
        valid_moves = game_state.get_valid_moves()
        for move in valid_moves:
            if move not in self.move_ordering:
                self.move_ordering[move] = 0

        # Iterative deepening with move ordering
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= self.max_time * 0.95:  # Leave some buffer time
                break

            try:
                eval_val, move = self.minimax(
                    game_state.clone(),
                    depth,
                    -math.inf,
                    math.inf,
                    True,
                    start_time,
                    self.max_time,
                    is_root=True
                )

                if move is not None:
                    best_move_found = move
                    # Update move ordering scores
                    self.move_ordering[move] = max(eval_val, self.move_ordering.get(move, 0))

            except TimeoutError:
                break

            depth += 1

        print(f"Reached depth: {depth-1}")
        return best_move_found


    # claude minimax
    def minimax(self, game_state, depth, alpha, beta, maximizing_player, start_time, max_time, is_root=False):
        # Time check
        if time.time() - start_time >= max_time * 0.95:
            raise TimeoutError()

        # Check transposition table
        state_hash = self._get_state_hash(game_state)
        if not is_root and state_hash in self.transposition_table:
            stored_depth, stored_value, stored_move = self.transposition_table[state_hash]
            if stored_depth >= depth:
                return stored_value, stored_move

        if game_state.game_over() or depth == 0:
            eval_val = game_state.claude_evaluate_V1()
            return eval_val, None

        moves = self._order_moves(game_state.get_valid_moves())
        if not moves:
            return game_state.claude_evaluate_V1(), None

        best_move = None
        best_value = -math.inf if maximizing_player else math.inf

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
                alpha = max(alpha, eval_val)
            else:
                if eval_val < best_value:
                    best_value = eval_val
                    best_move = move
                beta = min(beta, eval_val)

            if beta <= alpha:
                self.nodes_cut += 1
                break

        # Store in transposition table
        self.transposition_table[state_hash] = (depth, best_value, best_move)
        return best_value, best_move

    def _get_state_hash(self, game_state):
        """Create a hash of the current game state."""
        board_tuple = tuple(tuple(hole) for hole in game_state.board)
        return (board_tuple, tuple(game_state.scores), game_state.current_player)

    def _order_moves(self, moves):
        """Order moves based on previously successful moves."""
        return sorted(moves, key=lambda m: self.move_ordering.get(m, 0), reverse=True)
