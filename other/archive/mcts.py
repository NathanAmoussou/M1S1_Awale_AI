import math
import time
import random
import torch
from torch import Tensor
from typing import List, Tuple, Dict, Optional

from agents import Agent  # Adjust import based on your project structure

##################################################
# Helper Functions for Batched Rollout with PyTorch
##################################################

@torch.jit.script
def get_valid_moves_mask(states: Tensor, current_players: Tensor) -> Tensor:
    """
    Given a batch of states and the current player for each state,
    return a boolean mask of shape (batch_size, 16, 2) indicating
    which (hole, color) moves are valid.

    states[b, h, c] = number of seeds in hole h of color c
    current_players[b] = 0 or 1 (0-based)

    Player1 controls holes [0, 2, 4, ..., 14]
    Player2 controls holes [1, 3, 5, ..., 15]
    """
    batch_size = states.size(0)
    valid_moves = torch.zeros((batch_size, 16, 2), dtype=torch.bool, device=states.device)

    player1_holes = torch.arange(0, 16, 2, device=states.device, dtype=torch.long)
    player2_holes = torch.arange(1, 16, 2, device=states.device, dtype=torch.long)

    for b in range(batch_size):
        if current_players[b] == 0:
            valid_moves[b, player1_holes, :] = (states[b, player1_holes, :] > 0)
        else:
            valid_moves[b, player2_holes, :] = (states[b, player2_holes, :] > 0)

    return valid_moves

@torch.jit.script
def apply_moves_parallel(states: Tensor,
                         scores: Tensor,
                         current_players: Tensor,
                         moves: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Apply a batch of moves to the batch of states.

    moves[b] = (hole, color) for each batch element.
    Seeds are removed from the chosen hole/color, then sown.
    No capture is performed here; that is done after sowing.

    Returns:
        new_states: updated states after sowing seeds
        new_scores: same as input for now (captures happen later)
        new_current_players: toggled players (0->1 or 1->0)
        end_holes: the final hole index where the last seed was sown
    """
    batch_size = states.size(0)
    new_states = states.clone()
    new_scores = scores.clone()
    end_holes = torch.zeros(batch_size, dtype=torch.long, device=states.device)

    for b in range(batch_size):
        hole = moves[b, 0].long()
        color = moves[b, 1].long()
        seeds = new_states[b, hole, color].item()
        new_states[b, hole, color] = 0

        current_idx = hole
        sown = 0
        while sown < seeds:
            current_idx = (current_idx + 1) % 16
            # Do not sow into the original hole if you circle back
            if current_idx == hole:
                continue
            new_states[b, current_idx, color] += 1
            sown += 1

        end_holes[b] = current_idx

    # Toggle current player (0 -> 1, 1 -> 0)
    new_current_players = 1 - current_players
    return new_states, new_scores, new_current_players, end_holes

@torch.jit.script
def apply_captures_parallel(states: Tensor,
                            scores: Tensor,
                            current_players: Tensor,
                            end_holes: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Apply captures starting from the last sown hole (end_holes).
    Move backward while the sum of seeds in the hole is 2 or 3.
    If so, the current player captures them, and we keep going
    backward until it's not 2 or 3.
    """
    batch_size = states.size(0)
    new_states = states.clone()
    new_scores = scores.clone()

    for b in range(batch_size):
        idx = end_holes[b].long()
        player_idx = current_players[b].long()  # 0 or 1
        for _ in range(16):  # max 16 captures in a row
            total_seeds = new_states[b, idx].sum()
            if total_seeds == 2 or total_seeds == 3:
                new_scores[b, player_idx] += total_seeds
                new_states[b, idx, 0] = 0
                new_states[b, idx, 1] = 0
                idx = (idx - 1) % 16
            else:
                break

    return new_states, new_scores

##################################################
# A Node in the MCTS Tree
##################################################

class ParallelMCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        """
        game_state: A clone of the AwaleGame or your environment
        parent:     Parent ParallelMCTSNode
        move:       The (hole, color) that led to this state from the parent
        """
        self.game_state = game_state
        self.parent = parent
        self.move = move

        # Children: Dict[(hole, color), ParallelMCTSNode]
        self.children: Dict[Tuple[int, int], "ParallelMCTSNode"] = {}

        # For MCTS stats
        self.visits = 0
        self.total_value = 0.0

        # Unexpanded moves at this node
        self.untried_moves = game_state.get_valid_moves()

    def is_terminal(self) -> bool:
        """ Return True if this state is terminal (no moves or game over). """
        return self.game_state.game_over() or len(self.untried_moves) == 0

    def is_fully_expanded(self) -> bool:
        """ Return True if there are no moves left to expand. """
        return len(self.untried_moves) == 0

    def get_mean_value(self) -> float:
        """ Average value of this node. """
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb_score(self, exploration_constant: float) -> float:
        """
        Calculate the UCB1 score for selection.
        If this node has no visits, return +inf to force exploration.
        """
        if self.visits == 0:
            return float('inf')
        return (self.get_mean_value() +
                exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits))

##################################################
# The Parallel MCTS Agent
##################################################

class ParallelMCTSAgent(Agent):
    def __init__(self,
                 num_simulations: int = 800,
                 batch_size: int = 16,
                 exploration_constant: float = 1.41,
                 max_time: float = 2.0,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        num_simulations:     Maximum number of total expansions (rollouts)
        batch_size:          How many expansions we try to batch together
        exploration_constant:UCB exploration constant
        max_time:            Soft time limit in seconds (we’ll stop near this)
        device:              'cpu' or 'cuda', used for the parallel rollout
        """
        super().__init__()
        self.num_simulations = num_simulations
        self.batch_size = batch_size
        self.exploration_constant = exploration_constant
        self.max_time = max_time
        self.device = device

    def get_move(self, game_state) -> Tuple[Tuple[int, int], float, int]:
        """
        Perform MCTS (in parallel) from the given game state, return the best move.
        Returns: ((hole, color), total_time_spent, simulations_done)
        """
        start_time = time.time()
        root = ParallelMCTSNode(game_state.clone())
        simulations_done = 0

        # If no moves at all, return None
        if not root.untried_moves:
            return (None, None), (time.time() - start_time), 0

        while (time.time() - start_time) < self.max_time and simulations_done < self.num_simulations:
            # 1) Gather a batch of leaf nodes to expand
            batch_leaves = []
            for _ in range(self.batch_size):
                if (time.time() - start_time) >= self.max_time or simulations_done >= self.num_simulations:
                    break
                leaf = self._select(root)
                if leaf.is_terminal():
                    # No expansions from a terminal node
                    batch_leaves.append(leaf)
                else:
                    # Expand exactly one child from this leaf
                    move = leaf.untried_moves.pop()
                    new_state = leaf.game_state.clone()
                    new_state.play_move(*move)  # (hole, color)
                    child = ParallelMCTSNode(new_state, parent=leaf, move=move)
                    leaf.children[move] = child
                    batch_leaves.append(child)

                simulations_done += 1
                if simulations_done >= self.num_simulations:
                    break

            # 2) Roll out all these leaves in parallel (unless they’re terminal)
            if batch_leaves:
                self._parallel_rollout_and_backprop(batch_leaves)

        # Choose the child with the most visits (or best average) as the best move
        best_child = max(
            root.children.values(),
            key=lambda c: c.visits
        )
        best_move = best_child.move
        total_time_spent = time.time() - start_time
        return best_move, total_time_spent, simulations_done

    def _select(self, node: ParallelMCTSNode) -> ParallelMCTSNode:
        """
        Descend the tree using UCB until we reach a leaf or a node that isn't fully expanded.
        """
        current = node
        while not current.is_terminal() and current.is_fully_expanded():
            # Choose child with max UCB
            current = max(
                current.children.values(),
                key=lambda c: c.ucb_score(self.exploration_constant)
            )
        return current

    def _parallel_rollout_and_backprop(self, leaves: List[ParallelMCTSNode]):
        """
        Take a list of leaf nodes, do a batched random playout to get final
        results (from each leaf's perspective), then backprop those results.
        """
        # Separate terminal leaves vs non-terminal
        terminal_leaves = [leaf for leaf in leaves if leaf.is_terminal()]
        non_terminal_leaves = [leaf for leaf in leaves if not leaf.is_terminal()]

        # For terminal leaves, the final value can be computed by a quick evaluation
        for leaf in terminal_leaves:
            # Use the AwaleGame's score difference as a final value or something similar
            # Or a simpler approach: if leaf is terminal, no moves => Evaluate difference
            scores = leaf.game_state.scores
            current_p = leaf.game_state.current_player
            # For a typical score difference perspective:
            value = self._final_value(scores, current_p)
            self._backpropagate(leaf, value)

        if not non_terminal_leaves:
            return

        # 1) Build torch tensors for the parallel states
        states, scores, current_players = self._make_tensors(non_terminal_leaves)

        # 2) Run random rollout until we hit a terminal condition or max steps
        rollout_values = self._batched_random_rollout(states, scores, current_players)

        # 3) Backprop each result to the leaf
        for leaf, val in zip(non_terminal_leaves, rollout_values):
            self._backpropagate(leaf, float(val))

    def _final_value(self, scores: List[int], current_player: int) -> float:
        """
        Compute a final scalar from the perspective of the node’s current player.
        For example, if current_player=1 is leading, return a positive value.
        If tie, return 0. Otherwise negative.

        AwaleGame: current_player is 1 or 2.
        We convert that to 0-based: current_player - 1 in the rollout code.
        Here, do something like:
          value = (score[current_p-1] - score[1 - (current_p-1)])
        """
        # Convert 1-based to 0-based index
        cp0 = current_player - 1  # 0 or 1
        opp0 = 1 - cp0
        return scores[cp0] - scores[opp0]

    def _make_tensors(self, leaves: List[ParallelMCTSNode]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Pack the states of all leaf nodes into PyTorch tensors for parallel rollout.
        """
        batch_size = len(leaves)
        # shape: (batch_size, 16, 2)
        states = torch.zeros((batch_size, 16, 2), dtype=torch.float32, device=self.device)
        # shape: (batch_size, 2)
        scores = torch.zeros((batch_size, 2), dtype=torch.float32, device=self.device)
        # shape: (batch_size,)  # 0-based current player
        current_players = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        for i, leaf in enumerate(leaves):
            for h in range(16):
                states[i, h, 0] = float(leaf.game_state.board[h][0])
                states[i, h, 1] = float(leaf.game_state.board[h][1])
            scores[i, 0] = float(leaf.game_state.scores[0])
            scores[i, 1] = float(leaf.game_state.scores[1])
            # Convert AwaleGame.current_player (1 or 2) -> 0 or 1
            current_players[i] = leaf.game_state.current_player - 1

        return states, scores, current_players

    def _batched_random_rollout(self,
                                states: Tensor,
                                scores: Tensor,
                                current_players: Tensor,
                                max_steps: int = 50) -> Tensor:
        """
        A simple random playout in parallel for all states in the batch.
        Returns a 1D float tensor of final values from each state's perspective.
        The perspective is that of the *original* current player in `current_players`.
        So we store that in a separate tensor and do sign flipping if the current player changes.
        """
        batch_size = states.size(0)
        device = states.device

        # Keep track of the "original" current player so we know how to interpret final scores
        original_players = current_players.clone()

        # We create a mask for which states are still active (not terminal)
        active_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_steps):
            # Mark any obviously terminal states
            # For Awale: let's say if sum of seeds < 8 or if a player has >= 33 seeds or turn 150, etc.
            # But we only have partial info here. We'll do a simpler approach:
            # if no valid moves or a big lead in scores we treat as terminal.
            valid_moves_mask = get_valid_moves_mask(states, current_players)
            # If no moves for all holes and colors, it's terminal
            can_move = valid_moves_mask.view(batch_size, -1).any(dim=1)  # (b,)
            # Deactivate those that can't move
            active_mask = active_mask & can_move

            if not active_mask.any():
                # All states are terminal or stuck
                break

            # Construct a random move for each active state
            moves = torch.zeros((batch_size, 2), dtype=torch.long, device=device)
            for b in range(batch_size):
                if active_mask[b]:
                    valid_indices = torch.nonzero(valid_moves_mask[b])
                    if len(valid_indices) == 0:
                        # No moves -> skip
                        active_mask[b] = False
                    else:
                        idx = random.randrange(len(valid_indices))
                        moves[b] = valid_indices[idx]

            # Apply moves
            states, scores, current_players, end_holes = apply_moves_parallel(
                states, scores, current_players, moves
            )

            # Apply captures
            states, scores = apply_captures_parallel(states, scores, current_players, end_holes)

        # Compute final value for each state from the perspective of the original player
        # final_value = scores[original_player] - scores[1 - original_player]
        # But original_player is 0 or 1; if original_player[b]==0 => value = scores[b,0]-scores[b,1]
        # else => scores[b,1]-scores[b,0]
        final_values = torch.zeros(batch_size, dtype=torch.float32, device=device)
        for b in range(batch_size):
            op = original_players[b].item()  # 0 or 1
            s0 = scores[b, 0].item()
            s1 = scores[b, 1].item()
            if op == 0:
                final_values[b] = s0 - s1
            else:
                final_values[b] = s1 - s0

        return final_values

    def _backpropagate(self, leaf: ParallelMCTSNode, result: float):
        """
        Climb back up to the root, increment visits, and add the result to total_value.
        `result` is from the perspective of the leaf’s current_player at entry,
        but we typically store the same value up the chain.
        A common approach in Awale is to keep consistent perspective from the root's point of view,
        but you can adapt as needed. For simplicity, we just store the same result up the tree.
        """
        node = leaf
        while node is not None:
            node.visits += 1
            node.total_value += result
            node = node.parent
