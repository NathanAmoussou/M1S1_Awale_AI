# Awale AI Competition Project

## Students Information

- **Student 1**: Nathan AMOUSSOU
- **Student 2**: Ismail El Amrani

## Overview

This project implements an AI to play a modified version of Awale using various agents, including Minimax with Alpha-Beta pruning and advanced optimizations. The program is designed to simulate games, log results, and analyze performance.

## How to Play

To play a game of Minimax against Random:
```bash
python3 main_np.py
```

## Customizing Game Parameters

To modify game settings, edit the `main_np.py` file in the `if __name__ == "__main__":` block. The following parameters can be customized:

- **Agent Classes**:
  - `HumanAgent`: Player-controlled agent.
  - `RandomAgent`: Random move generator.
  - `MinimaxAgent6`: Minimax agent with basic optimizations.
  - `MinimaxAgent6_4`: Advanced version of MinimaxAgent6 used during the competition.

  Set `agent1_class` for the first player and `agent2_class` for the second player.

- **Other Parameters**:
  - `num_games`: Number of games to play in a single run.
  - `max_time`: Maximum computation time per move for Minimax agents.

The game runs in the terminal, logs the results to `game_generation.log`, and exports the game data to a `game_datas` file.

## Key Features of MinimaxAgent6_4

The main agent used in the competition, `MinimaxAgent6_4`, is implemented in `agents_np.py`. It includes:
- Minimax algorithm with iterative deepening.
- Alpha-Beta pruning for efficient search.
- Transposition table to avoid redundant calculations.
- Move ordering for better pruning efficiency.
- Evaluation function to assess board states (see the report for more details).
- Principal Variation search for deeper analysis of likely moves.
- 1-ply simulation for enhanced decision-making.
- Vectorized operations using NumPy for improved performance compared to native Python data types.

## File Descriptions

- **`board_rules_interface_np.py`**: Handles the game board logic, including rendering and enforcing the modified rules. Uses NumPy for performance optimization.
- **`main_np.py`**: The main script for running games.
- **`data_export_np.py`**: Exports game data for analysis, including moves, scores, and winners for evaluating AI performance.
- **`profile_minimax_np.py`**: Contains profiling tools to identify performance bottlenecks in Minimax agents.

## Reflections on Performance

The agent's performance in the competition was underwhelming probably due to:
1. The inherent slowness of Python, making it unsuitable for complex simulations.
2. A strategic focus on implementing advanced features (e.g., Transposition Tables, Principal Variation) instead of refining the evaluation function or addressing critical game rules (e.g., starving protection).

### Lessons Learned
- Future iterations should prioritize languages like Java or C++ for speed.
- A robust Minimax implementation with a simple yet effective evaluation function and Alpha-Beta pruning would likely outperform complex but less efficient agents.

## Game Rules

The game is based on modified Awale rules:
- 16 holes (8 per player), numbered 1-16 in clockwise order.
- The first player controls odd-numbered holes, and the second player controls even-numbered holes.
- Seeds can be captured when the total in a hole reaches exactly two or three, regardless of color.
- The game ends when fewer than 8 seeds remain or when a player captures 33 or more seeds.
- Starving the opponent is allowed.

For a complete description of the rules, see `rules.txt`.

---
