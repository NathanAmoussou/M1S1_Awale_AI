# main.py
from board_rules_interface import AwaleGame
from agents import HumanAgent, RandomAgent, MinimaxAgent

if __name__ == "__main__":
    # Instantiate agents
    # Example configurations:

    # Human vs. Random
    # player1_agent = HumanAgent()
    # player2_agent = RandomAgent()

    # Minimax vs. Random
    player1_agent = MinimaxAgent(max_time=2)
    player2_agent = RandomAgent()

    # Human vs. Minimax
    # player1_agent = HumanAgent()
    # player2_agent = MinimaxAgent(max_time=2)

    # Minimax vs. Minimax
    # player1_agent = MinimaxAgent(max_time=2)
    # player2_agent = MinimaxAgent(max_time=2)

    # Initialize the game with the selected agents
    game = AwaleGame(player1_agent=player1_agent, player2_agent=player2_agent)
    game.run_game()
