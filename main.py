# main.py
from board_rules_interface import AwaleGame
from agents import HumanAgent, RandomAgent, GPTMinimaxAgentV2, ClaudeMinimaxAgentV1

if __name__ == "__main__":
    # Instantiate agents
    # Example configurations:

    # Human vs. Random
    # player1_agent = HumanAgent()
    # player2_agent = RandomAgent()

    # Minimax vs. Random
    # player1_agent = MinimaxAgent(max_time=2)
    # player2_agent = RandomAgent()

    # Human vs. Minimax
    # player1_agent = HumanAgent()
    # player2_agent = MinimaxAgent(max_time=2)

    # Minimax vs. Minimax
    # player1_agent = MinimaxAgent(max_time=2)
    # player2_agent = MinimaxAgent(max_time=2)

    # GPTV2 vs. ClaudeV1
    player1_agent = GPTMinimaxAgentV2(max_time=2)
    player2_agent = ClaudeMinimaxAgentV1(max_time=2)

    # Initialize the game with the selected agents
    game = AwaleGame(player1_agent=player1_agent, player2_agent=player2_agent)
    game.run_game()
