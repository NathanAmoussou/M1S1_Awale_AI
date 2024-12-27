import math
import time
import random

class AwaleGame:
    def __init__(self, player1_type="human", player2_type="human"):
        # Plateau initial : 16 trous, chacun [2 rouges, 2 bleues]
        self.board = [[2, 2] for _ in range(16)]
        # Scores : [score_joueur1, score_joueur2]
        self.scores = [0, 0]
        # Joueur 1 contrôle les indices pairs (0,2,4,...), Joueur 2 les indices impairs (1,3,5,...)
        self.player_holes = {
            1: [i for i in range(0, 16, 2)],
            2: [i for i in range(1, 16, 2)]
        }
        # Le joueur en cours (1 ou 2)
        self.current_player = 1
        # Types de joueurs : "human", "ai_minimax" ou "ai_random"
        self.player_types = {
            1: player1_type,
            2: player2_type
        }

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------
    def display_board(self):
        """Affiche le plateau, puis les scores."""
        print("\nPlateau (dans l'ordre horaire) :")
        print(" " + "     ".join(f"{i+1:2}" for i in range(16)))
        print(" " + " ".join(f"{hole}" for hole in self.board))
        print(f"\nScores: Joueur 1 = {self.scores[0]}, Joueur 2 = {self.scores[1]}")

    # ------------------------------------------------------------------
    # Mouvements
    # ------------------------------------------------------------------
    def is_valid_move(self, hole, color):
        """Vérifie si (hole, color) est un mouvement valide pour le joueur courant."""
        # hole doit être dans les trous contrôlés par current_player
        if hole not in self.player_holes[self.current_player]:
            return False
        # color doit être 0 (rouge) ou 1 (bleu)
        if color not in [0, 1]:
            return False
        # Le trou doit contenir au moins 1 graine de la couleur choisie
        if self.board[hole][color] == 0:
            return False
        return True

    def play_move(self, hole, color):
        """Joue le coup (hole, color) et passe la main à l'autre joueur."""
        if not self.is_valid_move(hole, color):
            raise ValueError("Mouvement invalide !")

        seeds_to_sow = self.board[hole][color]
        self.board[hole][color] = 0  # Retire les graines du trou de départ

        initial_hole = hole
        current_index = hole

        # Distribution des graines selon les règles
        while seeds_to_sow > 0:
            current_index = (current_index + 1) % 16

            # Ne pas semer dans le trou de départ
            if current_index == initial_hole:
                continue

            if color == 0:  # Rouge
                # Semer seulement dans les trous adverses
                if current_index in self.player_holes[self.current_player]:
                    continue
                self.board[current_index][color] += 1
                seeds_to_sow -= 1
            else:  # Bleu
                # Semer dans tous les trous sauf le départ
                self.board[current_index][color] += 1
                seeds_to_sow -= 1

        # Application de la capture
        self.apply_capture(current_index)

        # Changement de joueur
        self.current_player = 3 - self.current_player

    def apply_capture(self, start_hole):
        """
        Applique la capture en rafale en remontant tant que le trou précédent
        a un total de 2 ou 3 graines.
        """
        current_index = start_hole
        while True:
            total_seeds = sum(self.board[current_index])
            if total_seeds in [2, 3]:
                # Le joueur qui vient de jouer capture
                self.scores[self.current_player - 1] += total_seeds
                self.board[current_index] = [0, 0]
                current_index = (current_index - 1) % 16
            else:
                break

    # ------------------------------------------------------------------
    # Fin de partie
    # ------------------------------------------------------------------
    def game_over(self):
        """
        La partie se termine si :
        - Il reste moins de 8 graines sur le plateau.
        - Un des joueurs a >= 33 points.
        - Les deux joueurs ont 32 points chacun (égalité).
        """
        total_seeds = sum(sum(hole) for hole in self.board)
        if total_seeds < 8:
            return True
        if self.scores[0] >= 33 or self.scores[1] >= 33:
            return True
        if self.scores[0] == 32 and self.scores[1] == 32:
            return True
        return False

    def get_winner(self):
        """Retourne 'Joueur 1', 'Joueur 2' ou 'Égalité'."""
        if self.scores[0] > self.scores[1]:
            return "Joueur 1"
        elif self.scores[1] > self.scores[0]:
            return "Joueur 2"
        else:
            return "Égalité"

    # ------------------------------------------------------------------
    # IA et évaluation
    # ------------------------------------------------------------------
    def clone(self):
        """Retourne une copie indépendante de l'état de la partie."""
        new_game = AwaleGame(
            player1_type=self.player_types[1],
            player2_type=self.player_types[2]
        )
        new_game.board = [h[:] for h in self.board]
        new_game.scores = self.scores[:]
        new_game.current_player = self.current_player
        return new_game

    def get_valid_moves(self):
        """Liste tous les coups possibles (hole, color) pour le joueur courant."""
        moves = []
        for hole in self.player_holes[self.current_player]:
            for color in [0, 1]:
                if self.is_valid_move(hole, color):
                    moves.append((hole, color))
        return moves

    def evaluate(self):
        """
        Fonction d'évaluation basique: différence de score
        vue du joueur courant.
        """
        if self.current_player == 1:
            return self.scores[0] - self.scores[1]
        else:
            return self.scores[1] - self.scores[0]

    def minimax(self, depth, alpha, beta, maximizing_player, start_time, max_time):
        """
        Minimax (avec coupure alpha-beta) et coupure temporelle.
        - depth: profondeur de recherche restante
        - alpha, beta: bornes de coupe
        - maximizing_player: True si on cherche à maximiser, False si on minimise
        - start_time, max_time: pour la coupure de temps
        """
        # Vérification du temps
        if time.time() - start_time >= max_time:
            return self.evaluate(), None

        if self.game_over() or depth == 0:
            return self.evaluate(), None

        moves = self.get_valid_moves()
        if not moves:
            return self.evaluate(), None

        best_move = None

        if maximizing_player:
            max_eval = -math.inf
            for move in moves:
                if time.time() - start_time >= max_time:
                    break

                clone_state = self.clone()
                clone_state.play_move(*move)
                eval_val, _ = clone_state.minimax(
                    depth - 1, alpha, beta,
                    False,  # on passe au joueur minimisant
                    start_time, max_time
                )
                if eval_val > max_eval:
                    max_eval = eval_val
                    best_move = move
                alpha = max(alpha, eval_val)
                if beta <= alpha:
                    break
            return max_eval, best_move
        else:
            min_eval = math.inf
            for move in moves:
                if time.time() - start_time >= max_time:
                    break

                clone_state = self.clone()
                clone_state.play_move(*move)
                eval_val, _ = clone_state.minimax(
                    depth - 1, alpha, beta,
                    True,  # on passe au joueur maximisant
                    start_time, max_time
                )
                if eval_val < min_eval:
                    min_eval = eval_val
                    best_move = move
                beta = min(beta, eval_val)
                if beta <= alpha:
                    break
            return min_eval, best_move

    def best_move_minimax(self, max_time=2):
        """
        Recherche itérative : on augmente la profondeur tant que le temps n'est pas écoulé.
        """
        start_time = time.time()
        depth = 1
        best_move_found = None

        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_time:
                break

            try:
                eval_val, move = self.minimax(
                    depth, -math.inf, math.inf,
                    True,  # Le joueur courant est supposé maximiser
                    start_time, max_time
                )
                if move is not None:
                    best_move_found = move
            except Exception:
                break

            depth += 1

        total_time = time.time() - start_time
        print(f"Temps de calcul (Minimax) : {total_time:.2f}s, profondeur atteinte : {depth - 1}")
        return best_move_found

    def best_move_random(self):
        """Retourne un coup au hasard parmi les coups valides."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return None
        move = random.choice(valid_moves)
        print("IA aléatoire a choisi un coup au hasard.")
        return move

    # ------------------------------------------------------------------
    # Choix du coup selon le type de joueur
    # ------------------------------------------------------------------
    def get_move_for_current_player(self):
        """
        Renvoie le coup (hole, color) pour le joueur courant,
        en fonction de son type (human, ai_minimax, ai_random).
        """
        ptype = self.player_types[self.current_player]

        if ptype == "human":
            # On demande à l'utilisateur un trou et une couleur
            hole = int(input("Choisissez un trou (1-16) : ")) - 1
            color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
            return hole, color

        elif ptype == "ai_random":
            return self.best_move_random()

        elif ptype == "ai_minimax":
            return self.best_move_minimax(max_time=2)

        else:
            # Par défaut, on ne sait pas -> aucun coup
            return None

# -------------------------------------------------------------
# EXEMPLE D'UTILISATION
# -------------------------------------------------------------
#
# Paramètres possibles pour chaque joueur:
#   "human", "ai_minimax", ou "ai_random"
#
# Exemple : Joueur 1 = Minimax, Joueur 2 = Random
player1_type = "ai_random"
player2_type = "ai_random"

game = AwaleGame(player1_type=player1_type, player2_type=player2_type)

turn_counter = 0
game.display_board()

while not game.game_over():
    turn_counter += 1
    print(f"\nTour n°{turn_counter}, Joueur {game.current_player}")

    move = game.get_move_for_current_player()

    # Si c'est un coup humain invalide, on retente
    if move is None:
        # Aucun coup possible (ou pas de coup renvoyé) => partie finie
        break

    # Le move vient soit de l'humain, soit de l'IA
    # Vérifions qu'il est valide
    hole, color = move
    try:
        game.play_move(hole, color)
    except ValueError as e:
        print(e)
        # On ne change pas de joueur si move invalide => annuler l'incrément
        turn_counter -= 1
        continue

    game.display_board()

print(f"\nPartie terminée en {turn_counter} tours ! Le gagnant est : {game.get_winner()}")
