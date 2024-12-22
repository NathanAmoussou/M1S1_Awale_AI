import math
import time
import random
import sys
from collections import defaultdict
import hashlib

class AwaleGame:
    def __init__(self, player1_type="human", player2_type="human"):
        """
        player1_type, player2_type peuvent être:
            - "human"
            - "ai_minimax"
            - "ai_random"
        """

        # Plateau initial : 16 trous, chacun [2 rouges, 2 bleues]
        self.board = [[2, 2] for _ in range(16)]

        # Scores : [score_joueur1, score_joueur2]
        self.scores = [0, 0]

        # Joueur 1 contrôle les indices pairs (0,2,4,...), Joueur 2 contrôle les indices impairs (1,3,5,...)
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

        # ----- Killer moves -----
        # On stocke 2 killer moves par profondeur (supposons profondeur max ~64)
        self.killer_moves = [[None, None] for _ in range(64)]

        # ----- Transposition Table -----
        # Dictionnaire : clé = hash d'état, valeur = (depth, score, nodeType, bestMove)
        # nodeType ∈ {"EXACT", "LOWER", "UPPER"}
        self.transpo_table = {}

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
        if hole not in self.player_holes[self.current_player]:
            return False
        if color not in [0, 1]:
            return False
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

        # Distribution des graines
        while seeds_to_sow > 0:
            current_index = (current_index + 1) % 16
            if current_index == initial_hole:
                continue

            if color == 0:  # Rouge => semer seulement dans les trous adverses
                if current_index in self.player_holes[self.current_player]:
                    continue
                self.board[current_index][0] += 1
                seeds_to_sow -= 1
            else:  # Bleu => semer partout sauf trou de départ
                self.board[current_index][1] += 1
                seeds_to_sow -= 1

        self.apply_capture(current_index)
        # Changement de joueur
        self.current_player = 3 - self.current_player

    def apply_capture(self, start_hole):
        """Applique la capture en remontant tant que trou = 2 ou 3 graines."""
        current_index = start_hole
        while True:
            total_seeds = sum(self.board[current_index])
            if total_seeds in [2, 3]:
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
        - Il reste moins de 8 graines sur le plateau,
        - Un des joueurs a >= 33 points,
        - Les deux joueurs ont 32 points chacun.
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
    # Clone, moves, hashing
    # ------------------------------------------------------------------
    def clone(self):
        new_game = AwaleGame(
            player1_type=self.player_types[1],
            player2_type=self.player_types[2]
        )
        new_game.board = [row[:] for row in self.board]
        new_game.scores = self.scores[:]
        new_game.current_player = self.current_player
        # killer_moves et TT ne sont pas copiés dans le clone
        return new_game

    def get_valid_moves(self):
        """Liste (hole, color) valides pour le joueur courant."""
        moves = []
        for hole in self.player_holes[self.current_player]:
            for color in [0, 1]:
                if self.is_valid_move(hole, color):
                    moves.append((hole, color))
        return moves

    def compute_hash(self):
        """
        Calcule un hash (ex : MD5) rapide de l'état.
        Pour un hashing plus robuste, on peut faire du Zobrist.
        """
        # Concaténer board + scores + current_player
        data = []
        data.append(str(self.current_player))
        data.append(str(self.scores[0]))
        data.append(str(self.scores[1]))
        for i in range(16):
            data.append(str(self.board[i][0]))
            data.append(str(self.board[i][1]))
        big_str = "_".join(data)
        return hashlib.md5(big_str.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Heuristique plus riche
    # ------------------------------------------------------------------
    def evaluate(self):
        """
        Heuristique améliorée :
        1) Différentiel de score (x10 pour l'accentuer),
        2) Bonus/malus en fonction de l'opportunité de capture imminente,
        etc.
        """
        if self.current_player == 1:
            base = self.scores[0] - self.scores[1]
        else:
            base = self.scores[1] - self.scores[0]

        base *= 10

        # Bonus : compter le nb de trous adverses = 1 ou 2 graines
        adv = 3 - self.current_player
        bonus = 0
        for hole in self.player_holes[adv]:
            s = sum(self.board[hole])
            if s in [1, 2]:
                bonus += 2

        return base + bonus

    # ------------------------------------------------------------------
    # Move Ordering
    # ------------------------------------------------------------------
    def order_moves(self, moves):
        """
        Évalue rapidement l'impact (ex: captures) de chaque move et trie
        (du plus prometteur au moins prometteur).
        """
        scored = []
        for m in moves:
            # One-step évaluation
            clone_state = self.clone()
            clone_state.play_move(m[0], m[1])
            if self.current_player == 1:
                delta = clone_state.scores[0] - self.scores[0]
            else:
                delta = clone_state.scores[1] - self.scores[1]
            scored.append((delta, m))

        # Tri décroissant
        scored.sort(key=lambda x: x[0], reverse=True)

        ordered = [x[1] for x in scored]
        return ordered

    # ------------------------------------------------------------------
    # Killer moves
    # ------------------------------------------------------------------
    def try_killer_moves(self, moves, depth):
        """
        Replace en tête de liste les killer moves si présents.
        On stocke 2 killer moves par profondeur.
        """
        if depth >= len(self.killer_moves):
            return moves  # Sécurité si la profondeur dépasse

        # Récupère les 2 killer moves potentiels
        km1, km2 = self.killer_moves[depth]

        # Transforme en liste pour manip plus facile
        moves_list = list(moves)

        for km in [km1, km2]:
            if km and km in moves_list:
                idx = moves_list.index(km)
                # Met en tête
                moves_list.insert(0, moves_list.pop(idx))
        return moves_list

    def add_killer_move(self, depth, move):
        """Insère 'move' en tête, décale l'autre."""
        if depth >= len(self.killer_moves):
            return
        km1, km2 = self.killer_moves[depth]
        if km1 != move:
            # Décale
            self.killer_moves[depth][1] = km1
            self.killer_moves[depth][0] = move

    # ------------------------------------------------------------------
    # Transposition Table
    # ------------------------------------------------------------------
    # On stocke : depth, score, nodeType ∈ {EXACT, LOWER, UPPER}, bestMove
    # nodeType nous permet de mettre à jour alpha/beta
    # ------------------------------------------------------------------
    def lookup_tt(self, state_hash, depth, alpha, beta):
        """
        Vérifie si on a une entrée TT. Renvoie (score, bestMove, alpha, beta, found).
        found = True si on a trouvé une TT valide
        """
        entry = self.transpo_table.get(state_hash, None)
        if entry is None:
            return None, None, alpha, beta, False

        stored_depth, stored_score, node_type, stored_best = entry
        if stored_depth >= depth:
            # On peut s'en servir
            if node_type == "EXACT":
                return stored_score, stored_best, alpha, beta, True
            elif node_type == "LOWER":
                if stored_score > alpha:
                    alpha = stored_score
            elif node_type == "UPPER":
                if stored_score < beta:
                    beta = stored_score
            if alpha >= beta:
                # cutoff
                return stored_score, stored_best, alpha, beta, True

        return None, None, alpha, beta, False

    def store_tt(self, state_hash, depth, score, node_type, best_move):
        self.transpo_table[state_hash] = (depth, score, node_type, best_move)

    # ------------------------------------------------------------------
    # Minimax avec TT, killer moves, move ordering
    # ------------------------------------------------------------------
    def minimax(self, depth, alpha, beta, maximizing_player, start_time, max_time):
        # 1) Vérification du temps
        if time.time() - start_time >= max_time:
            return self.evaluate(), None

        # 2) Condition d'arrêt
        if self.game_over() or depth == 0:
            return self.evaluate(), None

        # 3) TT check
        state_hash = self.compute_hash()
        stored_score, stored_move, alpha, beta, found = self.lookup_tt(state_hash, depth, alpha, beta)
        if found and stored_move is not None:
            # On peut renvoyer directement
            return stored_score, stored_move

        # 4) Générer + trier les moves
        moves = self.get_valid_moves()
        if not moves:
            return self.evaluate(), None

        moves = self.order_moves(moves)
        # On insère d'abord les killer moves
        moves = self.try_killer_moves(moves, depth)

        best_move = None

        if maximizing_player:
            best_score = -math.inf
            node_type = "EXACT"
            for mv in moves:
                if time.time() - start_time >= max_time:
                    break

                clone_state = self.clone()
                clone_state.play_move(mv[0], mv[1])
                child_score, _ = clone_state.minimax(
                    depth - 1, alpha, beta, False,
                    start_time, max_time
                )
                if child_score > best_score:
                    best_score = child_score
                    best_move = mv
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    # cutoff => killer move
                    self.add_killer_move(depth, mv)
                    node_type = "LOWER"
                    break

            # Stocker dans TT
            if best_score <= alpha:
                node_type = "UPPER"
            elif best_score >= beta:
                node_type = "LOWER"

            self.store_tt(state_hash, depth, best_score, node_type, best_move)
            return best_score, best_move

        else:
            best_score = math.inf
            node_type = "EXACT"
            for mv in moves:
                if time.time() - start_time >= max_time:
                    break

                clone_state = self.clone()
                clone_state.play_move(mv[0], mv[1])
                child_score, _ = clone_state.minimax(
                    depth - 1, alpha, beta, True,
                    start_time, max_time
                )
                if child_score < best_score:
                    best_score = child_score
                    best_move = mv
                beta = min(beta, best_score)
                if beta <= alpha:
                    self.add_killer_move(depth, mv)
                    node_type = "UPPER"
                    break

            if best_score <= alpha:
                node_type = "UPPER"
            elif best_score >= beta:
                node_type = "LOWER"

            self.store_tt(state_hash, depth, best_score, node_type, best_move)
            return best_score, best_move

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
                    True,  # maximizing
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
        """Retourne un coup au hasard parmi les coups possibles."""
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
            hole = int(input("Choisissez un trou (1-16) : ")) - 1
            color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
            return hole, color

        elif ptype == "ai_random":
            return self.best_move_random()

        elif ptype == "ai_minimax":
            # On peut mettre max_time plus haut pour plus de profondeur
            return self.best_move_minimax(max_time=2.0)

        else:
            return None


if __name__ == "__main__":
    # Exemple : J1 = ai_minimax, J2 = ai_random ou J2 = human
    player1_type = "ai_random"
    player2_type = "ai_minimax"

    game = AwaleGame(player1_type=player1_type, player2_type=player2_type)

    turn_counter = 0
    game.display_board()

    while not game.game_over():
        turn_counter += 1
        print(f"\nTour n°{turn_counter}, Joueur {game.current_player}")

        move = game.get_move_for_current_player()
        if move is None:
            break

        hole, color = move
        try:
            game.play_move(hole, color)
        except ValueError as e:
            print(e)
            turn_counter -= 1
            continue

        game.display_board()

    print(f"\nPartie terminée en {turn_counter} tours ! Le gagnant est : {game.get_winner()}")
