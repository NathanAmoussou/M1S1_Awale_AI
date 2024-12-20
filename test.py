class AwaleGame:
    def __init__(self):
        # Initialisation du plateau : chaque trou contient [2 graines rouges, 2 graines bleues]
        # board[i] = [nb_graines_rouges, nb_graines_bleues]
        self.board = [[2, 2] for _ in range(16)]

        # Scores des deux joueurs (joueur 1, joueur 2)
        self.scores = [0, 0]

        # Joueur 1 contrôle les trous d'index pair (0,2,4,...)
        # Joueur 2 contrôle les trous d'index impair (1,3,5,...)
        self.player_holes = {
            1: [i for i in range(0, 16, 2)],  # Index pairs pour Joueur 1
            2: [i for i in range(1, 16, 2)]   # Index impairs pour Joueur 2
        }

        # Joueur courant (1 ou 2)
        self.current_player = 1

    def display_board(self):
        """Affiche le plateau sous forme de deux lignes de 8 trous avec numérotation (1-based)."""
        print("\nPlateau (dans l'ordre horaire) :")

        # Ligne supérieure (trous 1 à 8)
        print(" " + "     ".join(f"{i+1:2}" for i in range(16)))
        print(" " + " ".join(f"{hole}" for hole in self.board))

        # Afficher les scores
        print(f"\nScores: Joueur 1 = {self.scores[0]}, Joueur 2 = {self.scores[1]}")

    def is_valid_move(self, hole, color):
        """Vérifie si un mouvement est valide pour le joueur actuel."""
        # Le trou doit appartenir au joueur courant
        if hole not in self.player_holes[self.current_player]:
            return False
        # La couleur doit être 0 (rouge) ou 1 (bleu)
        if color not in [0, 1]:
            return False
        # Il doit y avoir au moins une graine de la couleur choisie
        if self.board[hole][color] == 0:
            return False
        return True

    def play_move(self, hole, color):
        """Effectue un mouvement : prend les graines de la couleur spécifiée dans le trou donné,
        les sème selon les règles, puis applique la capture."""
        if not self.is_valid_move(hole, color):
            raise ValueError("Mouvement invalide !")

        seeds_to_sow = self.board[hole][color]
        self.board[hole][color] = 0  # On retire les graines du trou de départ

        initial_hole = hole
        current_index = hole

        # Distribution des graines
        # Règles :
        # - Bleu (color=1) : sème dans tous les trous sauf le trou de départ.
        # - Rouge (color=0) : sème uniquement dans les trous de l'adversaire, en excluant le trou de départ.
        while seeds_to_sow > 0:
            current_index = (current_index + 1) % 16

            # Ne jamais semer dans le trou de départ
            if current_index == initial_hole:
                continue

            if color == 0:  # Rouge
                # Sème seulement dans les trous adverses
                if current_index in self.player_holes[self.current_player]:
                    continue
                # Trou valide pour semer des graines rouges
                self.board[current_index][color] += 1
                seeds_to_sow -= 1
            else:  # Bleu
                # Sème dans tous les trous sauf le trou de départ (déjà exclu)
                self.board[current_index][color] += 1
                seeds_to_sow -= 1

        # Appliquer les captures à partir du dernier trou semé
        self.apply_capture(current_index)

        # Changer de joueur
        self.current_player = 3 - self.current_player  # Passe de 1 à 2 ou de 2 à 1

    def apply_capture(self, start_hole):
        """Applique les règles de capture en remontant en arrière à partir du trou start_hole."""
        current_index = start_hole
        while True:
            total_seeds = sum(self.board[current_index])
            if total_seeds in [2, 3]:
                # Capturer les graines de ce trou
                self.scores[self.current_player - 1] += total_seeds
                self.board[current_index] = [0, 0]
                current_index = (current_index - 1) % 16
            else:
                break

    def game_over(self):
        """Vérifie si la partie est terminée selon les règles."""
        total_seeds = sum(sum(hole) for hole in self.board)
        # Si moins de 8 graines sur le plateau, la partie s'arrête
        if total_seeds < 8:
            return True
        # Si un joueur a 33 graines ou plus, la partie s'arrête
        if self.scores[0] >= 33 or self.scores[1] >= 33:
            return True
        # Si c'est un match nul à 32 graines chacun
        if self.scores[0] == 32 and self.scores[1] == 32:
            return True
        return False

    def get_winner(self):
        """Détermine le gagnant ou s'il y a égalité."""
        # Si égalité parfaite (32-32), on retourne "Égalité"
        if self.scores[0] == 32 and self.scores[1] == 32:
            return "Égalité"

        # Dans les autres cas, celui qui a le plus de graines l'emporte
        if self.scores[0] > self.scores[1]:
            return "Joueur 1"
        elif self.scores[1] > self.scores[0]:
            return "Joueur 2"
        else:
            # Si on arrive ici (et qu'on a moins de 8 graines sur le plateau),
            # on compare les scores. S'ils sont égaux, c'est une égalité.
            return "Égalité"



if __name__ == "__main__":
    game = AwaleGame()
    game.display_board()

    while not game.game_over():
        print(f"\nTour du Joueur {game.current_player}")
        hole = int(input("Choisissez un trou (1-16) : ")) - 1
        color = int(input("Choisissez une couleur (0 = Rouge, 1 = Bleu) : "))
        try:
            game.play_move(hole, color)
        except ValueError as e:
            print(e)
            continue
        game.display_board()

    print(f"\nPartie terminée ! Le gagnant est : {game.get_winner()}")
