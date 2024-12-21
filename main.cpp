#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <algorithm>  // pour std::find, std::max, std::min
#include <limits>     // pour std::numeric_limits
#include <chrono>     // pour mesurer le temps
#include <random>     // pour la génération aléatoire
#include <stdexcept>  // pour std::runtime_error
#include <string>     // pour std::string


// Pour simplifier l'écriture
using namespace std;

// Pour distinguer les différents types de joueurs
enum class PlayerType {
    HUMAN,
    AI_MINIMAX,
    AI_RANDOM
};

// Fonction utilitaire pour la gestion du temps (en secondes).
double now_in_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

class AwaleGame {
public:
    // Constructeur : on passe le type de joueur (J1, J2) sous forme d'enum PlayerType
    AwaleGame(PlayerType p1_type, PlayerType p2_type)
    {
        // Plateau initial : 16 trous, chacun [2 rouges, 2 bleues]
        board.resize(16, std::vector<int>(2, 2));

        // Scores : [score_joueur1, score_joueur2]
        scores = {0, 0};

        // Joueur 1 contrôle les indices pairs (0,2,4,...), Joueur 2 les indices impairs (1,3,5,...)
        // On stocke ça dans un petit tableau.
        player_holes[1] = {0,2,4,6,8,10,12,14};
        player_holes[2] = {1,3,5,7,9,11,13,15};

        // Le joueur courant : 1 ou 2
        current_player = 1;

        // On enregistre les types de joueur
        player_types[1] = p1_type;
        player_types[2] = p2_type;
    }

    // Affichage console
    void display_board() {
        cout << "\nPlateau (dans l'ordre horaire) :" << endl;
        // Affichage des indices (1-based)
        cout << " ";
        for(int i=0; i<16; i++){
            cout << "    " << (i+1);
        }
        cout << endl << " ";
        for(int i=0; i<16; i++){
            cout << "[" << board[i][0] << "," << board[i][1] << "] ";
        }
        cout << endl;
        cout << "\nScores: Joueur 1 = " << scores[0]
             << ", Joueur 2 = " << scores[1] << endl;
    }

    // Vérifie si (hole, color) est un coup valide pour le joueur courant
    bool is_valid_move(int hole, int color) {
        // hole doit être dans la liste des trous contrôlés par current_player
        auto &holes = player_holes[current_player];
        if(std::find(holes.begin(), holes.end(), hole) == holes.end()){
            return false;
        }
        // color doit être 0 ou 1
        if(color != 0 && color != 1){
            return false;
        }
        // Il doit y avoir au moins 1 graine de cette couleur
        if(board[hole][color] == 0){
            return false;
        }
        return true;
    }

    // Joue le coup (hole, color)
    void play_move(int hole, int color){
        if(!is_valid_move(hole, color)){
            throw std::runtime_error("Mouvement invalide !");
        }
        int seeds_to_sow = board[hole][color];
        board[hole][color] = 0; // On vide ce trou pour la couleur donnée

        int initial_hole = hole;
        int current_index = hole;

        // Distribution des graines
        while(seeds_to_sow > 0){
            current_index = (current_index + 1) % 16;
            // Ne jamais semer dans le trou de départ
            if(current_index == initial_hole){
                continue;
            }
            // Règles rouges : semer seulement dans les trous de l'adversaire
            if(color == 0){ // rouge
                if(std::find(player_holes[current_player].begin(),
                             player_holes[current_player].end(),
                             current_index) != player_holes[current_player].end()){
                    // c'est un trou du joueur courant => on skip
                    continue;
                }
                // sinon on sème
                board[current_index][0] += 1;
                seeds_to_sow--;
            }
            else { // bleu
                board[current_index][1] += 1;
                seeds_to_sow--;
            }
        }

        // Application de la capture
        apply_capture(current_index);

        // On change de joueur : 1 -> 2, 2 -> 1
        current_player = 3 - current_player;
    }

    // Applique la capture en remontant
    void apply_capture(int start_hole){
        int current_index = start_hole;
        while(true){
            int total_seeds = board[current_index][0] + board[current_index][1];
            if(total_seeds == 2 || total_seeds == 3){
                // capture
                scores[current_player - 1] += total_seeds;
                board[current_index][0] = 0;
                board[current_index][1] = 0;
                current_index = (current_index - 1 + 16) % 16;
            }
            else {
                break;
            }
        }
    }

    // Conditions de fin de partie
    bool game_over(){
        int total_seeds = 0;
        for(int i=0; i<16; i++){
            total_seeds += (board[i][0] + board[i][1]);
        }
        if(total_seeds < 8){
            return true;
        }
        if(scores[0] >= 33 || scores[1] >= 33){
            return true;
        }
        if(scores[0] == 32 && scores[1] == 32){
            return true;
        }
        return false;
    }

    // Renvoie le gagnant ou "Égalité"
    std::string get_winner(){
        if(scores[0] > scores[1]){
            return "Joueur 1";
        }
        else if(scores[1] > scores[0]){
            return "Joueur 2";
        }
        else {
            return "Égalité";
        }
    }

    // Retourne une copie (clone) de l'état du jeu
    AwaleGame clone() {
        AwaleGame new_game(player_types[1], player_types[2]);
        new_game.board = board;
        new_game.scores = scores;
        new_game.current_player = current_player;
        return new_game;
    }

    // Liste tous les coups possibles (hole, color)
    std::vector<std::pair<int,int>> get_valid_moves(){
        std::vector<std::pair<int,int>> moves;
        for(auto hole : player_holes[current_player]){
            for(int color=0; color<2; color++){
                if(is_valid_move(hole, color)){
                    moves.push_back({hole,color});
                }
            }
        }
        return moves;
    }

    // Fonction d'évaluation basique : difference de scores
    int evaluate(){
        if(current_player == 1){
            return scores[0] - scores[1];
        } else {
            return scores[1] - scores[0];
        }
    }

    // Minimax + alpha-beta + coupure temporelle
    std::pair<int, std::pair<int,int> > minimax(
            int depth, int alpha, int beta, bool maximizing_player,
            double start_time, double max_time
    ){
        // Vérification du temps
        if(now_in_seconds() - start_time >= max_time){
            return { evaluate(), {-1,-1} };
        }
        if(game_over() || depth == 0){
            return { evaluate(), {-1,-1} };
        }
        auto moves = get_valid_moves();
        if(moves.empty()){
            return { evaluate(), {-1,-1} };
        }

        std::pair<int,int> best_move = {-1,-1};
        if(maximizing_player){
            int max_eval = std::numeric_limits<int>::min();
            for(auto &m : moves){
                // Vérif du temps
                if(now_in_seconds() - start_time >= max_time){
                    break;
                }
                // On clone
                AwaleGame cloned = clone();
                cloned.play_move(m.first, m.second);
                auto [eval_val, _] = cloned.minimax(
                        depth - 1, alpha, beta, false,
                        start_time, max_time
                );
                if(eval_val > max_eval){
                    max_eval = eval_val;
                    best_move = m;
                }
                alpha = std::max(alpha, eval_val);
                if(beta <= alpha){
                    break;
                }
            }
            return {max_eval, best_move};
        }
        else {
            int min_eval = std::numeric_limits<int>::max();
            for(auto &m : moves){
                if(now_in_seconds() - start_time >= max_time){
                    break;
                }
                AwaleGame cloned = clone();
                cloned.play_move(m.first, m.second);
                auto [eval_val, _] = cloned.minimax(
                        depth - 1, alpha, beta, true,
                        start_time, max_time
                );
                if(eval_val < min_eval){
                    min_eval = eval_val;
                    best_move = m;
                }
                beta = std::min(beta, eval_val);
                if(beta <= alpha){
                    break;
                }
            }
            return {min_eval, best_move};
        }
    }

    // Recherche itérative (augmentation de la profondeur) dans la limite max_time (en secondes)
    std::pair<int,int> best_move_minimax(double max_time=2.0){
        double start_time = now_in_seconds();
        int depth = 1;
        std::pair<int,int> best_move_found = {-1,-1};

        while(true){
            double elapsed = now_in_seconds() - start_time;
            if(elapsed >= max_time) break;

            try {
                auto [eval_val, mv] = minimax(
                    depth, std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::max(),
                    true,  // maximizing_player
                    start_time, max_time
                );
                if(mv.first != -1 && mv.second != -1){
                    best_move_found = mv;
                }
            } catch(...) {
                break;
            }
            depth++;
        }
        double total_time = now_in_seconds() - start_time;
        cout << "Temps de calcul (Minimax) : " << total_time
             << "s, profondeur atteinte : " << (depth - 1) << endl;
        return best_move_found;
    }

    // IA aléatoire
    std::pair<int,int> best_move_random(){
        auto moves = get_valid_moves();
        if(moves.empty()){
            return {-1,-1};
        }
        // Choix aléatoire
        static std::mt19937 rng( std::random_device{}() );
        std::uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
        int idx = dist(rng);
        cout << "IA aléatoire a choisi un coup au hasard." << endl;
        return moves[idx];
    }

    // Récupère le coup pour le joueur courant
    std::pair<int,int> get_move_for_current_player(){
        PlayerType ptype = player_types[current_player];
        if(ptype == PlayerType::HUMAN){
            // On lit depuis stdin
            cout << "Choisissez un trou (1-16) : ";
            int hole;
            cin >> hole;
            hole -= 1;
            cout << "Choisissez une couleur (0 = Rouge, 1 = Bleu) : ";
            int color;
            cin >> color;
            return {hole, color};
        }
        else if(ptype == PlayerType::AI_RANDOM){
            return best_move_random();
        }
        else if(ptype == PlayerType::AI_MINIMAX){
            return best_move_minimax(10.0); // ex: 10s
        }
        else {
            return {-1,-1};
        }
    }

    int get_current_player() const {
        return current_player;
    }

private:
    std::vector<std::vector<int>> board; // board[i] = [red, blue] pour le trou i
    std::array<int,2> scores;  // scores[0] -> J1, scores[1] -> J2
    std::map<int, std::vector<int>> player_holes;
    int current_player;  // 1 ou 2
    std::map<int, PlayerType> player_types;
};

// ----------------------------------------------------------------------
// Pour convertir un nom de type ("human", "ai_minimax", "ai_random")
// en enum PlayerType
PlayerType parse_type(const std::string &s){
    if(s == "human") return PlayerType::HUMAN;
    if(s == "ai_minimax") return PlayerType::AI_MINIMAX;
    if(s == "ai_random") return PlayerType::AI_RANDOM;
    // Par défaut
    return PlayerType::HUMAN;
}

int main(){
    // Exemple : Joueur 1 = "ai_random", Joueur 2 = "ai_minimax"
    std::string p1_str = "ai_random";
    std::string p2_str = "ai_minimax";

    PlayerType p1_type = parse_type(p1_str);
    PlayerType p2_type = parse_type(p2_str);

    AwaleGame game(p1_type, p2_type);

    int turn_counter = 0;
    game.display_board();

    while(!game.game_over()){
        turn_counter++;
        cout << "\nTour n°" << turn_counter
             << ", Joueur " << game.get_current_player() << endl;

        auto move = game.get_move_for_current_player();
        if(move.first == -1 && move.second == -1){
            // Aucun coup possible
            break;
        }
        try {
            game.play_move(move.first, move.second);
        }
        catch(const std::exception &e){
            cout << e.what() << endl;
            // Ne pas changer de joueur si coup invalide
            turn_counter--;
            continue;
        }
        game.display_board();
    }

    cout << "\nPartie terminée en " << turn_counter << " tours ! "
         << "Le gagnant est : " << game.get_winner() << endl;

    return 0;
}
