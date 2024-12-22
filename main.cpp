#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <limits>
#include <chrono>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>  // Pour la Transposition Table
#include <functional>     // std::hash

using namespace std;

enum class PlayerType {
    HUMAN,
    AI_MINIMAX,
    AI_RANDOM
};

// Pour la TT : type de nœud (ex. EXACT, LOWER_BOUND, UPPER_BOUND)
enum class NodeTypeTT {
    EXACT,
    LOWER_BOUND,
    UPPER_BOUND
};

// Structure pour stocker les infos dans la TT
struct TTEntry {
    int depth;             // profondeur à laquelle ce score a été calculé
    int score;             // score évalué
    NodeTypeTT nodeType;   // EXACT, LOWER_BOUND ou UPPER_BOUND
    pair<int,int> bestMove; // Meilleur coup connu à cette config
};

// ----------------------------------------------------
// Gestion du temps
// ----------------------------------------------------
double now_in_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// ----------------------------------------------------
// CLASSE AwaleGame
// ----------------------------------------------------
class AwaleGame {
public:
    // Constructeur
    AwaleGame(PlayerType p1_type, PlayerType p2_type)
    {
        board.resize(16, std::vector<int>(2, 2)); // 16 trous : [2R, 2B] chacun
        scores = {0, 0};
        player_holes[1] = {0,2,4,6,8,10,12,14};
        player_holes[2] = {1,3,5,7,9,11,13,15};

        current_player = 1;

        player_types[1] = p1_type;
        player_types[2] = p2_type;

        // On initialise le vecteur de "killer moves" : on en garde 2 par profondeur (exemple)
        killer_moves.resize(64, vector<pair<int,int>>(2, {-1,-1}));
    }

    //--------------------------------------------------
    // Fonctions principales (play, apply_capture, etc.)
    //--------------------------------------------------
    bool is_valid_move(int hole, int color) {
        auto &holes = player_holes[current_player];
        if(std::find(holes.begin(), holes.end(), hole) == holes.end()){
            return false;
        }
        if(color != 0 && color != 1) {
            return false;
        }
        if(board[hole][color] == 0){
            return false;
        }
        return true;
    }

    void play_move(int hole, int color){
        if(!is_valid_move(hole, color)){
            throw std::runtime_error("Mouvement invalide !");
        }
        int seeds_to_sow = board[hole][color];
        board[hole][color] = 0;

        int initial_hole = hole;
        int current_index = hole;

        while(seeds_to_sow > 0){
            current_index = (current_index + 1) % 16;
            if(current_index == initial_hole) {
                continue;
            }
            if(color == 0){ // rouge => semer seulement dans trous adverses
                if(std::find(player_holes[current_player].begin(),
                             player_holes[current_player].end(),
                             current_index) != player_holes[current_player].end()){
                    continue;
                }
                board[current_index][0] += 1;
                seeds_to_sow--;
            }
            else { // bleu
                board[current_index][1] += 1;
                seeds_to_sow--;
            }
        }

        apply_capture(current_index);
        current_player = 3 - current_player;
    }

    void apply_capture(int start_hole){
        int current_index = start_hole;
        while(true){
            int total_seeds = board[current_index][0] + board[current_index][1];
            if(total_seeds == 2 || total_seeds == 3){
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

    string get_winner(){
        if(scores[0] > scores[1]) return "Joueur 1";
        if(scores[1] > scores[0]) return "Joueur 2";
        return "Égalité";
    }

    //--------------------------------------------------
    // Clonage et moves
    //--------------------------------------------------
    AwaleGame clone() const {
        AwaleGame new_game(player_types.at(1), player_types.at(2));
        new_game.board = board;
        new_game.scores = scores;
        new_game.current_player = current_player;
        // Copier la TT n'est pas indispensable pour un clone, on la gère globalement
        return new_game;
    }

    vector<pair<int,int>> get_valid_moves(){
        vector<pair<int,int>> moves;
        for(auto hole : player_holes[current_player]){
            for(int color=0; color<2; color++){
                if(is_valid_move(hole, color)){
                    moves.push_back({hole, color});
                }
            }
        }
        return moves;
    }

    //--------------------------------------------------
    // Meilleure heuristique
    //--------------------------------------------------
    int evaluate() {
        // Heuristique :
        // 1) Différentiel de score
        // 2) + bonus si j'ai des trous adverses proches de 2 ou 3
        // 3) - malus si adversaire a des captures imminentes
        int base_score = 0;
        if(current_player == 1) {
            base_score = scores[0] - scores[1];
        } else {
            base_score = scores[1] - scores[0];
        }

        // Bonus : compter le nb de trous adverses qui sont à 1 ou 2 graines
        // car 1 graine en plus => 2 ou 3 => capture possible
        // (c'est une idée simple, à affiner)
        int bonus = 0;
        int adv = 3 - current_player;
        for(int hole : player_holes[adv]) {
            int sum_hole = board[hole][0] + board[hole][1];
            if(sum_hole == 1 || sum_hole == 2) {
                bonus += 2;
            }
        }
        return base_score*10 + bonus;
    }

    //--------------------------------------------------
    // Hashing pour la Transposition Table
    //--------------------------------------------------
    size_t compute_hash() const {
        // Exemple (très basique) : On combine la valeur de chaque trou + current_player
        // Pour un hashing plus robuste, on ferait un Zobrist par ex.
        // Ici on se contente d'un "accumulate" naïf
        size_t h = 0;
        std::hash<long long> hl;
        long long acc = current_player;
        acc = (acc << 4) ^ (scores[0]*131542391);
        acc = (acc << 4) ^ (scores[1]*265443578);

        for(int i=0; i<16; i++){
            acc ^= (board[i][0] + 17LL*board[i][1] + 31LL*i);
            acc *= 101LL;
        }
        h = hl(acc);
        return h;
    }

    //--------------------------------------------------
    // Move Ordering (Rapide)
    //--------------------------------------------------
    // On peut attribuer un "score" rapide aux moves (un mini one-step evaluation)
    // puis trier du plus grand au plus petit
    void order_moves(vector<pair<int,int>> &moves) {
        // On évalue vite fait "combien ce move capture immédiatement" ou "simulate short"
        // pour classer
        vector<pair<int, pair<int,int>>> scored_moves;
        for(auto &mv : moves) {
            int hole = mv.first;
            int color = mv.second;
            // Heuristique : plus c'est susceptible de capturer, plus c'est "haut".
            // On fait un mini apply ?
            AwaleGame temp = clone();
            temp.play_move(hole,color);
            int deltaScore = 0;
            if(current_player == 1){
                deltaScore = temp.scores[0] - scores[0];
            } else {
                deltaScore = temp.scores[1] - scores[1];
            }
            scored_moves.push_back({deltaScore, mv});
        }
        // On trie par deltaScore décroissant
        sort(scored_moves.begin(), scored_moves.end(),
             [](auto &a, auto &b){return a.first > b.first;});

        moves.clear();
        for(auto &sm : scored_moves) {
            moves.push_back(sm.second);
        }
    }

    //--------------------------------------------------
    // Killer Moves
    //--------------------------------------------------
    // killer_moves[depth][0..1] => on garde 2 killer moves par profondeur
    // Si on obtient un cutoff (beta < alpha), on stocke le move dans killer_moves
    // On essaie en priorité ces moves la prochaine fois
    // (Ici, on se limite à 64 comme profondeur max)
    // -----------------------------------------------
    void try_killer_moves(vector<pair<int,int>> &moves, int depth) {
        // On place en tête de liste les killer moves, si présents
        for(int k=0; k<2; k++){
            auto km = killer_moves[depth][k];
            if(km.first == -1) continue;
            // Cherche si km est dans moves
            auto it = find(moves.begin(), moves.end(), km);
            if(it != moves.end()){
                // On swap pour le mettre en tête
                iter_swap(it, moves.begin());
            }
        }
    }

    void add_killer_move(int depth, pair<int,int> move) {
        // On stocke ce move dans killer_moves[depth][0] ou [1]
        // s'il n'y est pas déjà
        if(killer_moves[depth][0] != move) {
            // on décale
            killer_moves[depth][1] = killer_moves[depth][0];
            killer_moves[depth][0] = move;
        }
    }

    //--------------------------------------------------
    // Transposition Table
    //--------------------------------------------------
    // On la définit en static, ou global, ou dans la classe (ici dans la classe).
    // Key = size_t (hash), Value = TTEntry (score, bestMove, depth, nodeType)
    //--------------------------------------------------
    unordered_map<size_t, TTEntry> transpoTable;

    // -------------------------------------------------
    // minimax (avec TT, killer moves, move ordering)
    // -------------------------------------------------
    pair<int, pair<int,int>> minimax(
        int depth, int alpha, int beta, bool maximizing_player,
        double start_time, double max_time
    ){
        // 1) Gestion du temps
        if(now_in_seconds() - start_time >= max_time){
            return { evaluate(), {-1,-1} };
        }
        // 2) Condition d'arrêt
        if(game_over() || depth == 0){
            return { evaluate(), {-1,-1} };
        }

        // 3) Transposition Table
        size_t stHash = compute_hash();
        auto itTT = transpoTable.find(stHash);
        if(itTT != transpoTable.end()) {
            // On a une entrée dans la TT
            TTEntry &tte = itTT->second;
            if(tte.depth >= depth) {
                // On peut potentiellement utiliser cette info
                if(tte.nodeType == NodeTypeTT::EXACT) {
                    return { tte.score, tte.bestMove };
                }
                else if(tte.nodeType == NodeTypeTT::LOWER_BOUND && tte.score > alpha){
                    alpha = tte.score;
                }
                else if(tte.nodeType == NodeTypeTT::UPPER_BOUND && tte.score < beta){
                    beta = tte.score;
                }
                if(alpha >= beta) {
                    // cutoff
                    return { tte.score, tte.bestMove };
                }
            }
        }

        // 4) Générer les moves
        auto moves = get_valid_moves();
        if(moves.empty()) {
            // pas de moves => évalue
            return { evaluate(), {-1,-1} };
        }

        // => Move Ordering
        order_moves(moves);

        // => Tenter killer moves en tête
        try_killer_moves(moves, depth);

        pair<int,int> best_move = {-1,-1};
        int best_score = maximizing_player
            ? std::numeric_limits<int>::min()
            : std::numeric_limits<int>::max();

        // on conserve pour la TT
        NodeTypeTT finalNodeType = NodeTypeTT::EXACT;

        // 5) Parcours
        for(auto &m : moves){
            if(now_in_seconds() - start_time >= max_time){
                break;
            }
            AwaleGame cloned = clone();
            cloned.play_move(m.first, m.second);

            auto [child_score, _] = cloned.minimax(
                depth - 1, alpha, beta, !maximizing_player,
                start_time, max_time
            );

            if(maximizing_player){
                if(child_score > best_score){
                    best_score = child_score;
                    best_move = m;
                }
                if(best_score > alpha){
                    alpha = best_score;
                }
                if(beta <= alpha){
                    // cutoff
                    // => On ajoute ce move en killer
                    add_killer_move(depth, m);
                    finalNodeType = NodeTypeTT::LOWER_BOUND;
                    break;
                }
            }
            else {
                // Minimizing
                if(child_score < best_score){
                    best_score = child_score;
                    best_move = m;
                }
                if(best_score < beta){
                    beta = best_score;
                }
                if(beta <= alpha){
                    // cutoff
                    add_killer_move(depth, m);
                    finalNodeType = NodeTypeTT::UPPER_BOUND;
                    break;
                }
            }
        }

        // 6) Stocker dans la TT
        TTEntry entry;
        entry.depth = depth;
        entry.score = best_score;
        entry.bestMove = best_move;
        if(best_score <= alpha) {
            // on a un upper bound
            entry.nodeType = NodeTypeTT::UPPER_BOUND;
        }
        else if(best_score >= beta) {
            // lower bound
            entry.nodeType = NodeTypeTT::LOWER_BOUND;
        }
        else {
            entry.nodeType = NodeTypeTT::EXACT;
        }
        transpoTable[stHash] = entry;

        return { best_score, best_move };
    }

    // recherche itérative
    pair<int,int> best_move_minimax(double max_time=2.0){
        double start_time = now_in_seconds();
        int depth = 1;
        pair<int,int> best_move_found = {-1,-1};

        while(true){
            double elapsed = now_in_seconds() - start_time;
            if(elapsed >= max_time) break;

            try {
                auto [eval_val, mv] = minimax(
                    depth,
                    std::numeric_limits<int>::min(),
                    std::numeric_limits<int>::max(),
                    true,
                    start_time,
                    max_time
                );
                if(mv.first != -1){
                    best_move_found = mv;
                }
            }
            catch(...) {
                break;
            }
            depth++;
        }
        double total_time = now_in_seconds() - start_time;
        cout << "Temps de calcul (Minimax) : " << total_time
             << "s, profondeur atteinte : " << (depth - 1) << endl;
        return best_move_found;
    }

    // IA random
    pair<int,int> best_move_random(){
        auto moves = get_valid_moves();
        if(moves.empty()){
            return {-1,-1};
        }
        static mt19937 rng( random_device{}() );
        uniform_int_distribution<int> dist(0, (int)moves.size()-1);
        int idx = dist(rng);
        return moves[idx];
    }

    //--------------------------------------------------
    // Accès
    //--------------------------------------------------
    PlayerType get_player_type(int p) const {
        return player_types.at(p);
    }

    int get_current_player() const {
        return current_player;
    }

    // Affichage
    void display_board() {
        cout << "\nPlateau (dans l'ordre horaire) :" << endl;
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

private:
    // Données du plateau
    vector<vector<int>> board; // board[i] = [red,blue]
    array<int,2> scores;
    map<int, vector<int>> player_holes;
    int current_player;
    map<int, PlayerType> player_types;

    // Killer moves (on prend un maximum de 64 de profondeur)
    vector<vector<pair<int,int>>> killer_moves;

//public:
    // Table de transposition
//    unordered_map<size_t, TTEntry> transpoTable;
};

// ----------------------------------------------------
// Convertir un string en PlayerType
// ----------------------------------------------------
PlayerType parse_type(const string &s){
    if(s == "human") return PlayerType::HUMAN;
    if(s == "ai_minimax") return PlayerType::AI_MINIMAX;
    if(s == "ai_random") return PlayerType::AI_RANDOM;
    return PlayerType::HUMAN;
}

// ----------------------------------------------------
// MAIN
// ----------------------------------------------------
int main(){
    // Exemple : Joueur 1 = "ai_minimax", Joueur 2 = "ai_minimax"
    // afin de tester la performance (les 2 IA)
    // ou Joueur 1 = "ai_random", Joueur 2 = "ai_minimax", etc.
    string p1_str = "ai_random";
    string p2_str = "ai_minimax";

    PlayerType p1_type = parse_type(p1_str);
    PlayerType p2_type = parse_type(p2_str);

    AwaleGame game(p1_type, p2_type);

    int turn_counter = 0;
    game.display_board();

    while(!game.game_over()){
        turn_counter++;
        cout << "\nTour n°" << turn_counter
             << ", Joueur " << game.get_current_player() << endl;

        pair<int,int> move;
        // On récupère le type de joueur
        PlayerType ptype = game.get_player_type(game.get_current_player());
        if(ptype == PlayerType::HUMAN){
            // Input console
            cout << "Trou (1-16) : ";
            int hole; cin >> hole; hole--;
            cout << "Couleur (0=rouge,1=bleu) : ";
            int col; cin >> col;
            move = {hole,col};
        }
        else if(ptype == PlayerType::AI_RANDOM){
            move = game.best_move_random();
            if(move.first == -1) {
                // plus de moves
                break;
            }
            cout << "IA random joue trou " << (move.first+1)
                 << ", couleur " << move.second << endl;
        }
        else { // AI_MINIMAX
            move = game.best_move_minimax(2.0); // ex. 3 secondes
            if(move.first == -1) {
                break;
            }
            cout << "IA Minimax joue trou " << (move.first+1)
                 << ", couleur " << move.second << endl;
        }

        try {
            game.play_move(move.first, move.second);
        }
        catch(exception &e){
            cout << e.what() << endl;
            turn_counter--;
            continue;
        }
        game.display_board();
    }

    cout << "\nPartie terminée en " << turn_counter << " tours ! "
         << "Le gagnant est : " << game.get_winner() << endl;

    return 0;
}
