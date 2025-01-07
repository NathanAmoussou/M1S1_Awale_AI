#include <iostream>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_map>
#include <chrono>
#include <cmath>
#include <random>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------
//                     Base Agent
// ---------------------------------------------------------
class AwaleGame; // Forward declaration to let Agent reference it.

class Agent {
public:
    virtual ~Agent() = default;

    // Return a tuple of:
    //   (std::pair<int,int> move, double compute_time, int depth)
    //   or ( { -1, -1 }, 0, 0 ) if no move is possible.
    // The move is (hole, color).
    //
    // If you do not care about compute_time/depth, you can set them to 0.
    virtual std::tuple<std::pair<int,int>, double, int> get_move(AwaleGame& game_state) = 0;
};

// ---------------------------------------------------------
//                        AwaleGame
// ---------------------------------------------------------
class AwaleGame {
public:
    // 16 holes, each with 2 possible "colors" of seeds: 0=Red, 1=Blue
    std::array<std::array<int8_t, 2>, 16> board{};
    // scores[0] => Player1's captured seeds, scores[1] => Player2's captured seeds
    std::array<int16_t, 2> scores{};
    // Mapping of player -> which holes belong to them
    // player 1 => even indices [0,2,4,...,14], player 2 => odd indices [1,3,5,...,15]
    std::array<std::vector<int8_t>, 3> player_holes;

    // Current player can be 1 or 2
    int current_player = 1;
    // Agents
    Agent* player1_agent = nullptr;
    Agent* player2_agent = nullptr;

    // Just for debugging or logging
    int turn_number = 0;

    AwaleGame(Agent* p1, Agent* p2) {
        player1_agent = p1;
        player2_agent = p2;
        // Initialize board with 2 seeds in each color for each of the 16 holes
        for (int i = 0; i < 16; i++) {
            board[i][0] = 2;  // red seeds
            board[i][1] = 2;  // blue seeds
        }
        scores[0] = 0;
        scores[1] = 0;

        // Precompute which holes belong to which player
        // In Python code: player1 has (0,2,4,...,14) => even indices
        //                 player2 has (1,3,5,...,15) => odd indices
        for (int i = 0; i < 16; i++) {
            if (i % 2 == 0) {
                player_holes[1].push_back(i);
            } else {
                player_holes[2].push_back(i);
            }
        }
    }

    // Copy constructor for "clone()"
    AwaleGame(const AwaleGame& other) {
        board = other.board;
        scores = other.scores;
        player_holes = other.player_holes;
        current_player = other.current_player;
        player1_agent = other.player1_agent;
        player2_agent = other.player2_agent;
        turn_number = other.turn_number;
    }

    AwaleGame clone() const {
        return AwaleGame(*this);
    }

    // Check if a given (hole, color) is a valid move for current_player
    bool is_valid_move(int hole, int color) const {
        if (hole < 0 || hole >= 16) return false;
        if (color < 0 || color > 1) return false;
        // hole must belong to current_player
        const auto& holes_for_current = player_holes[current_player];
        if (std::find(holes_for_current.begin(), holes_for_current.end(), hole) == holes_for_current.end()) {
            return false;
        }
        // Must have seeds
        if (board[hole][color] <= 0) {
            return false;
        }
        return true;
    }

    // Return all valid moves for current_player as vector of (hole, color)
    std::vector<std::pair<int,int>> get_valid_moves() const {
        std::vector<std::pair<int,int>> moves;
        const auto& holes_for_current = player_holes[current_player];
        for (auto hole : holes_for_current) {
            // If board[hole][0] > 0 => valid red move
            if (board[hole][0] > 0) {
                moves.emplace_back(hole, 0);
            }
            // If board[hole][1] > 0 => valid blue move
            if (board[hole][1] > 0) {
                moves.emplace_back(hole, 1);
            }
        }
        return moves;
    }

    // Perform the move (hole, color) if valid; otherwise throw
    void play_move(int hole, int color) {
        if (!is_valid_move(hole, color)) {
            throw std::invalid_argument("Invalid move!");
        }
        int seeds_to_sow = board[hole][color];
        board[hole][color] = 0;

        // Opponent holes mask (holes that do not belong to current_player)
        std::vector<bool> opponent_mask(16, true);
        for (auto h : player_holes[current_player]) {
            opponent_mask[h] = false;
        }

        int current_index = hole;
        while (seeds_to_sow > 0) {
            current_index = (current_index + 1) % 16;
            // skip redistributing into the same hole we took from
            if (current_index == hole) {
                continue;
            }

            if (color == 0) {  // Red seeds
                // Only place red seeds in opponent's holes
                if (opponent_mask[current_index]) {
                    board[current_index][color] += 1;
                    seeds_to_sow -= 1;
                }
            } else {
                // Blue seeds can be placed in any hole
                board[current_index][color] += 1;
                seeds_to_sow -= 1;
            }
        }

        // Attempt capture from the last hole sown
        apply_capture(current_index);

        // Switch player
        current_player = 3 - current_player;
    }

    // Apply capture logic starting from `start_hole` backward
    void apply_capture(int start_hole) {
        int idx = start_hole;
        while (true) {
            int total = board[idx][0] + board[idx][1];
            if (total == 2 || total == 3) {
                scores[current_player - 1] += total;
                board[idx][0] = 0;
                board[idx][1] = 0;
                idx = (idx - 1 + 16) % 16;
            } else {
                break;
            }
        }
    }

    // Check if the game is over
    bool game_over() const {
        int total_seeds = 0;
        for (int i = 0; i < 16; i++) {
            total_seeds += board[i][0];
            total_seeds += board[i][1];
        }
        // Game ends if fewer than 8 seeds remain
        if (total_seeds < 8) return true;
        // Or if any player reached 33 or more
        if (scores[0] >= 33 || scores[1] >= 33) return true;
        // Or if both players have 32 (tie)
        if (scores[0] == 32 && scores[1] == 32) return true;
        // Or if turn number >= 150
        if (turn_number >= 150) return true;
        return false;
    }

    // Return "Joueur 1", "Joueur 2", or "Égalité"
    std::string get_winner() const {
        if (scores[0] > scores[1]) return "Joueur 1";
        else if (scores[1] > scores[0]) return "Joueur 2";
        else return "Égalité";
    }

    // Helper to get the agent of the current player
    Agent* get_current_agent() const {
        return (current_player == 1) ? player1_agent : player2_agent;
    }

    // Display the board in a minimal textual format
    void display_board(int turn_num, std::pair<int,int> last_move = {-1, -1},
                       int depth_reached = -1, double calc_time = -1) {

        // If last_move is valid, show it
        if (last_move.first >= 0) {
            std::cout << "\nLast move by Player " << (3 - current_player)
                      << " => Hole " << (last_move.first + 1)
                      << (last_move.second == 0 ? "R" : "B");
            if (depth_reached >= 0 && calc_time >= 0) {
                std::cout << " [time=" << calc_time << "s, depth=" << depth_reached << "]";
            }
            std::cout << std::endl;
        }

        // Show turn, scores
        std::cout << "\nTurn " << turn_num << " (P1=" << scores[0] << ", P2=" << scores[1] << "):\n";

        // Hole numbering
        std::cout << "    N: ";
        for (int i = 0; i < 16; i++) {
            std::cout << (i+1) << ((i+1<10) ? "  " : " ");
        }
        std::cout << std::endl << "    -----------------------------------------------\n";

        // Red seeds row
        std::cout << "    R: ";
        for (int i = 0; i < 16; i++) {
            int val = board[i][0];
            if (val < 10) std::cout << "0" << val << " ";
            else std::cout << val << " ";
        }
        std::cout << std::endl;

        // Blue seeds row
        std::cout << "    B: ";
        for (int i = 0; i < 16; i++) {
            int val = board[i][1];
            if (val < 10) std::cout << "0" << val << " ";
            else std::cout << val << " ";
        }
        std::cout << std::endl << "    -----------------------------------------------\n";

        // Total seeds row
        std::cout << "    T: ";
        for (int i = 0; i < 16; i++) {
            int val = board[i][0] + board[i][1];
            if (val < 10) std::cout << "0" << val << " ";
            else std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    // Display endgame results
    void display_game_end() {
        std::cout << "\nENDGAME\n";
        if (scores[0] == scores[1]) {
            std::cout << "WINNER: TIE\n";
        } else if (turn_number >= 150 && scores[0] == scores[1]) {
            std::cout << "WINNER: TIE (turn limit reached)\n";
        } else {
            if (scores[0] > scores[1]) {
                std::cout << "WINNER: Player 1\n";
            } else {
                std::cout << "WINNER: Player 2\n";
            }
        }
        std::cout << "\nSCORE:\n";
        std::cout << "  Player1: " << scores[0] << "\n";
        std::cout << "  Player2: " << scores[1] << "\n";
        std::cout << std::endl;
    }

    // Run the game loop
    void run_game() {
        display_board(turn_number);

        while (!game_over()) {
            turn_number++;
            // Get move from the current agent
            auto [move, compute_time, depth_reached] = get_current_agent()->get_move(*this);

            // If agent returns a sentinel or no moves, we break
            if (move.first < 0) {
                std::cout << "No valid move. Game ends.\n";
                break;
            }

            // Copy board before move if needed (omitted)...

            // Attempt to play it
            try {
                play_move(move.first, move.second);
            } catch (std::exception& e) {
                // invalid move => revert turn_number and continue
                std::cerr << e.what() << std::endl;
                turn_number--;
                continue;
            }

            // Show new board
            display_board(turn_number, move, depth_reached, compute_time);
        }

        // Show final result
        display_game_end();
    }
};

// ---------------------------------------------------------
//                    HumanAgent
// ---------------------------------------------------------
class HumanAgent : public Agent {
public:
    std::tuple<std::pair<int,int>, double, int>
    get_move(AwaleGame& game_state) override {
        while (true) {
            std::cout << "\n[Human] Choose a hole (1-16): ";
            int hole_input;
            std::cin >> hole_input;
            if (!std::cin.good()) {
                std::cin.clear();
                std::cin.ignore(10000, '\n');
                std::cout << "Invalid input, try again.\n";
                continue;
            }
            std::cout << "[Human] Choose a color (0=Red, 1=Blue): ";
            int color_input;
            std::cin >> color_input;
            if (!std::cin.good()) {
                std::cin.clear();
                std::cin.ignore(10000, '\n');
                std::cout << "Invalid input, try again.\n";
                continue;
            }

            int hole = hole_input - 1;
            int color = color_input;

            if (game_state.is_valid_move(hole, color)) {
                return {{hole, color}, 0.0, 0}; // no timing or depth used
            } else {
                std::cout << "Invalid move. Try again.\n";
            }
        }
        // Fallback, unreachable normally
        return {{-1, -1}, 0, 0};
    }
};

// ---------------------------------------------------------
//                   RandomAgent
// ---------------------------------------------------------
class RandomAgent : public Agent {
private:
    std::mt19937 rng;
public:
    RandomAgent() {
        // Seed RNG
        std::random_device rd;
        rng.seed(rd());
    }

    std::tuple<std::pair<int,int>, double, int>
    get_move(AwaleGame& game_state) override {
        auto valid_moves = game_state.get_valid_moves();
        if (valid_moves.empty()) {
            std::cout << "Aucun coup valide disponible.\n";
            return {{-1, -1}, 0.0, 0};
        }
        std::uniform_int_distribution<size_t> dist(0, valid_moves.size()-1);
        auto choice = valid_moves[dist(rng)];
        std::cout << "[RandomAgent] Chooses Hole " << (choice.first+1)
                  << (choice.second == 0 ? "R" : "B") << "\n";
        return {choice, 0.0, 0};
    }
};

// ---------------------------------------------------------
//                 MinimaxAgent6 (parent)
// ---------------------------------------------------------
struct TTEntry {
    int depth;
    double value;
    int flag;
    std::pair<int,int> best_move;
};

class MinimaxAgent6 : public Agent {
protected:
    double max_time;
    // Some parameters used in evaluation
    const double SCORE_WEIGHT = 50.0;
    const double CONTROL_WEIGHT = 30.0;
    const double CAPTURE_WEIGHT = 20.0;
    const double MOBILITY_WEIGHT = 15.0;
    // Transposition table
    std::unordered_map<size_t, TTEntry> transposition_table;
    // Move ordering data
    std::unordered_map<std::pair<int,int>, double,
        std::function<size_t(const std::pair<int,int>&)> > move_ordering;
    // We can do a small custom pair-hash
    static size_t pair_hash(const std::pair<int,int>& p) {
        // a basic combination
        // or we can do something more robust
        auto h1 = std::hash<int>()(p.first);
        auto h2 = std::hash<int>()(p.second);
        // 31 is just a prime
        return (h1 * 31) ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1<<6) + (h1>>2));
    }

    // For transposition table flags
    enum { FLAG_EXACT, FLAG_LOWERBOUND, FLAG_UPPERBOUND };

    // This is a rough measure; you might want something more robust
    static size_t state_hash(const AwaleGame& st) {
        // We'll just combine all board seeds + scores + current_player
        // into a single hash. This is not collision-resistant for large use,
        // but it's a demonstration.
        size_t h = 0;
        for (int i = 0; i < 16; i++) {
            h ^= std::hash<int>()(st.board[i][0] << 8 | st.board[i][1]) + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2);
        }
        h ^= std::hash<int>()(st.scores[0] << 16 | st.scores[1]);
        h ^= std::hash<int>()(st.current_player);
        return h;
    }

public:
    MinimaxAgent6(double max_t = 2.0)
      : max_time(max_t),
        move_ordering(0,
          std::function<size_t(const std::pair<int,int>&)>(pair_hash)) {
        // Reserve or do any init as needed
    }

    // Evaluate the game state heuristically
    virtual double evaluate(const AwaleGame& gs) {
        int my_idx = gs.current_player - 1;
        int opp_idx = 1 - my_idx;

        double score_diff = gs.scores[my_idx] - gs.scores[opp_idx];

        // sum seeds on my holes vs. opponent holes
        int my_seeds = 0;
        for (auto h : gs.player_holes[gs.current_player]) {
            my_seeds += gs.board[h][0] + gs.board[h][1];
        }
        int opp_seeds = 0;
        for (auto h : gs.player_holes[3 - gs.current_player]) {
            opp_seeds += gs.board[h][0] + gs.board[h][1];
        }

        // capture potential (super naive).
        // Let's say each hole with total seeds==1 or 4 is "capture-prone"
        int total_per_hole[16];
        int my_capture_spots = 0, opp_capture_spots = 0;
        for (int i=0; i<16; i++) {
            total_per_hole[i] = gs.board[i][0] + gs.board[i][1];
        }
        // Now count how many of these are on my side vs opp side
        for (auto h : gs.player_holes[gs.current_player]) {
            if (total_per_hole[h] == 1 || total_per_hole[h] == 4) {
                my_capture_spots++;
            }
        }
        for (auto h : gs.player_holes[3 - gs.current_player]) {
            if (total_per_hole[h] == 1 || total_per_hole[h] == 4) {
                opp_capture_spots++;
            }
        }
        double capture_potential = (my_capture_spots - opp_capture_spots) * 2.0;

        // mobility
        auto my_valid_moves = gs.get_valid_moves(); // for current_player
        // Switch current_player to the other side just to measure their mobility:
        AwaleGame tmp = gs.clone();
        tmp.current_player = 3 - tmp.current_player;
        auto opp_valid_moves = tmp.get_valid_moves();
        double mobility_diff = (double)my_valid_moves.size() - (double)opp_valid_moves.size();

        double value =
            SCORE_WEIGHT * score_diff +
            CONTROL_WEIGHT * (my_seeds - opp_seeds) +
            CAPTURE_WEIGHT * capture_potential +
            MOBILITY_WEIGHT * mobility_diff;

        return value;
    }

    // The minimax function with alpha-beta
    // We'll define a recursive method
    virtual std::pair<double, std::pair<int,int>>
    minimax(AwaleGame& gs, int depth, double alpha, double beta, bool maximizing,
            const std::chrono::steady_clock::time_point& start_time,
            double allotted_time, bool is_root = false) {

        // Time check
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed >= allotted_time) {
            // We throw to break out
            throw std::runtime_error("Timeout");
        }

        // Transposition Table check
        size_t h = state_hash(gs);
        // We'll define "depth" as a cutoff. If we have an entry with >= depth, we might use it
        if (!is_root) {
            auto it = transposition_table.find(h);
            if (it != transposition_table.end()) {
                TTEntry& entry = it->second;
                if (entry.depth >= depth) {
                    if (entry.flag == FLAG_EXACT) {
                        return {entry.value, entry.best_move};
                    } else if (entry.flag == FLAG_LOWERBOUND) {
                        alpha = std::max(alpha, entry.value);
                    } else if (entry.flag == FLAG_UPPERBOUND) {
                        beta = std::min(beta, entry.value);
                    }
                    if (alpha >= beta) {
                        return {entry.value, entry.best_move};
                    }
                }
            }
        }

        // Terminal or depth check
        if (gs.game_over() || depth == 0) {
            double val = evaluate(gs);
            return {val, {-1, -1}};
        }

        auto moves = gs.get_valid_moves();
        if (moves.empty()) {
            double val = evaluate(gs);
            return {val, {-1, -1}};
        }

        // Possibly sort moves by some heuristic (move_ordering):
        std::sort(moves.begin(), moves.end(),
                  [&](auto& a, auto& b){
                    double va = (move_ordering.find(a) != move_ordering.end()) ? move_ordering[a] : 0.0;
                    double vb = (move_ordering.find(b) != move_ordering.end()) ? move_ordering[b] : 0.0;
                    return va > vb; // descending
                  });

        double best_value = maximizing ? -std::numeric_limits<double>::infinity()
                                       :  std::numeric_limits<double>::infinity();
        std::pair<int,int> best_mv = {-1, -1};

        for (auto& mv : moves) {
            // For demonstration, no fancy "late-move pruning" here
            AwaleGame clone_gs = gs.clone();
            clone_gs.play_move(mv.first, mv.second);

            double val;
            std::pair<int,int> dummy;
            try {
                std::tie(val, dummy) = minimax(clone_gs, depth-1, alpha, beta, !maximizing,
                                               start_time, allotted_time, false);
            } catch (std::runtime_error&) {
                // Bubble up the timeout
                throw;
            }

            if (maximizing) {
                if (val > best_value) {
                    best_value = val;
                    best_mv = mv;
                }
                alpha = std::max(alpha, val);
            } else {
                if (val < best_value) {
                    best_value = val;
                    best_mv = mv;
                }
                beta = std::min(beta, val);
            }

            if (beta <= alpha) {
                // alpha-beta cutoff
                break;
            }
        }

        // Store to transposition table
        if (transposition_table.size() < 1000000) {
            TTEntry entry;
            entry.depth = depth;
            entry.value = best_value;
            // Determine flag
            if (best_value <= alpha) {
                entry.flag = FLAG_UPPERBOUND;
            } else if (best_value >= beta) {
                entry.flag = FLAG_LOWERBOUND;
            } else {
                entry.flag = FLAG_EXACT;
            }
            entry.best_move = best_mv;
            transposition_table[h] = entry;
        }

        // Also update move_ordering
        if (best_mv.first >= 0) {
            // Increase ordering value for best move
            move_ordering[best_mv] = std::max(move_ordering[best_mv], best_value);
        }

        return {best_value, best_mv};
    }

    // The iterative deepening search
    std::tuple<std::pair<int,int>, double, int>
    get_move(AwaleGame& game_state) override {
        auto start = std::chrono::steady_clock::now();
        double allotted = max_time; // in seconds

        int depth = 1;
        std::pair<int,int> best_move = {-1, -1};
        while (true) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed >= allotted) {
                break;
            }
            try {
                auto [val, mv] = minimax(game_state, depth,
                                         -std::numeric_limits<double>::infinity(),
                                          std::numeric_limits<double>::infinity(),
                                         true,
                                         start,
                                         allotted,
                                         true);
                if (mv.first >= 0) {
                    best_move = mv;
                }
            } catch (std::runtime_error&) {
                // Timeout
                break;
            }
            depth++;
        }

        double total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        // Return (best_move, compute_time, depth-1)
        // because the last depth increment didn't finish
        return {best_move, total_time, depth-1};
    }
};

// ---------------------------------------------------------
//                MinimaxAgent6_4 (child)
// ---------------------------------------------------------
// Inherits from MinimaxAgent6 and can override certain methods.
// For brevity, we’ll just demonstrate that it extends the constructor
// and maybe modifies ordering. We'll keep it very close to the
// parent's approach. We won't implement the full advanced logic
// of the Python example (null move, capturing checks, etc.)
// but we show the skeleton.
class MinimaxAgent6_4 : public MinimaxAgent6 {
private:
    std::pair<int, int> principal_variation_move = {-1, -1};

public:
    MinimaxAgent6_4(double max_t = 2.0) : MinimaxAgent6(max_t) {}

    // Override the get_move method for iterative deepening with PV ordering
    std::tuple<std::pair<int, int>, double, int>
    get_move(AwaleGame &game_state) override {
        auto start = std::chrono::steady_clock::now();
        double allotted = max_time; // in seconds

        int depth = 1;
        std::pair<int, int> best_move = {-1, -1};

        while (true) {
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start).count();
            if (elapsed >= allotted) {
                break;
            }
            try {
                auto [val, mv] = minimax(game_state, depth,
                                         -std::numeric_limits<double>::infinity(),
                                         std::numeric_limits<double>::infinity(),
                                         true,
                                         start,
                                         allotted,
                                         true);
                if (mv.first >= 0) {
                    best_move = mv;
                    // Store as the principal variation move for the next iteration
                    principal_variation_move = mv;
                }
            } catch (std::runtime_error &) {
                // Timeout
                break;
            }
            depth++;
        }

        double total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        return {best_move, total_time, depth - 1};
    }

    // Override minimax to prioritize PV and incorporate 1-ply simulation
    std::pair<double, std::pair<int, int>>
    minimax(AwaleGame &gs, int depth, double alpha, double beta, bool maximizing,
            const std::chrono::steady_clock::time_point &start_time,
            double allotted_time, bool is_root = false) override {

        // Time check
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        if (elapsed >= allotted_time) {
            throw std::runtime_error("Timeout");
        }

        // Transposition Table lookup
        size_t h = state_hash(gs);
        if (!is_root) {
            auto it = transposition_table.find(h);
            if (it != transposition_table.end()) {
                TTEntry &entry = it->second;
                if (entry.depth >= depth) {
                    if (entry.flag == FLAG_EXACT) {
                        return {entry.value, entry.best_move};
                    } else if (entry.flag == FLAG_LOWERBOUND) {
                        alpha = std::max(alpha, entry.value);
                    } else if (entry.flag == FLAG_UPPERBOUND) {
                        beta = std::min(beta, entry.value);
                    }
                    if (alpha >= beta) {
                        return {entry.value, entry.best_move};
                    }
                }
            }
        }

        // Terminal or depth check
        if (gs.game_over() || depth == 0) {
            double val = evaluate(gs);
            return {val, {-1, -1}};
        }

        auto moves = gs.get_valid_moves();
        if (moves.empty()) {
            double val = evaluate(gs);
            return {val, {-1, -1}};
        }

        // Incorporate principal variation move if available
        if (is_root && principal_variation_move.first >= 0 &&
            std::find(moves.begin(), moves.end(), principal_variation_move) != moves.end()) {
            std::rotate(moves.begin(),
                        std::find(moves.begin(), moves.end(), principal_variation_move),
                        moves.end());
        }

        // Sort remaining moves by capture potential
        std::sort(moves.begin(), moves.end(), [&](auto &a, auto &b) {
            return move_score(gs, a) > move_score(gs, b);
        });

        double best_value = maximizing ? -std::numeric_limits<double>::infinity()
                                       : std::numeric_limits<double>::infinity();
        std::pair<int, int> best_mv = {-1, -1};

        for (auto &mv : moves) {
            AwaleGame clone_gs = gs.clone();
            clone_gs.play_move(mv.first, mv.second);

            double val;
            std::pair<int, int> dummy;
            try {
                std::tie(val, dummy) = minimax(clone_gs, depth - 1, alpha, beta, !maximizing,
                                               start_time, allotted_time, false);
            } catch (std::runtime_error &) {
                // Bubble up the timeout
                throw;
            }

            if (maximizing) {
                if (val > best_value) {
                    best_value = val;
                    best_mv = mv;
                }
                alpha = std::max(alpha, val);
            } else {
                if (val < best_value) {
                    best_value = val;
                    best_mv = mv;
                }
                beta = std::min(beta, val);
            }

            if (beta <= alpha) {
                break; // Alpha-beta pruning
            }
        }

        // Store to transposition table
        if (transposition_table.size() < 1000000) {
            TTEntry entry;
            entry.depth = depth;
            entry.value = best_value;
            entry.flag = (best_value <= alpha)
                             ? FLAG_UPPERBOUND
                             : (best_value >= beta) ? FLAG_LOWERBOUND : FLAG_EXACT;
            entry.best_move = best_mv;
            transposition_table[h] = entry;
        }

        return {best_value, best_mv};
    }

    // Add 1-ply simulation to refine move ordering
    double move_score(AwaleGame &gs, std::pair<int, int> move) {
        AwaleGame clone = gs.clone();
        clone.play_move(move.first, move.second);

        int before_seeds = 0, after_seeds = 0;
        for (int i = 0; i < 16; i++) {
            before_seeds += gs.board[i][0] + gs.board[i][1];
            after_seeds += clone.board[i][0] + clone.board[i][1];
        }

        // Difference indicates seeds captured
        int captured = before_seeds - after_seeds;
        return 10000 + captured; // Arbitrary weight for captured seeds
    }
};

// ---------------------------------------------------------
//                          MAIN
// ---------------------------------------------------------
int main() {
    // Example usage:
    // 1) Human vs Random
    //    AwaleGame game(new HumanAgent(), new RandomAgent());
    //    game.run_game();

    // 2) Random vs Random
    //    AwaleGame game(new RandomAgent(), new RandomAgent());
    //    game.run_game();

    // 3) MinimaxAgent6_4 vs Random
    //    AwaleGame game(new MinimaxAgent6_4(2.0), new RandomAgent());
    //    game.run_game();

    // 4) Human vs MinimaxAgent6_4
    //    AwaleGame game(new HumanAgent(), new MinimaxAgent6_4(2.0));
    //    game.run_game();

    // Modify the lines below as desired to test:

    std::cout << "=== Awale C++ Example ===\n";

    // Example: MinimaxAgent6_4 vs Random
    AwaleGame game(new MinimaxAgent6_4(2.0), new RandomAgent());
    game.run_game();

    // Clean up dynamic allocations:
    delete game.player1_agent;
    delete game.player2_agent;

    return 0;
}
