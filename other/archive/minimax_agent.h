// minimax_agent.h
#ifndef MINIMAX_AGENT_H
#define MINIMAX_AGENT_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include <math.h>

#define MAX_TABLE_SIZE 1000000
#define BOARD_SIZE 16
#define NUM_COLORS 2
#define FLAG_EXACT 0
#define FLAG_LOWERBOUND 1
#define FLAG_UPPERBOUND 2

// Constants for evaluation weights
#define SCORE_WEIGHT 50
#define CONTROL_WEIGHT 30
#define CAPTURE_WEIGHT 20
#define MOBILITY_WEIGHT 15

typedef struct {
    int hole;
    int color;
} Move;

typedef struct {
    int board[BOARD_SIZE][NUM_COLORS];
    int scores[2];
    int current_player;
    int game_over;
} GameState;

typedef struct {
    int depth;
    double value;
    int flag;
    Move best_move;
} TTEntry;

typedef struct {
    Move move;
    double score;
} ScoredMove;

typedef struct {
    TTEntry* table;
    size_t size;
    Move principal_variation_move;
    double max_time;
    int nodes_cut;
} MinimaxAgent;

// Function declarations
MinimaxAgent* create_minimax_agent(double max_time);
void free_minimax_agent(MinimaxAgent* agent);
Move get_move(MinimaxAgent* agent, GameState* game_state);
double evaluate(GameState* game_state);
int is_valid_move(GameState* game_state, Move move);
void get_valid_moves(GameState* game_state, Move* moves, int* num_moves);
GameState* clone_game_state(GameState* game_state);
void play_move(GameState* game_state, Move move);
void apply_capture(GameState* game_state, int start_hole);
double minimax(MinimaxAgent* agent, GameState* game_state, int depth, double alpha,
               double beta, int maximizing_player, double start_time, int is_root, Move* best_move);
size_t get_state_hash(GameState* game_state);
double move_score(GameState* game_state, Move move);
void order_moves(MinimaxAgent* agent, GameState* game_state, Move* moves,
                int num_moves, int depth, int is_root);

#endif

// minimax_agent.c
#include "minimax_agent.h"

MinimaxAgent* create_minimax_agent(double max_time) {
    MinimaxAgent* agent = (MinimaxAgent*)malloc(sizeof(MinimaxAgent));
    agent->table = (TTEntry*)calloc(MAX_TABLE_SIZE, sizeof(TTEntry));
    agent->size = 0;
    agent->max_time = max_time;
    agent->nodes_cut = 0;
    agent->principal_variation_move.hole = -1;
    agent->principal_variation_move.color = -1;
    return agent;
}

void free_minimax_agent(MinimaxAgent* agent) {
    free(agent->table);
    free(agent);
}

Move get_move(MinimaxAgent* agent, GameState* game_state) {
    double start_time = (double)clock() / CLOCKS_PER_SEC;
    int depth = 1;
    Move best_move = {-1, -1};
    Move current_best;

    while (1) {
        double elapsed_time = (double)clock() / CLOCKS_PER_SEC - start_time;
        if (elapsed_time >= agent->max_time) {
            break;
        }

        double eval = minimax(agent, game_state, depth, -DBL_MAX, DBL_MAX, 1,
                            start_time, 1, &current_best);

        if (current_best.hole != -1) {
            best_move = current_best;
            agent->principal_variation_move = best_move;
        }

        depth++;
    }

    return best_move;
}

double evaluate(GameState* game_state) {
    int my_index = game_state->current_player - 1;
    int opp_index = 1 - my_index;

    // Score difference
    double score_diff = game_state->scores[my_index] - game_state->scores[opp_index];

    // Board control
    int my_seeds = 0, opp_seeds = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i % 2 == my_index) {
            my_seeds += game_state->board[i][0] + game_state->board[i][1];
        } else {
            opp_seeds += game_state->board[i][0] + game_state->board[i][1];
        }
    }

    // Capture potential
    int my_capture = 0, opp_capture = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        int total = game_state->board[i][0] + game_state->board[i][1];
        if (total == 1 || total == 4) {
            if (i % 2 == my_index) my_capture++;
            else opp_capture++;
        }
    }

    // Mobility
    int my_mobility = 0, opp_mobility = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (game_state->board[i][0] > 0 || game_state->board[i][1] > 0) {
            if (i % 2 == my_index) my_mobility++;
            else opp_mobility++;
        }
    }

    return SCORE_WEIGHT * score_diff +
           CONTROL_WEIGHT * (my_seeds - opp_seeds) +
           CAPTURE_WEIGHT * (my_capture - opp_capture) * 2 +
           MOBILITY_WEIGHT * (my_mobility - opp_mobility);
}

int is_valid_move(GameState* game_state, Move move) {
    if (move.hole < 0 || move.hole >= BOARD_SIZE) return 0;
    if (move.color < 0 || move.color >= NUM_COLORS) return 0;
    if (move.hole % 2 != (game_state->current_player - 1)) return 0;
    return game_state->board[move.hole][move.color] > 0;
}

void get_valid_moves(GameState* game_state, Move* moves, int* num_moves) {
    *num_moves = 0;
    for (int i = 0; i < BOARD_SIZE; i++) {
        if (i % 2 == (game_state->current_player - 1)) {
            for (int color = 0; color < NUM_COLORS; color++) {
                if (game_state->board[i][color] > 0) {
                    moves[*num_moves].hole = i;
                    moves[*num_moves].color = color;
                    (*num_moves)++;
                }
            }
        }
    }
}

GameState* clone_game_state(GameState* game_state) {
    GameState* clone = (GameState*)malloc(sizeof(GameState));
    memcpy(clone, game_state, sizeof(GameState));
    return clone;
}

void play_move(GameState* game_state, Move move) {
    int seeds_to_sow = game_state->board[move.hole][move.color];
    game_state->board[move.hole][move.color] = 0;

    int current_index = move.hole;
    while (seeds_to_sow > 0) {
        current_index = (current_index + 1) % BOARD_SIZE;
        if (current_index == move.hole) continue;

        int total_seeds = game_state->board[current_index][0] +
                         game_state->board[current_index][1];
        if (total_seeds >= 16) continue;

        if (move.color == 0) { // Red seeds
            if (current_index % 2 != (game_state->current_player - 1)) {
                game_state->board[current_index][move.color]++;
                seeds_to_sow--;
            }
        } else { // Blue seeds
            game_state->board[current_index][move.color]++;
            seeds_to_sow--;
        }
    }

    apply_capture(game_state, current_index);
    game_state->current_player = 3 - game_state->current_player;
}

void apply_capture(GameState* game_state, int start_hole) {
    int current_index = start_hole;
    while (1) {
        int total_seeds = game_state->board[current_index][0] +
                         game_state->board[current_index][1];
        if (total_seeds == 2 || total_seeds == 3) {
            game_state->scores[game_state->current_player - 1] += total_seeds;
            game_state->board[current_index][0] = 0;
            game_state->board[current_index][1] = 0;
            current_index = (current_index - 1 + BOARD_SIZE) % BOARD_SIZE;
        } else {
            break;
        }
    }
}

double minimax(MinimaxAgent* agent, GameState* game_state, int depth, double alpha,
               double beta, int maximizing_player, double start_time, int is_root,
               Move* best_move) {
    if ((double)clock() / CLOCKS_PER_SEC - start_time >= agent->max_time) {
        best_move->hole = -1;
        best_move->color = -1;
        return 0.0;
    }

    size_t hash = get_state_hash(game_state);
    if (!is_root) {
        TTEntry* entry = &agent->table[hash % MAX_TABLE_SIZE];
        if (entry->depth >= depth) {
            if (entry->flag == FLAG_EXACT) {
                *best_move = entry->best_move;
                return entry->value;
            }
            if (entry->flag == FLAG_LOWERBOUND) alpha = fmax(alpha, entry->value);
            if (entry->flag == FLAG_UPPERBOUND) beta = fmin(beta, entry->value);
            if (alpha >= beta) {
                agent->nodes_cut++;
                *best_move = entry->best_move;
                return entry->value;
            }
        }
    }

    if (game_state->game_over || depth == 0) {
        best_move->hole = -1;
        best_move->color = -1;
        return evaluate(game_state);
    }

    Move moves[BOARD_SIZE * NUM_COLORS];
    int num_moves;
    get_valid_moves(game_state, moves, &num_moves);

    if (num_moves == 0) {
        best_move->hole = -1;
        best_move->color = -1;
        return evaluate(game_state);
    }

    order_moves(agent, game_state, moves, num_moves, depth, is_root);

    double best_value = maximizing_player ? -DBL_MAX : DBL_MAX;
    Move local_best_move = {-1, -1};

    for (int i = 0; i < num_moves; i++) {
        GameState* clone = clone_game_state(game_state);
        play_move(clone, moves[i]);

        Move child_move;
        double eval = minimax(agent, clone, depth - 1, alpha, beta,
                            !maximizing_player, start_time, 0, &child_move);

        free(clone);

        if (maximizing_player) {
            if (eval > best_value) {
                best_value = eval;
                local_best_move = moves[i];
            }
            alpha = fmax(alpha, best_value);
        } else {
            if (eval < best_value) {
                best_value = eval;
                local_best_move = moves[i];
            }
            beta = fmin(beta, best_value);
        }

        if (beta <= alpha) {
            agent->nodes_cut++;
            break;
        }
    }

    if (agent->size < MAX_TABLE_SIZE) {
        TTEntry entry;
        entry.depth = depth;
        entry.value = best_value;
        entry.best_move = local_best_move;

        if (best_value <= alpha) entry.flag = FLAG_UPPERBOUND;
        else if (best_value >= beta) entry.flag = FLAG_LOWERBOUND;
        else entry.flag = FLAG_EXACT;

        agent->table[hash % MAX_TABLE_SIZE] = entry;
        agent->size++;
    }

    *best_move = local_best_move;
    return best_value;
}

size_t get_state_hash(GameState* game_state) {
    size_t hash = 5381;
    unsigned char* bytes = (unsigned char*)game_state;
    for (size_t i = 0; i < sizeof(GameState); i++) {
        hash = ((hash << 5) + hash) + bytes[i];
    }
    return hash;
}

double move_score(GameState* game_state, Move move) {
    GameState* clone = clone_game_state(game_state);
    int before_seeds = 0;
    int after_seeds = 0;

    for (int i = 0; i < BOARD_SIZE; i++) {
        before_seeds += game_state->board[i][0] + game_state->board[i][1];
    }

    play_move(clone, move);

    for (int i = 0; i < BOARD_SIZE; i++) {
        after_seeds += clone->board[i][0] + clone->board[i][1];
    }

    free(clone);

    if (after_seeds < before_seeds) {
        return 10000.0 + (before_seeds - after_seeds);
    }
    return 0.0;
}

void order_moves(MinimaxAgent* agent, GameState* game_state, Move* moves,
                int num_moves, int depth, int is_root) {
    if (is_root && agent->principal_variation_move.hole != -1) {
        // Find and move PV move to front if present
        for (int i = 0; i < num_moves; i++) {
            if (moves[i].hole == agent->principal_variation_move.hole &&
                moves[i].color == agent->principal_variation_move.color) {
                Move temp = moves[0];
                moves[0] = moves[i];
                moves[i] = temp;
                break;
            }
        }
    }

    // Sort remaining moves by score
    ScoredMove* scored_moves = (ScoredMove*)malloc(num_moves * sizeof(ScoredMove));
    for (int i = 0; i < num_moves; i++) {
        scored_moves[i].move = moves[i];
        scored_moves[i].score = move_score(game_state, moves[i]);
    }

    // Bubble sort (for simplicity - can be replaced with quicksort for better performance)
    for (int i = 0; i < num_moves - 1; i++) {
        for (int j = 0; j < num_moves - i - 1; j++) {
            if (scored_moves[j].score < scored_moves[j + 1].score) {
                ScoredMove temp = scored_moves[j];
                scored_moves[j] = scored_moves[j + 1];
                scored_moves[j + 1] = temp;
            }
        }
    }

    // Copy back sorted moves
    for (int i = 0; i < num_moves; i++) {
        moves[i] = scored_moves[i].move;
    }

    free(scored_moves);
}
