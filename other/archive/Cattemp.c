#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <limits.h>
#include <string.h>

#define HOLES 16
#define SEEDS_PER_HOLE 4
#define PLAYER_1 1
#define PLAYER_2 2
#define RED 0
#define BLUE 1
#define MAX_DEPTH 10
#define MAX_TIME 2.0
#define FLAG_EXACT 0
#define FLAG_LOWERBOUND 1
#define FLAG_UPPERBOUND 2

// Transposition table structure
typedef struct {
    unsigned long long hash;
    int depth;
    int value;
    int flag;
    int best_hole;
    int best_color;
} TranspositionTableEntry;

#define MAX_TRANSPOSITION_TABLE_SIZE 100000
TranspositionTableEntry transposition_table[MAX_TRANSPOSITION_TABLE_SIZE];

unsigned long long principal_variation_move_hash = 0;

// Function prototypes
void initialize_board(int board[HOLES][2]);
void print_board(int board[HOLES][2]);
int make_random_move(int board[HOLES][2], int player, int *captured_red, int *captured_blue);
int make_human_move(int board[HOLES][2], int player, int *captured_red, int *captured_blue);
int make_minimax_move(int board[HOLES][2], int player, int *captured_red, int *captured_blue);
int is_valid_move(int board[HOLES][2], int hole, int player, int color);
void sow_seeds(int board[HOLES][2], int hole, int color, int player, int *captured_red, int *captured_blue);
int check_game_end(int captured_red_1, int captured_blue_1, int captured_red_2, int captured_blue_2);
int evaluate_board(int board[HOLES][2], int player);
int minimax(int board[HOLES][2], int depth, int alpha, int beta, int maximizing_player, int player, int *best_hole, int *best_color, clock_t start_time);
unsigned long long compute_hash(int board[HOLES][2], int player);
int _move_score(int board[HOLES][2], int hole, int color, int player);

// Player types
#define PLAYER_TYPE_HUMAN 0
#define PLAYER_TYPE_RANDOM 1
#define PLAYER_TYPE_MINIMAX 2

int main() {
    int board[HOLES][2];
    int captured_red_1 = 0, captured_blue_1 = 0;
    int captured_red_2 = 0, captured_blue_2 = 0;
    int current_player = PLAYER_1;

    // Player type selection
    int player1_type, player2_type;
    printf("Select Player 1 type (0: Human, 1: Random, 2: Minimax): ");
    scanf("%d", &player1_type);
    printf("Select Player 2 type (0: Human, 1: Random, 2: Minimax): ");
    scanf("%d", &player2_type);

    srand(time(NULL));
    initialize_board(board);

    while (1) {
        print_board(board);
        printf("Player %d's turn.\n", current_player);

        if (current_player == PLAYER_1) {
            if (player1_type == PLAYER_TYPE_HUMAN) {
                if (!make_human_move(board, current_player, &captured_red_1, &captured_blue_1)) {
                    printf("No valid moves for Player %d.\n", current_player);
                }
            } else if (player1_type == PLAYER_TYPE_RANDOM) {
                if (!make_random_move(board, current_player, &captured_red_1, &captured_blue_1)) {
                    printf("No valid moves for Player %d.\n", current_player);
                }
            } else if (player1_type == PLAYER_TYPE_MINIMAX) {
                if (!make_minimax_move(board, current_player, &captured_red_1, &captured_blue_1)) {
                    printf("No valid moves for Player %d.\n", current_player);
                }
            }
        } else {
            if (player2_type == PLAYER_TYPE_HUMAN) {
                if (!make_human_move(board, current_player, &captured_red_2, &captured_blue_2)) {
                    printf("No valid moves for Player %d.\n", current_player);
                }
            } else if (player2_type == PLAYER_TYPE_RANDOM) {
                if (!make_random_move(board, current_player, &captured_red_2, &captured_blue_2)) {
                    printf("No valid moves for Player %d.\n", current_player);
                }
            } else if (player2_type == PLAYER_TYPE_MINIMAX) {
                if (!make_minimax_move(board, current_player, &captured_red_2, &captured_blue_2)) {
                    printf("No valid moves for Player %d.\n", current_player);
                }
            }
        }

        if (check_game_end(captured_red_1, captured_blue_1, captured_red_2, captured_blue_2)) {
            print_board(board);
            printf("Game Over!\n");
            printf("Player 1 captured %d seeds.\n", captured_red_1 + captured_blue_1);
            printf("Player 2 captured %d seeds.\n", captured_red_2 + captured_blue_2);
            if (captured_red_1 + captured_blue_1 > captured_red_2 + captured_blue_2) {
                printf("Player 1 wins!\n");
            } else if (captured_red_2 + captured_blue_2 > captured_red_1 + captured_blue_1) {
                printf("Player 2 wins!\n");
            } else {
                printf("It's a draw!\n");
            }
            break;
        }

        current_player = (current_player == PLAYER_1) ? PLAYER_2 : PLAYER_1;
    }

    return 0;
}

void initialize_board(int board[HOLES][2]) {
    for (int i = 0; i < HOLES; i++) {
        board[i][RED] = SEEDS_PER_HOLE / 2;
        board[i][BLUE] = SEEDS_PER_HOLE / 2;
    }
    memset(transposition_table, 0, sizeof(transposition_table));
    principal_variation_move_hash = 0;
}

void print_board(int board[HOLES][2]) {
    printf("\nBoard:\n");
    for (int i = 0; i < HOLES; i++) {
        printf("Hole %2d: Red: %d, Blue: %d\n", i + 1, board[i][RED], board[i][BLUE]);
    }
    printf("\n");
}

int make_random_move(int board[HOLES][2], int player, int *captured_red, int *captured_blue) {
    int valid_moves[HOLES];
    int move_count = 0;

    for (int i = 0; i < HOLES; i++) {
        if ((player == PLAYER_1 && (i % 2 == 0)) || (player == PLAYER_2 && (i % 2 != 0))) {
            for (int color = 0; color <= 1; color++) {
                if (is_valid_move(board, i, player, color)) {
                    valid_moves[move_count++] = i * 2 + color; // Encoded as hole * 2 + color
                }
            }
        }
    }

    if (move_count == 0) {
        return 0;
    }

    int random_index = rand() % move_count;
    int selected_hole = valid_moves[random_index] / 2;
    int selected_color = valid_moves[random_index] % 2;
    sow_seeds(board, selected_hole, selected_color, player, captured_red, captured_blue);

    return 1;
}

int make_human_move(int board[HOLES][2], int player, int *captured_red, int *captured_blue) {
    int hole, color;
    char color_char;

    while (1) {
        printf("Enter your move (hole and color, e.g., 3R or 4B): ");
        scanf("%d%c", &hole, &color_char);

        hole -= 1; // Convert to 0-based index
        color = (color_char == 'R' || color_char == 'r') ? RED : BLUE;

        if (is_valid_move(board, hole, player, color)) {
            sow_seeds(board, hole, color, player, captured_red, captured_blue);
            return 1;
        } else {
            printf("Invalid move. Try again.\n");
        }
    }
}

int make_minimax_move(int board[HOLES][2], int player, int *captured_red, int *captured_blue) {
    int best_hole = -1, best_color = -1;
    clock_t start_time = clock();
    int depth = 1;

    while (((double)(clock() - start_time) / CLOCKS_PER_SEC) < MAX_TIME) {
        int best_value = minimax(board, depth, INT_MIN, INT_MAX, 1, player, &best_hole, &best_color, start_time);
        depth++;
    }

    if (best_hole == -1 || best_color == -1) {
        return 0; // No valid move
    }

    sow_seeds(board, best_hole, best_color, player, captured_red, captured_blue);
    return 1;
}

int is_valid_move(int board[HOLES][2], int hole, int player, int color) {
    if (hole < 0 || hole >= HOLES) {
        return 0;
    }
    if ((player == PLAYER_1 && (hole % 2 == 1)) || (player == PLAYER_2 && (hole % 2 == 0))) {
        return 0;
    }
    return board[hole][color] > 0;
}

void sow_seeds(int board[HOLES][2], int hole, int color, int player, int *captured_red, int *captured_blue) {
    int seeds = board[hole][color];
    board[hole][color] = 0;
    int current_hole = hole;

    while (seeds > 0) {
        current_hole = (current_hole + 1) % HOLES;

        if (current_hole == hole) {
            continue; // Skip the starting hole
        }

        if (color == BLUE || ((color == RED) && ((player == PLAYER_1 && (current_hole % 2 != 0)) || (player == PLAYER_2 && (current_hole % 2 == 0))))) {
            board[current_hole][color]++;
            seeds--;

            if ((board[current_hole][RED] + board[current_hole][BLUE] == 2 || board[current_hole][RED] + board[current_hole][BLUE] == 3)) {
                *captured_red += board[current_hole][RED];
                *captured_blue += board[current_hole][BLUE];
                board[current_hole][RED] = 0;
                board[current_hole][BLUE] = 0;
            }
        }
    }
}

int evaluate_board(int board[HOLES][2], int player) {
    int player_seeds = 0;
    int opponent_seeds = 0;

    for (int i = 0; i < HOLES; i++) {
        if ((player == PLAYER_1 && i % 2 == 0) || (player == PLAYER_2 && i % 2 != 0)) {
            player_seeds += board[i][RED] + board[i][BLUE];
        } else {
            opponent_seeds += board[i][RED] + board[i][BLUE];
        }
    }

    return player_seeds - opponent_seeds;
}

int _move_score(int board[HOLES][2], int hole, int color, int player) {
    int board_copy[HOLES][2];
    memcpy(board_copy, board, sizeof(board_copy));
    int captured_red = 0, captured_blue = 0;

    sow_seeds(board_copy, hole, color, player, &captured_red, &captured_blue);

    return captured_red + captured_blue;
}

int minimax(int board[HOLES][2], int depth, int alpha, int beta, int maximizing_player, int player, int *best_hole, int *best_color, clock_t start_time) {
    if (((double)(clock() - start_time) / CLOCKS_PER_SEC) >= MAX_TIME) {
        return evaluate_board(board, player);
    }

    if (depth == 0) {
        return evaluate_board(board, player);
    }

    unsigned long long hash = compute_hash(board, player);
    TranspositionTableEntry *entry = &transposition_table[hash % MAX_TRANSPOSITION_TABLE_SIZE];
    if (entry->hash == hash && entry->depth >= depth) {
        if (entry->flag == FLAG_EXACT) {
            if (best_hole && best_color) {
                *best_hole = entry->best_hole;
                *best_color = entry->best_color;
            }
            return entry->value;
        } else if (entry->flag == FLAG_LOWERBOUND) {
            alpha = (alpha > entry->value) ? alpha : entry->value;
        } else if (entry->flag == FLAG_UPPERBOUND) {
            beta = (beta < entry->value) ? beta : entry->value;
        }
        if (alpha >= beta) {
            return entry->value;
        }
    }

    int value;
    int best_local_hole = -1, best_local_color = -1;
    int moves[HOLES * 2], move_count = 0;

    for (int i = 0; i < HOLES; i++) {
        for (int color = 0; color <= 1; color++) {
            if (is_valid_move(board, i, player, color)) {
                moves[move_count++] = i * 2 + color;
            }
        }
    }

    for (int i = 0; i < move_count - 1; i++) {
        for (int j = i + 1; j < move_count; j++) {
            int move1 = moves[i];
            int move2 = moves[j];
            int score1 = _move_score(board, move1 / 2, move1 % 2, player);
            int score2 = _move_score(board, move2 / 2, move2 % 2, player);
            if (score2 > score1) {
                int temp = moves[i];
                moves[i] = moves[j];
                moves[j] = temp;
            }
        }
    }

    if (maximizing_player) {
        value = INT_MIN;
        for (int i = 0; i < move_count; i++) {
            int hole = moves[i] / 2;
            int color = moves[i] % 2;

            int board_copy[HOLES][2];
            memcpy(board_copy, board, sizeof(board_copy));
            int captured_red = 0, captured_blue = 0;

            sow_seeds(board_copy, hole, color, player, &captured_red, &captured_blue);
            int child_value = minimax(board_copy, depth - 1, alpha, beta, 0, player, NULL, NULL, start_time);

            if (child_value > value) {
                value = child_value;
                best_local_hole = hole;
                best_local_color = color;
            }

            alpha = (alpha > value) ? alpha : value;
            if (alpha >= beta) {
                break;
            }
        }
    } else {
        value = INT_MAX;
        for (int i = 0; i < move_count; i++) {
            int hole = moves[i] / 2;
            int color = moves[i] % 2;

            int board_copy[HOLES][2];
            memcpy(board_copy, board, sizeof(board_copy));
            int captured_red = 0, captured_blue = 0;

            sow_seeds(board_copy, hole, color, player, &captured_red, &captured_blue);
            int child_value = minimax(board_copy, depth - 1, alpha, beta, 1, player, NULL, NULL, start_time);

            if (child_value < value) {
                value = child_value;
                best_local_hole = hole;
                best_local_color = color;
            }

            beta = (beta < value) ? beta : value;
            if (alpha >= beta) {
                break;
            }
        }
    }

    if (best_hole && best_color) {
        *best_hole = best_local_hole;
        *best_color = best_local_color;
    }

    entry->hash = hash;
    entry->depth = depth;
    entry->value = value;
    entry->best_hole = best_local_hole;
    entry->best_color = best_local_color;
    entry->flag = (value <= alpha) ? FLAG_UPPERBOUND : (value >= beta) ? FLAG_LOWERBOUND : FLAG_EXACT;

    return value;
}

int check_game_end(int captured_red_1, int captured_blue_1, int captured_red_2, int captured_blue_2) {
    int total_captured = captured_red_1 + captured_blue_1 + captured_red_2 + captured_blue_2;
    if (captured_red_1 + captured_blue_1 >= 33 || captured_red_2 + captured_blue_2 >= 33 || total_captured >= HOLES * SEEDS_PER_HOLE - 8) {
        return 1;
    }
    return 0;
}

unsigned long long compute_hash(int board[HOLES][2], int player) {
    unsigned long long hash = player;
    for (int i = 0; i < HOLES; i++) {
        hash = hash * 31 + board[i][RED];
        hash = hash * 31 + board[i][BLUE];
    }
    return hash % MAX_TRANSPOSITION_TABLE_SIZE;
}
