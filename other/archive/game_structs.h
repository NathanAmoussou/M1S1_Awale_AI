#ifndef GAME_STRUCTS_H
#define GAME_STRUCTS_H

#include <stdint.h>

#define BOARD_SIZE 16
#define MAX_MOVES 32

typedef struct {
    int8_t red;
    int8_t blue;
} Hole;

typedef struct {
    int hole;
    int color;
} Move;

typedef struct {
    Hole board[BOARD_SIZE];
    int16_t scores[2];
    int current_player;
} GameState;

typedef struct {
    double value;
    Move move;
} MoveEvaluation;

#endif // GAME_STRUCTS_H
