# cython: language_level=3
# distutils: language = c

from libc.stdlib cimport malloc, free
from libc.math cimport INFINITY
from libc.float cimport DBL_MAX
from libc.stdint cimport int8_t, int16_t
from cpython cimport array
import numpy as np
cimport numpy as np

# Define structures as ctypedef instead of extern
ctypedef struct Hole:
    int8_t red
    int8_t blue

ctypedef struct Move:
    int hole
    int color

ctypedef struct GameState:
    Hole[16] board
    int16_t[2] scores
    int current_player

ctypedef struct MoveEvaluation:
    double value
    Move move

cdef class MinimaxAgent:
    cdef:
        public double max_time
        public int nodes_cut

    def __cinit__(self, double max_time=2.0):
        self.max_time = max_time
        self.nodes_cut = 0

    cdef GameState _convert_to_c_state(self, game_state) nogil:
        cdef GameState c_state
        cdef int i

        for i in range(16):
            c_state.board[i].red = game_state.board[i][0]
            c_state.board[i].blue = game_state.board[i][1]

        c_state.scores[0] = game_state.scores[0]
        c_state.scores[1] = game_state.scores[1]
        c_state.current_player = game_state.current_player

        return c_state

    cdef Move _convert_to_c_move(self, tuple py_move) nogil:
        cdef Move c_move
        c_move.hole = py_move[0]
        c_move.color = py_move[1]
        return c_move

    cdef tuple _convert_from_c_move(self, Move c_move) nogil:
        return (c_move.hole, c_move.color)

    cdef double _evaluate(self, GameState* state) nogil:
        return (state.scores[0] - state.scores[1]) * (1 if state.current_player == 1 else -1)

    cdef MoveEvaluation _minimax(self, GameState* state, int depth, double alpha, double beta, bint maximizing) nogil:
        cdef:
            MoveEvaluation result
            double value
            int i, color
            Move move

        # Base case
        if depth == 0:
            result.value = self._evaluate(state)
            result.move.hole = -1
            result.move.color = -1
            return result

        result.value = -DBL_MAX if maximizing else DBL_MAX
        result.move.hole = -1
        result.move.color = -1

        # Simple move generation (placeholder - implement actual game rules)
        for i in range(16):
            for color in range(2):
                move.hole = i
                move.color = color

                # Validate and make move (implement actual game rules)
                # Then recursively evaluate

                value = self._minimax(state, depth - 1, alpha, beta, not maximizing).value

                if maximizing and value > result.value:
                    result.value = value
                    result.move = move
                    alpha = max(alpha, value)
                elif not maximizing and value < result.value:
                    result.value = value
                    result.move = move
                    beta = min(beta, value)

                if beta <= alpha:
                    self.nodes_cut += 1
                    break

        return result

    def get_move(self, game_state):
        cdef:
            GameState c_state
            MoveEvaluation result
            int depth = 1
            double eval_val
            Move best_move

        c_state = self._convert_to_c_state(game_state)

        try:
            result = self._minimax(&c_state, depth, -DBL_MAX, DBL_MAX, True)
            best_move = result.move
            eval_val = result.value
        except:
            return None, None

        return self._convert_from_c_move(best_move), eval_val
