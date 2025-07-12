//
// Created by Marco  on 11/07/25.
//

#ifndef GAME_H
#define GAME_H


// Game board representation.
typedef struct {
    char board[9];          // Can be "." (empty) or "X", "O".
    int current_player;     // 0 for player (X), 1 for computer (O).
} GameState;


void init_game(GameState *state);
void display_board(GameState *state);
void board_to_inputs(GameState *state, float *inputs);
int check_game_over(GameState *state, char *winner);
int get_random_move(GameState *state);
#endif //GAME_H
