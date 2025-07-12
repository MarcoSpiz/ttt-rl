//
// Created by Marco  on 11/07/25.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "game.h"

void init_game(GameState *state) {
    memset(state->board,'.',9);
    state->current_player = 0;  // Player (X) goes first
}

/* Show board on screen in ASCII "art"... */
void display_board(GameState *state) {
    for (int row = 0; row < 3; row++) {
        // Display the board symbols.
        printf("%c%c%c ", state->board[row*3], state->board[row*3+1],
                          state->board[row*3+2]);

        // Display the position numbers for this row, for the poor human.
        printf("%d%d%d\n", row*3, row*3+1, row*3+2);
    }
    printf("\n");
}

void board_to_inputs(GameState *state, float *inputs) {
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.') {
            inputs[i*2] = 0;
            inputs[i*2+1] = 0;
        } else if (state->board[i] == 'X') {
            inputs[i*2] = 1;
            inputs[i*2+1] = 0;
        } else {  // 'O'
            inputs[i*2] = 0;
            inputs[i*2+1] = 1;
        }
    }
}


int check_game_over(GameState *state, char *winner) {
    for (int i = 0; i < 3; i++) {
        if (state->board[i*3] != '.' &&
            state->board[i*3] == state->board[i*3+1] &&
            state->board[i*3+1] == state->board[i*3+2]) {
            *winner = state->board[i*3];
            return 1;
            }
    }

    // Check columns.
    for (int i = 0; i < 3; i++) {
        if (state->board[i] != '.' &&
            state->board[i] == state->board[i+3] &&
            state->board[i+3] == state->board[i+6]) {
            *winner = state->board[i];
            return 1;
            }
    }

    // Check diagonals.
    if (state->board[0] != '.' &&
        state->board[0] == state->board[4] &&
        state->board[4] == state->board[8]) {
        *winner = state->board[0];
        return 1;
        }
    if (state->board[2] != '.' &&
        state->board[2] == state->board[4] &&
        state->board[4] == state->board[6]) {
        *winner = state->board[2];
        return 1;
        }

    // Check for tie (no free tiles left).
    int empty_tiles = 0;
    for (int i = 0; i < 9; i++) {
        if (state->board[i] == '.') empty_tiles++;
    }
    if (empty_tiles == 0) {
        *winner = 'T';  // Tie
        return 1;
    }

    return 0; // Game continues.
}

int get_random_move(GameState *state) {
    while(1) {
        int move = rand() % 9;
        if (state->board[move] != '.') continue;
        return move;
    }
}