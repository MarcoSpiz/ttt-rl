#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "neural_network.h"
#include "game.h"

/* Play one game of Tic Tac Toe against the neural network. */
void play_game(NeuralNetwork *nn) {
    GameState state;
    char winner;
    int move_history[9]; // Maximum 9 moves in a game.
    int num_moves = 0;

    init_game(&state);

    printf("Welcome to Tic Tac Toe! You are X, the computer is O.\n");
    printf("Enter positions as numbers from 0 to 8 (see picture).\n");

    while (!check_game_over(&state, &winner)) {
        display_board(&state);

        if (state.current_player == 0) {
            // Human turn.
            int move;
            char movec;
            printf("Your move (0-8): ");
            scanf(" %c", &movec);
            move = movec-'0'; // Turn character into number.

            // Check if move is valid.
            if (move < 0 || move > 8 || state.board[move] != '.') {
                printf("Invalid move! Try again.\n");
                continue;
            }

            state.board[move] = 'X';
            move_history[num_moves++] = move;
        } else {
            // Computer's turn
            printf("Computer's move:\n");
            int move = get_computer_move(&state, nn, 1);
            state.board[move] = 'O';
            printf("Computer placed O at position %d\n", move);
            move_history[num_moves++] = move;
        }

        state.current_player = !state.current_player;
    }

    display_board(&state);

    if (winner == 'X') {
        printf("You win!\n");
    } else if (winner == 'O') {
        printf("Computer wins!\n");
    } else {
        printf("It's a tie!\n");
    }

    // Learn from this game
    learn_from_game(nn, move_history, num_moves, 1, winner);
}

int main(int argc, char **argv) {
    int random_games = 150000; // Fast and enough to play in a decent way.

    if (argc > 1) random_games = atoi(argv[1]);
    srand(time(NULL));

    // Initialize neural network.
    NeuralNetwork nn;
    // Load pretrained model or training again
    setup_neural_network(&nn,random_games);

    // Play game with human and learn more.
    while(1) {
        char play_again;
        play_game(&nn);

        printf("Play again? (y/n): ");
        scanf(" %c", &play_again);
        if (play_again != 'y' && play_again != 'Y') break;
    }
    return 0;
}
