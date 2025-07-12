//
// Created by Marco  on 11/07/25.
//

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H


#include "game.h"


#define NN_INPUT_SIZE 18
#define NN_HIDDEN_SIZE 100
#define NN_OUTPUT_SIZE 9
#define LEARNING_RATE 0.1



typedef struct {
    // Weights and biases.
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
    float biases_h[NN_HIDDEN_SIZE];
    float biases_o[NN_OUTPUT_SIZE];

    // Activations are part of the structure itself for simplicity.
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float raw_logits[NN_OUTPUT_SIZE]; // Outputs before softmax().
    float outputs[NN_OUTPUT_SIZE];    // Outputs after softmax().
} NeuralNetwork;

void setup_neural_network(NeuralNetwork* nn, int random_games) ;
int get_computer_move(GameState *state, NeuralNetwork *nn, int display_probs);
void learn_from_game(NeuralNetwork *nn, int *move_history, int num_moves, int nn_moves_even, char winner);
#endif //NEURAL_NETWORK_H
