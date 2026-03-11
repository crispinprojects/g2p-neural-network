/*
 *
 * Copyright 2026 Alan Crispin <crispinalan@gmail.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: GNU Lesser General Public License v2.1
 */

#include "neuralnetwork.h"
#include <cmath>
#include <fstream>
#include <qdebug.h>

NeuralNetwork::NeuralNetwork() {
    num_inputs = NUM_INPUTS; // Now 189
    num_hidden = NUM_HIDDEN;
    num_outputs = NUM_PHONEMES; // Now 45

    inputs.resize(num_inputs, 0.0);
    hidden.resize(num_hidden, 0.0);
    output.resize(num_outputs, 0.0);

    initializeWeights();
}

/**
 * @brief initialization initial weights
 */
void NeuralNetwork::initializeWeights() {
    srand(time(NULL));

    // Resize weights and biases
    weights_ih.assign(num_inputs, std::vector<double>(num_hidden));
    weights_ho.assign(num_hidden, std::vector<double>(num_outputs));
    bias_h.assign(num_hidden, 0.0);
    bias_o.assign(num_outputs, 0.0);

    // Resize Velocity Vectors for Momentum
    vel_ih.assign(num_inputs, std::vector<double>(num_hidden, 0.0));
    vel_ho.assign(num_hidden, std::vector<double>(num_outputs, 0.0));
    vel_bh.assign(num_hidden, 0.0);
    vel_bo.assign(num_outputs, 0.0);

    // Initialize weights with small random values
    for (int i = 0; i < num_inputs; i++)
        for (int j = 0; j < num_hidden; j++)
            weights_ih[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;

    for (int i = 0; i < num_hidden; i++)
        for (int j = 0; j < num_outputs; j++)
            weights_ho[i][j] = ((double)rand() / RAND_MAX) * 0.1 - 0.05;
}

/**
 * @brief Performs Backpropagation to update weights based on error.
 * @param input The current window of characters.
 * @param target The expected "correct" phoneme (One-Hot).
 */
void NeuralNetwork::train_step(const std::vector<double>& input, const std::vector<double>& target) {
    predict(input); // Forward Pass

    // 1. Output Layer Gradients (How much did each output neuron contribute to the error?)
    std::vector<double> output_gradients(num_outputs);
    for (int j = 0; j < num_outputs; j++) {
        double error = target[j] - output[j];
        output_gradients[j] = error * sigmoid_derivative(output[j]);
    }

    // 2. Hidden Layer Gradients (Back-propagating the error through the weights)
    std::vector<double> hidden_gradients(num_hidden);
    for (int i = 0; i < num_hidden; i++) {
        double error = 0.0;
        for (int j = 0; j < num_outputs; j++) {
            error += output_gradients[j] * weights_ho[i][j];
        }
        hidden_gradients[i] = error * sigmoid_derivative(hidden[i]);
    }


    // 3. Update Hidden-to-Output Weights & Biases (Corrected Indices)
    for (int j = 0; j < num_outputs; j++) {
        for (int i = 0; i < num_hidden; i++) {
            double gradient = output_gradients[j] * hidden[i];
            vel_ho[i][j] = (MOMENTUM * vel_ho[i][j]) + (LEARNING_RATE * gradient);
            weights_ho[i][j] += vel_ho[i][j];
        }
        // Update bias outside the inner loop, using index j
        vel_bo[j] = (MOMENTUM * vel_bo[j]) + (LEARNING_RATE * output_gradients[j]);
        bias_o[j] += vel_bo[j];
    }

    // 4. Update Input-to-Hidden Weights & Biases (Corrected Indices)
    for (int j = 0; j < num_hidden; j++) {
        for (int i = 0; i < num_inputs; i++) {
            double gradient = hidden_gradients[j] * inputs[i];
            vel_ih[i][j] = (MOMENTUM * vel_ih[i][j]) + (LEARNING_RATE * gradient);
            weights_ih[i][j] += vel_ih[i][j];
        }
        vel_bh[j] = (MOMENTUM * vel_bh[j]) + (LEARNING_RATE * hidden_gradients[j]);
        bias_h[j] += vel_bh[j];
    }
}


void NeuralNetwork::predict(const std::vector<double>& input) {
    // Copy input to network inputs
    for (int i = 0; i < num_inputs && i < input.size(); i++) {
        inputs[i] = input[i];
    }

    // Calculate hidden layer activation
    for (int j = 0; j < num_hidden; j++) {
        double activation = 0.0;
        for (int i = 0; i < num_inputs; i++) {
            activation += inputs[i] * weights_ih[i][j];
        }
        activation += bias_h[j];
        hidden[j] = sigmoid(activation);
    }

    // Calculate output layer activation
    for (int j = 0; j < num_outputs; j++) {
        double output_activation = 0.0;
        for (int i = 0; i < num_hidden; i++) {
            output_activation += hidden[i] * weights_ho[i][j];
        }
        output_activation += bias_o[j];
        output[j] = sigmoid(output_activation);
    }
}


// Output Layer Activation: Sigmoid (Keeps values 0-1 for confidence)
double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Output Layer Derivative (Standard sigmoid math)
double NeuralNetwork::sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

/**
 * @brief Saves the current weights and biases to a binary file.
 */
bool NeuralNetwork::saveWeights(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) return false;

    for (auto& row : weights_ih) out.write((char*)row.data(), row.size() * sizeof(double));
    for (auto& row : weights_ho) out.write((char*)row.data(), row.size() * sizeof(double));
    out.write((char*)bias_h.data(), bias_h.size() * sizeof(double));
    out.write((char*)bias_o.data(), bias_o.size() * sizeof(double));
    return true;
}
/**
 * @brief Loads weights from a file so training doesn't need to be repeated.
 */
bool NeuralNetwork::loadWeights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) return false;

    for (auto& row : weights_ih) in.read((char*)row.data(), row.size() * sizeof(double));
    for (auto& row : weights_ho) in.read((char*)row.data(), row.size() * sizeof(double));
    in.read((char*)bias_h.data(), bias_h.size() * sizeof(double));
    in.read((char*)bias_o.data(), bias_o.size() * sizeof(double));
    return true;
}


