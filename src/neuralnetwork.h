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
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <cstdlib>
#include <ctime>
#include <vector>
#include<QString>

#define WINDOW_SIZE 7       // 3 left + 1 target + 3 right context
#define NUM_CHARS 27        // a-z (26) + space (1)
#define NUM_INPUTS (WINDOW_SIZE * NUM_CHARS) // 189 total inputs
#define NUM_PHONEMES 45     // Matches your m_phonemeList size
#define NUM_HIDDEN 200      // Slightly more hidden neurons for the larger input
#define LEARNING_RATE 0.001    // Lowered from 0.01 to prevent explosion
#define MOMENTUM 0.9           // Keep this

class NeuralNetwork
{
public:
    NeuralNetwork();
    void initializeWeights();  // Initialize weights randomly
    void predict(const std::vector<double>& input);  // Predict phonemes
    void train_step(const std::vector<double>& input, const std::vector<double>& target);  // Train one step
    double sigmoid(double x);
    double sigmoid_derivative(double x);

    double leaky_relu(double x);
    double leaky_relu_derivative(double x);


    // Public member variables to access outputs
    std::vector<double> inputs;      // Input vector (character codes)
    std::vector<double> hidden;      // Hidden layer activations
    std::vector<double> output;      // Output vector (phoneme probabilities)

    // Weights and biases
    std::vector<std::vector<double>> weights_ih;  // Input to hidden weights
    std::vector<std::vector<double>> weights_ho;  // Hidden to output weights
    std::vector<double> bias_h;                   // Hidden layer biases
    std::vector<double> bias_o;                   // Output layer biases

    bool saveWeights(const std::string& filename);
    bool loadWeights(const std::string& filename);

private:
    int num_inputs;
    int num_hidden;
    int num_outputs;

    // Add these to store velocity
    std::vector<std::vector<double>> vel_ih;
    std::vector<std::vector<double>> vel_ho;
    std::vector<double> vel_bh;
    std::vector<double> vel_bo;



};

#endif // NEURALNETWORK_H

