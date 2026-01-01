//
// Created by Krish Patel on 4/03/25.
//

#pragma once

#include <vector>
#include "Neuron.hpp"

class Layer {
public:
    Layer(int n_neurons, int n_weights);

    ~Layer();

    // Return mutable reference to the neurons
    std::vector<Neuron> &get_neurons();

    // Return const reference to the neurons
    const std::vector<Neuron> &get_neurons() const;

private:
    void initNeurons(int n_neurons, int n_weights);

    std::vector<Neuron> m_neurons;
};