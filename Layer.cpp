//
// Created by Krish Patel on 4/05/25.
//
#include "Layer.h"

Layer::Layer(int n_neurons, int n_weights) {
    initNeurons(n_neurons, n_weights);
}

Layer::~Layer() {
}

void Layer::initNeurons(int n_neurons, int n_weights) {
    for (int n = 0; n < n_neurons; n++) {
        m_neurons.push_back(Neuron(n_weights));
    }
}


std::vector<Neuron> &Layer::get_neurons() {
    return m_neurons;
}

const std::vector<Neuron> &Layer::get_neurons() const {
    return m_neurons;
}
