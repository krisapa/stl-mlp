//
// Created by Krish Patel on 4/03/25.
//
#include "Neuron.hpp"
#include <cstdlib>
#include <cmath>
#include <ctime>

Neuron::Neuron(int n_weights) {
    initWeights(n_weights);
    m_nWeights = n_weights;
    m_activation = 0;
    m_output = 0;
    m_delta = 0;
}

Neuron::~Neuron() {
}

void Neuron::initWeights(int n_weights) {
    for (int w = 0; w < n_weights; w++) {
        m_weights.push_back(static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX));
    }
}

void Neuron::activate(std::vector<float> inputs) {
    m_activation = m_weights[m_nWeights - 1];

    for (size_t i = 0; i < m_nWeights - 1; i++) {
        m_activation += m_weights[i] * inputs[i];
    }
}

void Neuron::transfer() {
    m_output = 1.0f / (1.0f + std::exp(-m_activation));
}

float Neuron::transfer_derivative() {
    return static_cast<float>(m_output * (1.0f - m_output));
}

std::vector<float> &Neuron::get_weights() {
    return m_weights;
}

float Neuron::get_output() {
    return m_output;
}

float Neuron::get_activation() {
    return m_activation;
}

float Neuron::get_delta() {
    return m_delta;
}

void Neuron::set_delta(float delta) {
    m_delta = delta;
}