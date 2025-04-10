//
// Created by Krish Patel on 4/02/25.
//

#pragma once

#include <vector>

class Neuron {
public:
    Neuron(int n_weights);

    ~Neuron();

    void activate(std::vector<float> inputs);

    void transfer();

    float transfer_derivative();

    // returns mutable reference to the  weights
    std::vector<float> &get_weights();

    float get_output();

    float get_activation();

    float get_delta();

    void set_delta(float delta);

private:
    size_t m_nWeights;
    std::vector<float> m_weights;
    float m_activation;
    float m_output;
    float m_delta;

private:
    void initWeights(int n_weights);
};