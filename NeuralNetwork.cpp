#include "NeuralNetwork.hpp"
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>

#include "Layer.hpp"

Network::Network() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    m_nLayers = 0;
}

Network::~Network() {
}

void Network::initialize_network(int n_inputs, int n_hidden, int n_outputs) {
    //  hidden layer
    add_layer(n_hidden, n_inputs + 1);

    //  output layer
    add_layer(n_outputs, n_hidden + 1);
}

void Network::add_layer(int n_neurons, int n_weights) {
    m_layers.push_back(Layer(n_neurons, n_weights));
    m_nLayers++;
}

std::vector<float> Network::forward_propagate(std::vector<float> inputs) {
    std::vector<float> new_inputs;
    for (size_t i = 0; i < m_nLayers; i++) {
        new_inputs.clear();
        std::vector<Neuron> &layer_neurons = m_layers[i].get_neurons();

        for (size_t n = 0; n < layer_neurons.size(); n++) {
            layer_neurons[n].activate(inputs);
            layer_neurons[n].transfer();
            new_inputs.push_back(layer_neurons[n].get_output());
        }
        inputs = new_inputs;
    }
    return inputs;
}

void Network::backward_propagate_error(std::vector<float> expected) {
    for (size_t i = m_nLayers; i-- > 0;) {
        std::vector<Neuron> &layer_neurons = m_layers[i].get_neurons();
        for (size_t n = 0; n < layer_neurons.size(); n++) {
            float error = 0.0;
            if (i == m_nLayers - 1) {
                error = expected[n] - layer_neurons[n].get_output();
            } else {
                for (auto &neu: m_layers[i + 1].get_neurons()) {
                    error += neu.get_weights()[n] * neu.get_delta();
                }
            }
            layer_neurons[n].set_delta(error * layer_neurons[n].transfer_derivative());
        }
    }
}

void Network::update_weights(std::vector<float> inputs, float l_rate) {
    for (size_t i = 0; i < m_nLayers; i++) {
        std::vector<float> new_inputs;
        if (i != 0) {
            // take outputs from prev layer
            for (auto &neuron: m_layers[i - 1].get_neurons()) {
                new_inputs.push_back(neuron.get_output());
            }
        } else {
            // just use original input for first layer
            new_inputs = std::vector<float>(inputs.begin(), inputs.end() - 1);
        }

        std::vector<Neuron> &layer_neurons = m_layers[i].get_neurons();
        for (auto &neuron: layer_neurons) {
            std::vector<float> &weights = neuron.get_weights();

            // updates weights
            for (size_t j = 0; j < new_inputs.size(); j++) {
                weights[j] += l_rate * neuron.get_delta() * new_inputs[j];
            }
            // update biases
            weights.back() += l_rate * neuron.get_delta();
        }
    }
}

void Network::train(std::vector<std::vector<float>> trainings_data, float l_rate, size_t n_epoch, size_t n_outputs) {
    for (size_t e = 0; e < n_epoch; e++) {
        float sum_error = 0.0;
        for (const auto &row: trainings_data) {
            std::vector<float> outputs = forward_propagate(row);
            std::vector<float> expected(n_outputs, 0.0);
            expected[static_cast<int>(row.back())] = 1.0f;

            for (size_t x = 0; x < n_outputs; x++) {
                sum_error += static_cast<float>(std::pow(expected[x] - outputs[x], 2));
            }

            backward_propagate_error(expected);
            update_weights(row, l_rate);
        }
        std::cout << "[>] epoch=" << e << ", l_rate=" << l_rate << ", error=" << sum_error << std::endl;
    }
}

int Network::predict(std::vector<float> input) {
    std::vector<float> outputs = forward_propagate(input);
    return static_cast<int>(std::max_element(outputs.begin(), outputs.end()) - outputs.begin());
}

void Network::display_human() {
    std::cout << "[Network] (Layers: " << m_nLayers << ")" << std::endl;
    std::cout << "{" << std::endl;

    for (size_t l = 0; l < m_layers.size(); l++) {
        const auto &layer = m_layers[l];
        std::cout << "\t (Layer " << l << "): {";

        for (size_t i = 0; i < layer.get_neurons().size(); i++) {
            Neuron neuron = layer.get_neurons()[i];
            std::cout << "<(Neuron " << i << "): [ weights={";
            const auto &weights = neuron.get_weights();
            for (size_t w = 0; w < weights.size(); w++) {
                std::cout << weights[w];
                if (w < weights.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "}, output=" << neuron.get_output()
                      << ", activation=" << neuron.get_activation()
                      << ", delta=" << neuron.get_delta() << "]>";
            if (i < layer.get_neurons().size() - 1) {
                std::cout << ", ";
            }
        }

        std::cout << "}";
        if (l < m_layers.size() - 1) {
            std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "}" << std::endl;
}