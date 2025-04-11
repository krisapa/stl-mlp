//
// Created by Krish Patel on 4/10/25.
//
#include <iostream>
#include <vector>
#include <set>
#include <string>
#include <fstream>
#include <regex>
#include <iterator>
#include <map>
#include <numeric>
#include <cmath>
#include "NeuralNetwork.h"

std::vector<std::vector<float>> load_csv_data(std::string filename);

std::vector<float>
evaluate_network(std::vector<std::vector<float>> dataset, int n_folds, float l_rate, int n_epoch, int n_hidden);

float accuracy_metric(std::vector<int> expect, std::vector<int> predict);


/*
* Loads csv-dataset and normalizes the data. Then, the network is initialized, trained, and tested using cross validation.
* Feel free to play around with the folds, learning rate, epochs and hidden neurons.
* 
* (See at the bottom for a second main function that's for displaying and testing a very small network.)
*/
int main(int argc, char *argv[]) {
    std::cout << "Neural Network with Backpropagation in C++ from scratch" << std::endl;

    std::vector<std::vector<float>> csv_data;
    csv_data = load_csv_data("dataset.csv");

//	Normalize last col for one-hot encoding
    std::map<int, int> lookup = {};
    int index = 0;
    for (auto &vec: csv_data) {
        std::pair<std::map<int, int>::iterator, bool> ret;
        ret = lookup.insert(std::pair<int, int>(static_cast<int>(vec.back()), index));
        vec.back() = static_cast<float>(ret.first->second);
        if (ret.second) {
            index++;
        }
    }

    int n_folds = 5;        // fold count
    float l_rate = 0.2f;    // learning rate
    int n_epoch = 600;        // epochs
    int n_hidden = 8;        // number of neurons in first layer

//	test the network
    std::vector<float> scores = evaluate_network(csv_data, n_folds, l_rate, n_epoch, n_hidden);

    // calculate the mean average of the scores across each cross validation
    float mean = std::accumulate(scores.begin(), scores.end(), decltype(scores)::value_type(0)) /
                 static_cast<float>(scores.size());

    std::cout << "Mean accuracy: " << mean << std::endl;

    return 0;
}

std::vector<float>
evaluate_network(std::vector<std::vector<float>> dataset, int n_folds, float l_rate, int n_epoch, int n_hidden) {
//	Split into folds
    std::vector<std::vector<std::vector<float>>> dataset_splits;
    // init prng
    std::srand(static_cast<unsigned int>(std::time(nullptr)));

    std::vector<float> scores;

    size_t fold_size = static_cast<unsigned int>(dataset.size() / n_folds);
    for (int f = 0; f < n_folds; f++) {
        std::vector<std::vector<float>> fold;
        while (fold.size() < fold_size) {
            int n = rand() % dataset.size();

            // add the element to the fold and remove from data set
            std::swap(dataset[n], dataset.back());
            fold.push_back(dataset.back());
            dataset.pop_back();
        }

        dataset_splits.push_back(fold);
    }

    // choose one fold as test and use the rest as training data
    for (size_t i = 0; i < dataset_splits.size(); i++) {
        std::vector<std::vector<std::vector<float>>> train_sets = dataset_splits;
        std::swap(train_sets[i], train_sets.back());
        std::vector<std::vector<float>> test_set = train_sets.back();
        train_sets.pop_back();

        // merge training sets
        std::vector<std::vector<float>> train_set;
        for (auto &s: train_sets) {
            for (auto &row: s) {
                train_set.push_back(row);
            }
        }

        // Store expected results
        std::vector<int> expected;
        for (auto &row: test_set) {
            expected.push_back(static_cast<int>(row.back()));
            // check result isnt already in the training set
            row.back() = 42;
        }

        std::vector<int> predicted;

        std::set<float> results;
        for (const auto &r: train_set) {
            results.insert(r.back());
        }
        int n_outputs = results.size();
        int n_inputs = train_set[0].size() - 1;

        // Backpropagation w/ stochastic gradient descent
        Network *network = new Network();
        network->initialize_network(n_inputs, n_hidden, n_outputs);
        network->train(train_set, l_rate, n_epoch, n_outputs);

        for (const auto &row: test_set) {
            predicted.push_back(network->predict(row));
        }

        scores.push_back(accuracy_metric(expected, predicted));
    }

    return scores;
}


float accuracy_metric(std::vector<int> expect, std::vector<int> predict) {
    int correct = 0;

    for (size_t i = 0; i < predict.size(); i++) {
        if (predict[i] == expect[i]) {
            correct++;
        }
    }
    return static_cast<float>(correct * 100.0f / predict.size());
}


std::vector<std::vector<float>> load_csv_data(std::string filename) {
    const std::regex comma(",");

    std::ifstream csv_file(filename);

    std::vector<std::vector<float>> data;

    std::string line;

    std::vector<float> mins;
    std::vector<float> maxs;
    bool first = true;

    while (csv_file && std::getline(csv_file, line)) {
        std::vector<std::string> srow{std::sregex_token_iterator(line.begin(), line.end(), comma, -1),
                                      std::sregex_token_iterator()};
        std::vector<float> row(srow.size());
        std::transform(srow.begin(), srow.end(), row.begin(), [](std::string const &val) { return std::stof(val); });

        if (first) {
            mins = row;
            maxs = row;
            first = false;
        } else {
            for (size_t t = 0; t < row.size(); t++) {
                if (row[t] > maxs[t]) {
                    maxs[t] = row[t];
                } else if (row[t] < mins[t]) {
                    mins[t] = row[t];
                }
            }
        }

        data.push_back(row);
    }

    for (auto &vec: data) {
        for (size_t i = 0; i < vec.size() - 1; i++) {
            vec[i] = (vec[i] - mins[i]) / (maxs[i] - mins[i]);
        }
    }

    return data;
}
