# Minimal Neural Network From Scratch in C++

This repository has a very simple feedforward neural network built using the C++ STL, no external ML frameworks. I've been wanting to learn more about neural networks at the fundamental level and managing the math, activation functions, and training loop manually has been fun. Although very minimal compared to production frameworks, this implementation was a lot of fun and performs pretty well.

## How It Works

- **Feedforward Structure**  
  The network accepts a set of normalized feature inputs, processes them through one or more hidden layers (sigmoid activation by default), and produces a classification output.
- **Backpropagation**  
  During training, the network calculates the error between predicted outputs and the true labels, then propagates this error backward to update weights (stochastic gradient descent).
- **Training Flow**  
  1. **Initialize weights** randomly.  
  2. **Forward pass**: compute outputs layer by layer.  
  3. **Compute error** against the target.  
  4. **Backward pass**: adjust weights based on gradients.  
  5. Repeat for multiple epochs until the model converges or reaches a desired accuracy.

## Usage

1. **Clone and Build**  
   ```bash
   git clone https://github.com/6b70/Neural-Network-from-Scratch.git
   cd Neural-Network-from-Scratch
   make run
   ```
   This compiles the code and runs the executable.

2. **Adjust Parameters**  
   Inside `main.cpp`, you can tweak the number of hidden neurons, epochs, or learning rate. For instance:
   - **Hidden Neurons**: 8  
   - **Epochs**: 600  
   - **Learning Rate**: 0.2

3. **Output**  
   The program displays logs for each epoch, including the accumulated error. After training, it reports final accuracy on the included dataset.

## Example

With 8 neurons in the hidden layer, 600 epochs, and a 0.2 learning rate, the network has about 94% accuracy on the sample dataset in the repository:

<img src="https://github.com/user-attachments/assets/3bc06db7-b930-4bd6-81e1-fed768de4292" alt="Training Output" width="450" />

