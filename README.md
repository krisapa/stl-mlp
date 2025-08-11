# Minimal Neural Network From Scratch in C++

Small MLP implemented with nothing but C++ STL. It was a fun opportunity to practice C++ and work with backprop math directly.


## Build & Run
```bash
make run
```
By default, make run trains on the UCI seeds dataset in this repo. Tweak hyperparams in `main.cpp`.

## What’s here
- Fully-connected MLP, forward + backprop written by hand
- Only using the C++ STL: `std::vector`, `std::map`, `std::regex` (for CSV ingest), `<cmath>`, `<random>`, etc. No third-party dependencies like Eigen or BLAS.
- Simple SGD training loop

## Result
On the included dataset: ~94% accuracy with 8 hidden units, 600 epochs, LR=0.2.

<img src="https://github.com/user-attachments/assets/3bc06db7-b930-4bd6-81e1-fed768de4292" alt="Training Output" width="450" />

