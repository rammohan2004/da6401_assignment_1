## GitHub Repository
You can find the complete project on [GitHub](https://github.com/rajkumarseelam/DA6401-Introduction-to-Deep-Learning.git).

## Wandb Report Link
You can find the Report on [WandB Report](https://wandb.ai/cs24m042-iit-madras-foundation/DA6401-Assignment-1/reports/DA6401-Assignment-1--VmlldzoxMTgyMDMzNQ).

# Feedforward Neural Network with Various Optimizers

## Overview
[cite_start]This project implements a configurable, modular Multi-Layer Perceptron (MLP) using only NumPy[cite: 24]. [cite_start]The project explores fundamental Deep Learning concepts, including forward propagation, backpropagation, and various optimization strategies to classify the MNIST and Fashion-MNIST datasets[cite: 25].

## Features
- [cite_start]Implements a customizable Feedforward Neural Network relying exclusively on NumPy for mathematical operations[cite: 26].
- [cite_start]Supports activation functions: sigmoid, tanh, relu[cite: 49].
- [cite_start]Supports loss functions: mean_squared_error, cross_entropy[cite: 44].
- Includes multiple optimizers:
  - [cite_start]Stochastic Gradient Descent (sgd) [cite: 43]
  - [cite_start]Momentum-based SGD (momentum) [cite: 43]
  - [cite_start]Nesterov Accelerated Gradient (nag) [cite: 43]
  - [cite_start]RMSprop (rmsprop) [cite: 43]
- [cite_start]Uses Weights & Biases (wandb) for experiment tracking[cite: 28].
- [cite_start]Evaluates models using Accuracy, Precision, Recall, and F1-score[cite: 55].

## Repository Hierarchy
    DA6401-Introduction-to-Deep-Learning/
    |-- src/
    |   |-- best_config.json
    |   |-- best_model.npy
    |   |-- train.py
    |   |-- inference.py
    |   |-- ann/
    |       |-- neural_network.py
    |-- README.md

## Usage
### Running the Training Script
Execute the training script with default parameters using the command line interface:
    python src/train.py

### Command-line Arguments
You can customize the training process using the following mandatory arguments:
     python train.py -d mnist -e 15 -b 16 -l cross_entropy -o rmsprop -lr 0.0001 -wd 0.0 -nhl 3 -sz 128 64 32 -a tanh -w_i xavier

### Argument Details
- -d, --dataset: Choose between mnist and fashion_mnist.
- -e, --epochs: Number of training epochs.
- -b, --batch_size: Mini-batch size.
- -o, --optimizer: Select an optimizer from sgd, momentum, nag, rmsprop.
- -lr, --learning_rate: Initial learning rate.
- -a, --activation: Choice of sigmoid, tanh, relu for every hidden layer.
- -l, --loss: Choice of mean_squared_error or cross_entropy.
- -nhl, --num_layers: Number of hidden layers.
- -sz, --hidden_size: Number of neurons in each hidden layer (list of values).
- -wi, --weight_init: Choice of random or xavier.
- -wd, --weight_decay: Weight decay for L2 regularization.
- -w_p, --wandb_project: Weights and Biases Project ID.

## Output
- 
- The best model weights saved as best_model.npy based on test F-1 score
- The optimal configuration saved as best_config.json representing your optimized weights and the model configuration
- Both the model and config files are placed in the src folder


