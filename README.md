## GitHub Repository
You can find the complete project on [GitHub](https://github.com/rammohan2004/da6401_assignment_1).

## Wandb Report Link
You can find the Report on [WandB Report](https://wandb.ai/cs25m017-indian-institute-of-technology-madras/da6401-assignment1/reports/DA6401-Assignment-CS25M017--VmlldzoxNjEyNDA1OQ?accessToken=8n6803hd0pvpllguqe0tjlc4e7lmqe0yq0ffswigxuf4g9n0vginjocmu3tdr0n8).

# Feedforward Neural Network with Various Optimizers

## Overview
This project implements a configurable, modular Multi-Layer Perceptron (MLP) using only NumPy. The project explores fundamental Deep Learning concepts, including forward propagation, backpropagation, and various optimization strategies to classify the MNIST and Fashion-MNIST datasets.

## Features
- Implements a customizable Feedforward Neural Network relying exclusively on NumPy for mathematical operations.
- Supports activation functions: sigmoid, tanh, relu.
- Supports loss functions: mean_squared_error, cross_entropy.
- Includes multiple optimizers:
  - Stochastic Gradient Descent (sgd) 
  - Momentum-based SGD (momentum) 
  - Nesterov Accelerated Gradient (nag)
  - RMSprop (rmsprop)
- Uses Weights & Biases (wandb) for experiment tracking.
- Evaluates models using Accuracy, Precision, Recall, and F1-score.

## Repository Hierarchy
    da6401_assignment_1/
    |-- src/
    |   |-- best_config.json
    |   |-- best_model.npy
    |   |-- train.py
    |   |-- inference.py
    |   |-- ann/
    |       |-- activations.py
    |       |-- neural_layer.py
    |       |-- neural_network.py
    |       |-- objective_function.py
    |       |-- optimizrs.py
    |-- README.md
    |-- requirements.txt

## Usage
### Running the Training Script
Execute the training script with default parameters using the command line interface:
    python src/train.py

### Command-line Arguments
You can customize the training process using the following mandatory arguments:
     python src/train.py -d mnist -e 15 -b 16 -l cross_entropy -o rmsprop -lr 0.0001 -wd 0.0 -nhl 3 -sz 128 64 32 -a tanh -w_i xavier

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
- The best model weights saved as best_model.npy based on test F-1 score
- The optimal configuration saved as best_config.json representing your optimized weights and the model configuration
- Both the model and config files are placed in the src folder


