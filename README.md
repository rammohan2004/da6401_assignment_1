## GitHub Repository

You can find the complete project on [GitHub](https://github.com/rajkumarseelam/DA6401-Introduction-to-Deep-Learning.git).

## Wandb Report Link

You can find the Report on [WandB Report](https://wandb.ai/cs24m042-iit-madras-foundation/DA6401-Assignment-1/reports/DA6401-Assignment-1--VmlldzoxMTgyMDMzNQ).

# Feedforward Neural Network with Various Optimizers

## Overview

This project implements a feedforward neural network from scratch using NumPy. The network supports multiple hidden layers, various weight initialization techniques, activation functions, and loss functions. It also integrates multiple optimization techniques for training. The model can be trained on the MNIST and Fashion-MNIST datasets.

## Features

* Implements a customizable Feedforward Neural Network using only NumPy.


* Supports activation functions: Sigmoid, Tanh, ReLU.


* Supports loss functions: Mean Squared Error, Cross Entropy.


* Includes multiple optimizers:
* Stochastic Gradient Descent (SGD) 


* Momentum-based SGD 


* Nesterov Accelerated Gradient (NAG) 


* RMSprop 




* Uses Weights & Biases (WandB) for experiment tracking.


* Generates confusion matrix plots using scikit-learn.



## Code Organization

The project follows the required GitHub skeleton. The main execution scripts are `train.py` and `inference.py`, which use `argparse` for configuration.

Creating a model object and initializing the network is handled via the parsed arguments:

```python
from src.ann.neural_network import NeuralNetwork
model = NeuralNetwork(args)

```

The training process can be executed directly by passing the data:

```python
history = model.train(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size)

```

For generating the confusion matrix for the test data:

```python
import wandb
wandb.plot.confusion_matrix(preds=y_pred, y_true=y_test, class_names=class_names)

```

Training metrics such as `train_accuracy`, `train_loss`, `val_accuracy`, and `val_loss` are tracked and logged directly to Weights and Biases during the training loop.

## Usage

### Running the Training Script

Execute the training script with default parameters using the command line interface:

```bash
python train.py

```

### Command-line Arguments

You can customize the training process using the following arguments:

```bash
python train.py --dataset fashion_mnist --epochs 20 --batch_size 32 --optimizer rmsprop --learning_rate 0.001 --activation relu

```

### Argument Details

* 
`--dataset`: Choose between `mnist` and `fashion_mnist`.


* 
`--epochs`: Number of training epochs.


* 
`--batch_size`: Mini-batch size for training.


* 
`--optimizer`: Select an optimizer from `sgd`, `momentum`, `nag`, `rmsprop`.


* 
`--learning_rate` or `-lr`: Initial learning rate.


* 
`--activation` or `-a`: Choose activation function from `sigmoid`, `tanh`, `relu`.


* 
`--loss`: Choose loss function from `mean_squared_error` or `cross_entropy`.


* 
`--num_layers` or `-nhl`: Number of hidden layers.


* 
`--hidden_size` or `-sz`: Number of neurons per hidden layer.


* 
`--weight_init` or `-wi`: Weight initialization method choice of `random` or `xavier`.


* 
`--weight_decay` or `-wd`: Weight decay for L2 regularization.


* 
`--wandb_project` or `-w_p`: Weights and Biases Project ID.



## Output

* Training and validation accuracy/loss logged for each epoch.
* Confusion matrix plotted for the test set evaluations. - The best model weights saved as `best_model.npy` based on the test F-1 score.


* The optimal configuration saved as `best_config.json`.



## Example

To train a model with ReLU activation and RMSprop optimizer:

```bash
python train.py --dataset mnist --epochs 15 --batch_size 32 --optimizer rmsprop --activation relu

```

## Results Tracking with WandB

Ensure you are logged into WandB before running the script to track your hyperparameter sweeps and metrics:

```bash
wandb login

```