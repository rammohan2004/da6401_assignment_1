"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import numpy as np
import wandb 
from utils.data_loader import load_and_preprocess_data 
from ann.neural_network import NeuralNetwork
import os

def parse_arguments():
    """
    Parse command-line arguments.
    
    TODO: Implement argparse with the following arguments:
    - dataset: 'mnist' or 'fashion_mnist'
    - epochs: Number of training epochs
    - batch_size: Mini-batch size
    - learning_rate: Learning rate for optimizer
    - optimizer: 'sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'
    - hidden_layers: List of hidden layer sizes
    - num_neurons: Number of neurons in hidden layers
    - activation: Activation function ('relu', 'sigmoid', 'tanh')
    - loss: Loss function ('cross_entropy', 'mse')
    - weight_init: Weight initialization method
    - wandb_project: W&B project name
    - model_save_path: Path to save trained model (do not give absolute path, rather provide relative path)
    """

    print("Train : parse_argument")
    
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('-d', '--dataset', type=str, default='mnist', choices=['mnist', 'fashion_mnist'])
    parser.add_argument('-e', '--epochs', type=int, default=10) # Use your best epoch count
    parser.add_argument('-b', '--batch_size', type=int, default=128) # Use your best batch size
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy', choices=['mse', 'cross_entropy'])
    parser.add_argument('-o', '--optimizer', type=str, default='rmsprop', choices=['sgd', 'momentum', 'nag', 'rmsprop', 'adam', 'nadam'])
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0)
    parser.add_argument('-nhl', '--num_layers', type=int, default=3)
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128, 64, 32])
    parser.add_argument('-a', '--activation', type=str, default='relu', choices=['sigmoid', 'tanh', 'relu'])
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier', choices=['random', 'xavier'])
    parser.add_argument('-w_p','--wandb_project', type=str, default=None)
    parser.add_argument('--model_save_path', type=str, default='best_model.npy')
    
    args = parser.parse_args()
    
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) != args.num_layers:
        raise ValueError(f"--hidden_size must have exactly {args.num_layers} values when specified as a list.")
    
    print("Train parse arguments : ")
    print(args)
    
    return args


def main():
    """
    Main training function.
    """
    print('Train : arg parse')
    args = parse_arguments()
    
    # Initializing wandb
    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, config=vars(args))
    
    #Loading data 
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(args.dataset)
    print(f"Data loaded: train {X_train.shape}, test {X_test.shape}")
    
    #Initializing the Model
    print("Initializing model...")
    model = NeuralNetwork(args) 
    #Training model
    print("Starting training...")
    history = model.train(X_train, y_train, args.epochs, args.batch_size)

    

    # Safely extract the save path, defaulting to 'src/'
    '''  save_path = getattr(args, 'model_save_path', 'src/')
    os.makedirs(save_path, exist_ok=True)
    
    # Saving the configurations as json
    config_path = os.path.join(save_path, 'best_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")
    '''
    print("Train main : model saving : best weights")
    
    # Saving model weights
    best_weights = model.best_weights
    
    #model_path = os.path.join(save_path, 'best_model.npy')
    np.save('best_model.npy', best_weights)
    print(f"Model weights saved")

    '''
    #Saving the model
    os.makedirs(args.model_save_path, exist_ok=True)
    
    #Saving the configurations as json
    config_path = os.path.join(args.model_save_path, 'best_config.json')
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    print(f"Configuration saved to {config_path}")
    

    #Saving model weights and biases as numpy .npy file

    best_weights = model.best_weights
    model_path = os.path.join(args.model_save_path, 'best_model.npy')
    np.save(model_path, best_weights)
    print(f"Model weights saved to {model_path}") '''
    
    #Finishing wandb
    if args.wandb_project is not None:
        wandb.finish()
    
    print("Training complete!") 

    ''' weights_dict = {}
    for i, layer in enumerate(model.layers):
        weights_dict[f'W_{i}'] = layer.W
        weights_dict[f'b_{i}'] = layer.b
    model_path = os.path.join(args.model_save_path, 'model.npy')
    np.save(model_path, weights_dict)
    print(f"Model weights saved to {model_path}")'''


if __name__ == '__main__':
    main()