# DA6401 Assignment 1: Feedforward Neural Network from Scratch

**Quick Links:**
* 🔗 **GitHub Repository:** https://github.com/rammohan2004/da6401_assignment_1
* 📊 **Weights & Biases Full Report:** https://wandb.ai/cs25m017-indian-institute-of-technology-madras/da6401-assignment1/reports/DA6401-Assignment-CS25M017--VmlldzoxNjEyNDA1OQ?accessToken=8n6803hd0pvpllguqe0tjlc4e7lmqe0yq0ffswigxuf4g9n0vginjocmu3tdr0n8

---

## 📌 Project Overview
This repository contains a complete, from-scratch implementation of a Multi-Layer Perceptron (MLP) using only NumPy. The project explores fundamental Deep Learning concepts, including backpropagation, hyperparameter tuning, weight initialization, and the mathematical behaviors of different loss and activation functions.

All experiments, hyperparameter sweeps, and visualizations were tracked and logged using **Weights & Biases (W&B)**.

## 🚀 Features & Experiments Completed
This project implements the following custom components and experiments:

* **Custom Neural Network API:** Fully modular `NeuralNetwork` class supporting variable hidden layers, node sizes, and custom configurations.
* **Optimizers:** Implemented standard SGD, Momentum, Nesterov Accelerated Gradient (NAG), and RMSprop.
* **Activations & Loss:** Supported ReLU, Sigmoid, Tanh, and Softmax, paired with Mean Squared Error (MSE) and Cross-Entropy loss.
* **Hyperparameter Sweep:** Conducted a 100-run W&B sweep to find the optimal architecture for the MNIST dataset.
* **Deep Learning Diagnostics:**
  * **The "Dead Neuron" Phenomenon:** Visualized ReLU collapse vs. Tanh stability at high learning rates.
  * **Loss Function Comparison:** Demonstrated why Cross-Entropy converges faster than MSE for classification.
  * **Overfitting Analysis:** Analyzed training vs. validation accuracy gaps across 100 sweep runs.
  * **Error Analysis:** Generated a test-set confusion matrix and isolated misclassified edge-case images.
  * **Symmetry Breaking:** Tracked bias gradients to prove mathematically why Xavier initialization succeeds where Zero initialization fails.
  * **Transfer Challenge:** Evaluated the performance drop of the top-3 MNIST architectures when applied to the more complex Fashion-MNIST dataset.

## 📂 Repository Structure
```text
📦 DA6401-Assignment1
 ┣ 📂 src
 ┃ ┣ 📜 neural_network.py     # Main MLP and backpropagation logic
 ┃ ┣ 📜 activations.py        # ReLU, Sigmoid, Tanh, Softmax implementations
 ┃ ┣ 📜 objective_functions.py# MSE and Cross-Entropy loss
 ┃ ┣ 📜 optimizers.py         # SGD, Momentum, NAG, RMSprop
 ┃ ┣ 📜 best_config.json      # Top hyperparameter configuration
 ┃ ┗ 📜 best_model.npy        # Saved weights of the best performing model
 ┣ 📂 notebooks
 ┃ ┗ 📜 experiments.ipynb     # Jupyter notebook containing all Q2.4 - Q2.10 experiments
 ┣ 📂 utils
 ┃ ┗ 📜 data_loader.py        # Helper to load MNIST and Fashion-MNIST
 ┗ 📜 README.md