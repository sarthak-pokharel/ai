# Boltzmann Machine Implementation

This repository contains a basic implementation of a Boltzmann Machine in Python, along with an example usage for handwritten digit recognition.

## Overview

A Boltzmann Machine is a type of stochastic neural network that can learn to represent and generate data by modeling the joint probability distribution of its inputs and outputs. It consists of visible units (inputs) and hidden units, with weighted connections between them. The training process involves adjusting the weights and biases to maximize the likelihood of the training data.

This implementation uses the contrastive divergence algorithm for training, which approximates the gradient of the log-likelihood by taking a small number of Gibbs sampling steps.

## Usage

The main script `boltzmann_machine.py` contains the `BoltzmannMachine` class and an example usage for handwritten digit recognition using the digits dataset from scikit-learn.

To run the example:

1. Install the required dependencies:
2. Run the script:

This will train a Boltzmann Machine on the digits dataset and evaluate its accuracy for classification.

## Code Structure

- `BoltzmannMachine` class:
  - `__init__`: Initialize the Boltzmann Machine with the specified number of visible and hidden units, learning rate, and batch size.
  - `sigmoid`: Helper function to compute the sigmoid activation.
  - `sample_visible`: Sample the visible units given the hidden unit states.
  - `sample_hidden`: Sample the hidden units given the visible unit states.
  - `contrastive_divergence`: Perform one step of contrastive divergence training.
  - `train`: Train the Boltzmann Machine on the provided data for the specified number of epochs.

- `main` function:
  - Load the digits dataset from scikit-learn.
  - Split the data into train and test sets.
  - Create and train a Boltzmann Machine instance on the training data.
  - Use the trained Boltzmann Machine for classification on the test data.
  - Evaluate and print the classification accuracy.

