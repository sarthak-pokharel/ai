# Boltzmann Machine Implementation

This repository contains a basic implementation of a Boltzmann Machine in Python, along with an example usage for handwritten digit recognition.

## Overview

A Boltzmann Machine is a type of stochastic neural network that can learn to represent and generate data by modeling the joint probability distribution of its inputs and outputs. It consists of visible units (inputs) and hidden units, with weighted connections between them. The training process involves adjusting the weights and biases to maximize the likelihood of the training data.

This implementation uses the contrastive divergence algorithm for training, which approximates the gradient of the log-likelihood by taking a small number of Gibbs sampling steps.

## Usage

The main script `boltzmann_machine.py` contains the `BoltzmannMachine` class and an example usage for handwritten digit recognition using the digits dataset from scikit-learn.

To run the example:

1. Install the required dependencies: