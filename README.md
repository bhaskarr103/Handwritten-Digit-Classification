# Multi-Label Classification on MNIST Dataset

## Overview
This project implements a **multi-label classification** model using the **MNIST dataset**. It utilizes **TensorFlow** and **Keras** to train a deep learning model for handwritten digit classification.

## Features
- Loads the **MNIST dataset** from Keras.
- Builds a neural network using **Sequential API**.
- Uses **Dense** and **Flatten** layers for classification.
- Trains and evaluates the model on MNIST images.

## Requirements
To run this notebook, install the required dependencies:
```bash
pip install tensorflow numpy matplotlib
```

## Usage
1. Open the Jupyter Notebook and run all cells.
2. The dataset is loaded and preprocessed automatically.
3. The neural network model is trained and evaluated.

## Dataset
The **MNIST dataset** consists of **28x28 grayscale images** of handwritten digits (0-9). It is a benchmark dataset for classification tasks in deep learning.

## Model Architecture
The implemented model follows this architecture:
- **Flatten Layer**: Converts 28x28 images into a 1D array.
- **Dense Layer (128 neurons, ReLU activation)**
- **Dense Output Layer (10 neurons, Softmax activation)**

![Screenshot (86)](https://github.com/user-attachments/assets/13942fd7-c493-4a36-a253-f1f45c9fc3ec)<br>

![Screenshot (87)](https://github.com/user-attachments/assets/2d9375bb-cef7-4163-8f87-b55235a3b705)


## Results
After training, the model achieves a reasonable accuracy on the MNIST dataset. The exact performance metrics are displayed in the notebook output.



