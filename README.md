# Multi-label Classification with Dense Embeddings

This repository contains a multi-label classification project focused on classifying questions from StackExchange into various tech domains using dense embeddings within a neural network. The task involves building a custom neural network architecture and using various PyTorch functionalities to achieve meaningful results. The project also handles class imbalances using appropriate metrics like Hamming Loss.

## Project Overview

### Dataset
This dataset is a Kaggle competition dataset, where the goal is to classify questions from StackExchange across multiple tech domains. Each post can mention more than one tech domain, making it a multi-label classification problem.

### Key Features
- **Text Processing**: Tokenizes text and builds a vocabulary using the training data.
- **Custom Dataset**: Implements a custom dataset class for handling input data and labels.
- **Model**: A neural network using dense embeddings via PyTorch’s `EmbeddingBag` layer.
- **Class Imbalance Handling**: Uses weighted loss functions to account for class imbalance.
- **Evaluation**: Evaluates the model using Hamming Loss, Precision, Recall, and F1-Score.
- **Early Stopping**: Incorporates early stopping to avoid overfitting.

## Model Architecture

The architecture of the model consists of the following layers:

1. **Embedding Bag Layer** - Captures dense embeddings for text sequences.
2. **Fully Connected Layer 1** - First hidden layer with ReLU activation, BatchNorm, and Dropout.
3. **Fully Connected Layer 2** - Second hidden layer with ReLU activation, BatchNorm, and Dropout.
4. **Output Layer** - Produces outputs for each class (multi-label classification).

### Hyperparameters
- Embedding Dimension: 300
- Hidden Layer 1 Neurons: 200
- Hidden Layer 2 Neurons: 100
- Dropout Probability: 0.5
- Learning Rate: 0.001
- Batch Size: 128
- Epochs: 5

## Key Steps

### 1. Preprocessing
- Convert text into indices using the vocabulary.
- Compute offsets for batch processing.

### 2. Training
- The model is trained using `BCEWithLogitsLoss` with custom class weights to balance the dataset.
- Gradient clipping is applied to prevent exploding gradients.

### 3. Evaluation
- Evaluated using **Hamming Loss** for multi-label classification.
- **Precision**, **Recall**, and **F1-Score** are calculated for each class.

## Model Results
_Model results and relevant plots go here._

## Inference Pipeline
The inference pipeline follows these steps:
1. Preprocess the text data, converting them to indices using the vocab.
2. Pass the indices through the model to get the predictions.
3. Post-process the logits to obtain class labels and map them back to their class names.

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
