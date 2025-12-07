# Neural Network From Scratch (NumPy Only) – MNIST Digit Classification

This project implements a fully-connected neural network **from scratch** using only **NumPy** (no TensorFlow, no PyTorch).  
The model is trained on the classic **MNIST handwritten digits** dataset and achieves around **97% accuracy on the test set**.

> 🎯 Goal: Understand and showcase how neural networks work internally – forward pass, backpropagation, loss, gradients, and optimization – without relying on deep learning frameworks.

---

## 1. Project Highlights

- Built a multi-layer neural network **entirely from scratch** with NumPy
- Implemented:
  - Fully-connected (Dense) layers
  - ReLU activation
  - Softmax activation
  - Categorical Cross-Entropy loss
  - Manual backpropagation
  - Mini-batch Gradient Descent
- Achieved **~97% test accuracy** on MNIST
- Clean, step-by-step notebook with explanations, plots, and evaluation

This project is designed to demonstrate **deep understanding of the mathematics and implementation details behind neural networks**, not just using high-level libraries.

---

## 2. Dataset

- **Dataset**: MNIST (handwritten digits 0–9)
- **Source**: Kaggle MNIST in CSV format (`mnist_train.csv`)
- **Samples**: 60,000 images (used)
- **Image size**: 28 × 28 pixels (flattened to 784 features)
- **Task**: Multi-class classification (10 classes: digits 0–9)

For this project:

- Data is loaded from `mnist_train.csv`
- Pixels are normalized to `[0, 1]` by dividing by 255
- Dataset is split into:
  - **80% train**
  - **10% validation**
  - **10% test**

---

## 3. Neural Network Architecture

The model is a simple feed-forward fully-connected neural network:

- **Input Layer**: 784 features (flattened 28×28 image)
- **Hidden Layer 1**: 128 neurons + ReLU
- **Hidden Layer 2**: 64 neurons + ReLU
- **Output Layer**: 10 neurons + Softmax (one per class 0–9)

### Components Implemented From Scratch

- `DenseLayer`
  - Linear transformation: `X @ W + b`
  - Stores gradients `dW`, `db`, `dinputs` for backpropagation
- `ReLU` activation
  - `f(x) = max(0, x)`
  - Backprop: passes gradient only where input > 0
- `Softmax` activation
  - Converts logits into probabilities
  - Numerically stable implementation
- `CategoricalCrossEntropy` loss
  - For multi-class classification
- `SoftmaxWithCrossEntropy` combined
  - Combines softmax + cross-entropy
  - Provides efficient gradient `dinputs` for backprop
- Manual backpropagation through all layers

---

## 4. Training Setup

- **Optimizer**: Mini-batch Gradient Descent
- **Batch size**: 128
- **Learning rate**: 0.1
- **Epochs**: 10
- **Metrics tracked**:
  - Training loss
  - Training accuracy
  - Validation loss
  - Validation accuracy

During each epoch:

1. Training data is shuffled
2. Mini-batches are passed through:
   - Dense → ReLU → Dense → ReLU → Dense → Softmax
3. Loss and accuracy are computed
4. Backpropagation computes gradients
5. Weights and biases are updated using gradient descent

---

## 5. Results

### 5.1 Training & Validation Curves

Loss and accuracy over epochs:

![Loss Curve](images/loss_curve.png)
![Accuracy Curve](images/accuracy_curve.png)

- Training and validation loss decrease smoothly
- Training and validation accuracy increase and remain close to each other
- No strong signs of overfitting on this task

### 5.2 Final Test Performance

Using the held-out test set:

- **Test Loss**: ~0.10  
- **Test Accuracy**: ~**97.0%**

### 5.3 Confusion Matrix

The confusion matrix below shows how well the model performs across individual digit classes:

![Confusion Matrix](images/confusion_matrix.png)

Most digits are classified correctly, with occasional confusion between visually similar digits (e.g., 4 vs 9, 3 vs 5), which is expected.

---

## 6. Project Structure

```text
.
├── notebook.ipynb          # Main analysis + implementation notebook
├── images/
│   ├── loss_curve.png      # Training vs validation loss
│   ├── accuracy_curve.png  # Training vs validation accuracy
│   └── confusion_matrix.png
└── README.md               # This file


src/
├── layers.py               # DenseLayer implementation
├── activations.py          # ReLU, Softmax
├── losses.py               # Cross-entropy loss
└── model.py                # High-level NeuralNetwork wrapper

7. How to Run This Project
Option 1 – Google Colab

Open the notebook (notebook.ipynb) in Google Colab.

Upload the MNIST CSV file (mnist_train.csv).

Run all cells step by step:

Data loading & preprocessing

Model definition

Training loop

Evaluation & plots

Option 2 – Local (Python environment)

Create a virtual environment and activate it

Install dependencies:

pip install numpy pandas scikit-learn matplotlib


Make sure mnist_train.csv is available in the working directory.

Run the notebook using Jupyter:

jupyter notebook notebook.ipynb

8. Skills Demonstrated

Understanding of neural network fundamentals

Forward pass and linear layers

Activation functions (ReLU, Softmax)

Loss functions for classification

Manual backpropagation and gradient descent

Working with real-world datasets (MNIST)

Data preprocessing:

Normalization

Train/validation/test split

Model evaluation:

Accuracy, loss curves

Confusion matrix

Writing clean, documented, and reproducible ML code and notebooks

9. Possible Extensions / Future Work

Add L2 regularization or dropout

Implement Adam optimizer

Try deeper networks or wider layers

Extend to:

Fashion-MNIST

Simple CNNs for image data (convolutional layers from scratch)

Refactor into a reusable NeuralNetwork class with:

fit, predict, evaluate methods

10. About Me

I am an aspiring Data Scientist / Machine Learning Engineer with strong interest in:

Deep learning fundamentals

Building models from scratch

Working with real-world datasets

ML engineering and model deployment

I’m actively looking for opportunities in Data Science / Machine Learning / AI Engineer roles.



GitHub: [https://github.com/maviyauddin]

LinkedIn: [https://www.linkedin.com/in/khaja-maviya-uddin-018995309/]

Gmail: [khajamaviyauddin@gmail.com]


























