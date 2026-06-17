# Deep Learning from Scratch: Dynamic Multilayer Perceptron (MLP) Implementation via NumPy

This repository contains a production-grade, object-oriented implementation of a dynamic Multilayer Perceptron (MLP) built entirely from scratch using **NumPy**. It bypasses high-level deep learning frameworks (such as PyTorch or TensorFlow) to expose the underlying matrix calculus, vectorization paradigms, and optimization mechanics involved in training deep neural networks.

---

## Technical Highlights

* **Dynamic Network Topology:** The architecture accepts arbitrary configurations for hidden layers and automatically computes, allocates, and aligns the dimensions of all weight matrices and bias vectors.
* **He (Kaiming) Initialization:** Implements the He initialization strategy to stabilize the variance of activation gradients across deep architectures, systematically mitigating the issues of vanishing and exploding gradients.
* **Vectorized Forward Pass with Linear Output:** Computes linear transformations and applies the ReLU activation function across vectorized mini-batches. The output layer retains raw linear outputs (Logits) to support both unconstrained regression tasks and flexible loss configurations.
* **Strict Dimensional Alignment in Backpropagation:** Backpropagation is computed via the vectorized chain rule. Mini-batch dimensional conflicts for bias gradients are resolved using aggregate row-reduction (`axis=0`), ensuring algebraic synchronization.
* **Boundary Layer Indexing Protection:** Implements localized memory caching (`self.input_x`) to protect the input layer boundary condition ($l=0$) from Pythonic negative-indexing errors, guaranteeing clean gradient mapping back to raw features.
* **Mini-batch Gradient Descent Orchestration:** Includes an integrated training pipeline featuring stochastic data shuffling per epoch, flexible mini-batch slicing, and automated tracking of cross-validation evaluation metrics.

---

## Mathematical Architecture & Vectorization

For a given layer $l$, where $A^{[0]} = X$ (the input matrix), the vectorized computations are governed by the following mathematical principles:

### 1. Forward Pass
The linear combination and activation are evaluated sequentially per layer:
$$Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$$
$$A^{[l]} = \text{ReLU}(Z^{[l]}) = \max(0, Z^{[l]})$$
*Note: For the terminal layer $L$, $A^{[L]} = Z^{[L]}$ (Linear Activation).*

### 2. Backward Pass (Gradient Calculus)
The error signal (Delta) is backpropagated through the network layers in reverse order:
$$dZ^{[l]} = \frac{\partial \text{Loss}}{\partial Z^{[l]}} = \frac{\partial \text{Loss}}{\partial A^{[l]}} \odot \sigma'(Z^{[l]})$$

Where $\sigma'$ is the element-wise derivative of the ReLU function. The parameter gradients are accumulated across the batch dimensions:
$$\frac{\partial \text{Loss}}{\partial W^{[l]}} = (A^{[l-1]})^T \cdot dZ^{[l]}$$
$$\frac{\partial \text{Loss}}{\partial b^{[l]}} = \sum_{\text{rows}} dZ^{[l]}$$

The upstream loss gradient is then passed to the preceding layer:
$$\frac{\partial \text{Loss}}{\partial A^{[l-1]}} = dZ^{[l]} \cdot (W^{[l]})^T$$

---

## Comprehensive Source Code

```python
import numpy as np

# Mock utility functions for standalone integrity
def relu(z):
    return np.maximum(0, z)

def he_initialize(d_in, d_out):
    return np.random.randn(d_in, d_out) * np.sqrt(2.0 / d_in)

class NeuralNetwork:
    """
    A fully dynamic, vectorized Multilayer Perceptron (MLP) implementation 
    utilizing raw matrix calculus via NumPy.
    """
    def __init__(self, input_dim, output_dim, hidden_layers, training_data, testing_data):
        self.weight_list = []
        self.bias_list = []
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.training_data = training_data
        self.testing_data = testing_data
        self.initialize_weights(he_initializer=True)

    def initialize_weights(self, he_initializer=True):
        """
        Dynamically allocates memory matrices for weights and biases 
        based on the defined network topology.
        """
        if he_initializer:
            self.weight_list.append(he_initialize(self.input_dim, self.hidden_layers[0]))
            for index in range(len(self.hidden_layers) - 1):
                self.weight_list.append(he_initialize(self.hidden_layers[index], self.hidden_layers[index+1]))
                self.bias_list.append(np.zeros((1, self.hidden_layers[index])))
            self.bias_list.append(np.zeros((1, self.hidden_layers[-1])))
            self.weight_list.append(he_initialize(self.hidden_layers[-1], self.output_dim))
            self.bias_list.append(np.zeros((1, self.output_dim)))
        else:
            self.weight_list.append(np.random.normal(0, 1, size=(self.input_dim, self.hidden_layers[0])))
            for index in range(len(self.hidden_layers) - 1):
                self.weight_list.append(np.random.normal(0, 1, size=(self.hidden_layers[index], self.hidden_layers[index+1])))
                self.bias_list.append(np.zeros((1, self.hidden_layers[index])))
            self.bias_list.append(np.zeros((1, self.hidden_layers[-1])))
            self.weight_list.append(np.random.normal(0, 1, size=(self.hidden_layers[-1], self.output_dim)))
            self.bias_list.append(np.zeros((1, self.output_dim)))

    def forward(self, x):
        """
        Executes the vectorized forward propagation pass across the entire topology.
        Caches internal states for downstream gradient evaluation.
        """
        self.input_x = x  # Cache raw input features for boundary-layer gradient computation
        self.z_list = []
        self.a_list = []
        current_input = x

        # Forward pass through hidden layers with ReLU activation
        for index in range(len(self.weight_list) - 1):
            z = current_input @ self.weight_list[index] + self.bias_list[index]
            self.z_list.append(z)
            a = relu(self.z_list[index])
            self.a_list.append(a)
            current_input = self.a_list[index]

        # Output Layer: Terminal linear transformation (No Activation)
        last_z = current_input @ self.weight_list[-1] + self.bias_list[-1]
        self.z_list.append(last_z)
        self.a_list.append(last_z)
        return self.a_list[-1]

    def back_propagation(self, y_target):
        """
        Computes analytical gradients via reverse-mode automatic differentiation.
        Aligns mini-batch dimensions and stores output weight/bias gradients.
        """
        dLoss_dA = [0] * len(self.weight_list)
        # Derivative of MSE Loss with respect to the final activation layer
        dLoss_dA[-1] = 2 * (self.a_list[-1] - y_target)

        dLoss_dZ = [0] * len(self.weight_list)
        self.dLoss_dw = [0] * len(self.weight_list)
        self.dLoss_db = [0] * len(self.weight_list)

        # Reverse topological traversal
        for l in reversed(range(len(self.weight_list))):
            # Evaluate activation derivative boundary conditions
            if l == len(self.weight_list) - 1:
                relu_derivative = 1  # Identity derivative for linear output layer
            else:
                relu_derivative = (self.z_list[l] > 0).astype(int)

            # Compute local layer gradient (Delta)
            dLoss_dZ[l] = relu_derivative * dLoss_dA[l]
            
            # Resolve batch dimension discrepancy for the bias vector
            self.dLoss_db[l] = np.sum(dLoss_dZ[l], axis=0, keepdims=True)
            
            # Resolve boundary condition mapping to input features or internal activations
            if l == 0:
                a_prev = self.input_x
            else:
                a_prev = self.a_list[l-1]

            # Matrix multiplication for structural parameter gradients
            self.dLoss_dw[l] = a_prev.T @ dLoss_dZ[l]
            
            # Backpropagate error downstream to preceding hidden layers
            if l > 0:
                dLoss_dA[l-1] = dLoss_dZ[l] @ self.weight_list[l].T

    def update_weights(self, lr=0.001):
        """
        Performs parameter optimization via Stochastic/Mini-batch Gradient Descent.
        """
        for l in range(len(self.weight_list)):
            self.weight_list[l] -= lr * self.dLoss_dw[l]
            self.bias_list[l] -= lr * self.dLoss_db[l]

    def train(self, epochs, batch_size, lr=0.001):
        """
        Orchestrates the global training pipeline using Mini-batch Gradient Descent.
        Tracks Mean Squared Error (MSE) metrics across train and test partitions.
        """
        X_train, y_train = self.training_data
        num_samples = X_train.shape[0]

        for epoch in range(epochs):
            # Stochastic Shuffling per epoch to ensure distribution consistency
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            epoch_loss = 0
            num_batches = int(np.ceil(num_samples / batch_size))

            for b in range(num_batches):
                start_idx = b * batch_size
                end_idx = min(start_idx + batch_size, num_samples)

                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                # Execution Pipeline Loop
                out = self.forward(xb)
                batch_loss = np.mean((out - yb) ** 2)
                epoch_loss += batch_loss * (end_idx - start_idx)

                self.back_propagation(yb)
                self.update_weights(lr)

            epoch_loss /= num_samples

            # Cross-Validation Evaluation Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                X_test, y_test = self.testing_data
                test_out = self.forward(X_test)
                test_loss = np.mean((test_out - y_test) ** 2)
                print(f"Epoch {epoch+1:03d}/{epochs} | Train MSE: {epoch_loss:.6f} | Test MSE: {test_loss:.6f}")


```

## Execution & Usage Verification
The framework can be instantiated and executed using synthetic datasets to verify topological convergence:

```python
# 1. Generate synthetic regression matrices
X_train_dummy = np.random.randn(500, 10)
y_train_dummy = np.random.randn(500, 1)
X_test_dummy = np.random.randn(100, 10)
y_test_dummy = np.random.randn(100, 1)

train_set = (X_train_dummy, y_train_dummy)
test_set = (X_test_dummy, y_test_dummy)

# 2. Instantiate Network Configuration (Input: 10, Hidden Layers: [64, 32], Output: 1)
mlp = NeuralNetwork(
    input_dim=10, 
    output_dim=1, 
    hidden_layers=[64, 32], 
    training_data=train_set, 
    testing_data=test_set
)

# 3. Trigger optimization sequence
mlp.train(epochs=100, batch_size=32, lr=0.005)

```
## Requirements & Environment
Core Engine: Python 3.8+

Dependencies: numpy (No additional sub-packages required)
