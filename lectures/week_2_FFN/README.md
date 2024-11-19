# Week 2: Feed-Forward Networks (FFN)

## Overview
This week covers the fundamentals of Feed-Forward Neural Networks, implementing them in multiple frameworks, and transitioning from toy examples to real applications. We'll explore the architecture, mathematics, and practical implementation aspects of FFNs.

## Learning Objectives
By the end of this week, students should be able to:
- Understand the basic architecture of feed-forward neural networks
- Implement FFNs using NumPy, PyTorch, and TensorFlow
- Compare and contrast different implementation approaches
- Debug common issues in neural network training
- Transition from simple to complex examples

## Topics Covered
1. **Neural Network Basics**
   - ## Neurons and Activation Functions

In the context of artificial neural networks, neurons serve as the fundamental building blocks that process and transmit information. A neuron receives input signals, applies a set of weights to these inputs, and then passes the weighted sum through an activation function to produce an output signal. The choice of activation function plays a crucial role in determining the behavior and capabilities of the neural network.

Mathematically, the output of a neuron can be expressed as:

$$y = f\left(\sum_{i=1}^{n} w_i x_i + b\right)$$

where $y$ is the output of the neuron, $f$ is the activation function, $w_i$ are the weights associated with each input $x_i$, $n$ is the number of inputs, and $b$ is the bias term. The bias term allows the neuron to shift the activation function, providing additional flexibility in learning complex patterns.

Activation functions introduce non-linearity into the neural network, enabling it to learn and represent complex relationships between inputs and outputs. Without non-linear activation functions, neural networks would be limited to learning only linear transformations, severely restricting their ability to model intricate patterns and solve complex problems. Some commonly used activation functions include the sigmoid function ($f(x) = \frac{1}{1 + e^{-x}}$), the hyperbolic tangent function ($f(x) = \tanh(x)$), and the rectified linear unit (ReLU) function ($f(x) = \max(0, x)$).

The choice of activation function depends on the specific requirements of the problem at hand. For example, the sigmoid function is often used in the output layer of binary classification problems, as it maps the output to a probability between 0 and 1. The ReLU function has become popular in recent years due to its computational efficiency and ability to alleviate the vanishing gradient problem, which can occur when using sigmoid or hyperbolic tangent functions in deep neural networks. Other activation functions, such as the leaky ReLU and the exponential linear unit (ELU), have been proposed to address some of the limitations of the standard ReLU function.

The combination of neurons and activation functions forms the basis for constructing deep neural networks. By stacking multiple layers of neurons with appropriate activation functions, neural networks can learn hierarchical representations of the input data, enabling them to capture increasingly complex patterns and relationships. The weights of the neurons are learned through a process called backpropagation, which involves computing the gradients of the loss function with respect to the weights and iteratively updating them to minimize the loss. The choice of activation functions, along with other hyperparameters such as the number of layers, the number of neurons per layer, and the learning rate, plays a significant role in determining the performance and generalization ability of the neural network.
   - ## Forward Propagation in Neural Networks

Forward propagation is a fundamental process in neural networks that involves computing the output of the network given an input. It is the first step in the training process, where the input data is passed through the network, and the output is calculated based on the current weights and biases of the neurons. The goal of forward propagation is to compute the predicted output of the network, which is then compared to the actual output to calculate the error or loss.

In a feedforward neural network, the input data is passed through the network layer by layer. Each neuron in a layer receives the weighted sum of the outputs from the previous layer, applies an activation function, and passes the result to the next layer. The activation function introduces non-linearity into the network, allowing it to learn complex patterns and relationships in the data. Common activation functions include the sigmoid function, hyperbolic tangent (tanh), and rectified linear unit (ReLU). The output of a neuron in layer $l$ can be expressed as:

$$a_j^{(l)} = \sigma\left(\sum_{i=1}^{n_{l-1}} w_{ji}^{(l)} a_i^{(l-1)} + b_j^{(l)}\right)$$

where $a_j^{(l)}$ is the activation of neuron $j$ in layer $l$, $\sigma$ is the activation function, $w_{ji}^{(l)}$ is the weight connecting neuron $i$ in layer $l-1$ to neuron $j$ in layer $l$, $a_i^{(l-1)}$ is the activation of neuron $i$ in layer $l-1$, $b_j^{(l)}$ is the bias term for neuron $j$ in layer $l$, and $n_{l-1}$ is the number of neurons in layer $l-1$.

The forward propagation process continues through the network until the output layer is reached. The output layer typically uses a different activation function depending on the task at hand. For example, in a binary classification problem, the output layer may use the sigmoid function to produce a probability between 0 and 1. In a multi-class classification problem, the softmax function is commonly used to produce a probability distribution over the classes. The softmax function is defined as:

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

where $z_i$ is the input to the softmax function for class $i$, and $K$ is the total number of classes.

Once the output of the network is computed, it is compared to the actual output using a loss function. The choice of loss function depends on the task and the type of output. For example, in a regression problem, the mean squared error (MSE) loss function is commonly used:

$$\text{MSE} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

where $N$ is the number of samples, $y_i$ is the actual output for sample $i$, and $\hat{y}_i$ is the predicted output for sample $i$.

The computed loss is then used in the backpropagation algorithm to update the weights and biases of the network, with the goal of minimizing the loss and improving the network's performance. Forward propagation is an essential step in this process, as it provides the predicted output necessary for computing the loss and updating the network's parameters.
   - ## Backpropagation

Backpropagation is a fundamental algorithm in machine learning, particularly in the training of artificial neural networks. It is a method for efficiently computing the gradients of the loss function with respect to the weights and biases of the network. The algorithm leverages the chain rule of calculus to propagate the error signal from the output layer back through the hidden layers, allowing for the adjustment of the network's parameters in order to minimize the loss function.

The process of backpropagation can be broken down into several key steps. First, the input data is fed forward through the network, and the output is computed. The loss function, denoted as $L$, is then calculated by comparing the predicted output with the true labels. The objective is to minimize this loss function by adjusting the weights and biases of the network. The gradient of the loss function with respect to the output of the final layer is computed as:

$$\frac{\partial L}{\partial \hat{y}} = \frac{\partial L}{\partial z^{(L)}} \odot \sigma'(z^{(L)})$$

where $\hat{y}$ is the predicted output, $z^{(L)}$ is the pre-activation output of the final layer, $\sigma$ is the activation function, and $\odot$ denotes element-wise multiplication.

Next, the gradients of the loss function with respect to the weights and biases of each layer are computed using the chain rule. For a given layer $l$, the gradient of the loss function with respect to the pre-activation output $z^{(l)}$ is:

$$\frac{\partial L}{\partial z^{(l)}} = \left(\frac{\partial L}{\partial z^{(l+1)}} \cdot W^{(l+1)}\right) \odot \sigma'(z^{(l)})$$

where $W^{(l+1)}$ represents the weights of the next layer. The gradients with respect to the weights and biases of layer $l$ are then calculated as:

$$\frac{\partial L}{\partial W^{(l)}} = \frac{\partial L}{\partial z^{(l)}} \cdot a^{(l-1)T}$$
$$\frac{\partial L}{\partial b^{(l)}} = \frac{\partial L}{\partial z^{(l)}}$$

where $a^{(l-1)}$ is the activation output of the previous layer.

The process of computing gradients is repeated for each layer, propagating the error signal backwards through the network. Once the gradients have been computed, the weights and biases are updated using an optimization algorithm, such as gradient descent, which adjusts the parameters in the direction of steepest descent of the loss function:

$$W^{(l)} := W^{(l)} - \alpha \frac{\partial L}{\partial W^{(l)}}$$
$$b^{(l)} := b^{(l)} - \alpha \frac{\partial L}{\partial b^{(l)}}$$

where $\alpha$ is the learning rate, a hyperparameter that controls the step size of the parameter updates.

The backpropagation algorithm has proven to be highly effective in training deep neural networks and has been a key enabler of the success of deep learning in various domains, such as computer vision, natural language processing, and speech recognition. However, the algorithm is not without its challenges, such as the vanishing and exploding gradient problems, which can hinder the training of very deep networks. Techniques like weight initialization, gradient clipping, and the use of alternative activation functions have been developed to mitigate these issues and improve the stability and convergence of the training process.
   - ## Loss Functions and Optimization

In the context of machine learning, loss functions play a crucial role in quantifying the discrepancy between the predicted outputs of a model and the actual target values. The choice of an appropriate loss function is dependent on the nature of the problem at hand, such as regression, classification, or ranking tasks. For instance, in a regression problem, where the objective is to predict continuous values, common loss functions include Mean Squared Error (MSE) and Mean Absolute Error (MAE). MSE is defined as:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

where $n$ is the number of samples, $y_i$ is the actual target value, and $\hat{y}_i$ is the predicted value. MAE, on the other hand, is less sensitive to outliers and is defined as:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

In classification tasks, where the goal is to assign input samples to discrete categories, loss functions such as Cross-Entropy Loss and Hinge Loss are commonly employed. Cross-Entropy Loss, also known as Log Loss, measures the dissimilarity between the predicted probability distribution and the true probability distribution. For binary classification, the Cross-Entropy Loss is given by:

$$\text{CE Loss} = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$$

where $y_i \in \{0, 1\}$ represents the true binary label and $\hat{y}_i \in [0, 1]$ is the predicted probability of the positive class.

Optimization algorithms are employed to minimize the chosen loss function and update the model's parameters accordingly. Gradient Descent is a fundamental optimization algorithm that iteratively adjusts the parameters in the direction of the negative gradient of the loss function. The update rule for Gradient Descent is given by:

$$\theta_{t+1} = \theta_t - \alpha \nabla_{\theta} J(\theta_t)$$

where $\theta_t$ represents the parameter values at iteration $t$, $\alpha$ is the learning rate, and $\nabla_{\theta} J(\theta_t)$ denotes the gradient of the loss function $J$ with respect to the parameters $\theta$ at iteration $t$. Variants of Gradient Descent, such as Stochastic Gradient Descent (SGD) and Mini-Batch Gradient Descent, are commonly used in practice to improve computational efficiency and convergence properties.

Advanced optimization algorithms, such as Adaptive Gradient (AdaGrad), Root Mean Square Propagation (RMSProp), and Adaptive Moment Estimation (Adam), have been developed to address the limitations of vanilla Gradient Descent. These algorithms adapt the learning rates for each parameter based on historical gradients, allowing for faster convergence and better handling of sparse gradients. For example, the update rule for Adam is given by:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_{\theta} J(\theta_t)$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_{\theta} J(\theta_t))^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$$
$$\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

where $m_t$ and $v_t$ are the first and second moment estimates, respectively, $\beta_1$ and $\beta_2$ are hyperparameters controlling the decay rates, and $\epsilon$ is a small constant for numerical stability.

2. **Implementation in Different Frameworks**
## Building an FFN from Scratch using NumPy

```python
import numpy as np

class FeedForwardNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        self.z1 = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.weights_hidden_output) + self.bias_output
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m

    def backward(self, X, y_true, y_pred, learning_rate):
        m = X.shape[0]
        d_loss_a2 = y_pred - y_true
        d_loss_z2 = d_loss_a2 * self.sigmoid_derivative(y_pred)
        d_loss_weights_hidden_output = np.dot(self.a1.T, d_loss_z2) / m
        d_loss_bias_output = np.sum(d_loss_z2, axis=0, keepdims=True) / m

        d_loss_a1 = np.dot(d_loss_z2, self.weights_hidden_output.T)
        d_loss_z1 = d_loss_a1 * self.sigmoid_derivative(self.a1)
        d_loss_weights_input_hidden = np.dot(X.T, d_loss_z1) / m
        d_loss_bias_hidden = np.sum(d_loss_z1, axis=0, keepdims=True) / m

        self.weights_input_hidden -= learning_rate * d_loss_weights_input_hidden
        self.bias_hidden -= learning_rate * d_loss_bias_hidden
        self.weights_hidden_output -= learning_rate * d_loss_weights_hidden_output
        self.bias_output -= learning_rate * d_loss_bias_output

    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.compute_loss(y, y_pred)
            self.backward(X, y, y_pred, learning_rate)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Example usage:
# X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y = np.array([[0], [1], [1], [0]])
# nn = FeedForwardNeuralNetwork(input_size=2, hidden_size=2, output_size=1)
# nn.train(X, y, epochs=1000, learning_rate=0.1)
```

An FFN (Feed-Forward Neural Network) is a fundamental architecture in deep learning, consisting of an input layer, one or more hidden layers, and an output layer. Each layer is composed of neurons that receive weighted inputs from the previous layer, apply an activation function, and pass the output to the next layer. The weights of the connections between neurons are learned during the training process using optimization algorithms such as gradient descent. In this section, we will explore the process of building an FFN from scratch using NumPy, a powerful numerical computing library in Python.

To begin constructing an FFN, we first need to define the architecture of the network. This involves specifying the number of layers, the number of neurons in each layer, and the activation functions to be used. Let's consider a simple example of an FFN with one hidden layer. The input layer has $n$ neurons, the hidden layer has $m$ neurons, and the output layer has $k$ neurons. We can represent the weights of the connections between the input and hidden layers as a matrix $W_1$ of shape $(n, m)$, and the weights between the hidden and output layers as a matrix $W_2$ of shape $(m, k)$. Additionally, we need to define bias vectors $b_1$ and $b_2$ for the hidden and output layers, respectively.

The forward propagation process in an FFN involves computing the activations of each layer given an input vector $x$. First, we calculate the weighted sum of the inputs for the hidden layer: $z_1 = W_1^T x + b_1$. We then apply an activation function $\sigma$ element-wise to $z_1$ to obtain the activations of the hidden layer: $a_1 = \sigma(z_1)$. Common activation functions include the sigmoid function $\sigma(z) = \frac{1}{1 + e^{-z}}$, the hyperbolic tangent function $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, and the rectified linear unit (ReLU) function $\text{ReLU}(z) = \max(0, z)$. The choice of activation function depends on the specific problem and the desired properties of the network. We repeat the process for the output layer, computing $z_2 = W_2^T a_1 + b_2$ and applying an appropriate activation function to obtain the final output $\hat{y} = \sigma(z_2)$.

During the training phase, we aim to minimize a loss function that measures the discrepancy between the predicted outputs $\hat{y}$ and the true labels $y$. A commonly used loss function for regression problems is the mean squared error (MSE): $\mathcal{L}(\hat{y}, y) = \frac{1}{2N} \sum_{i=1}^N (\hat{y}_i - y_i)^2$, where $N$ is the number of training examples. For classification tasks, the cross-entropy loss is often employed: $\mathcal{L}(\hat{y}, y) = -\frac{1}{N} \sum_{i=1}^N \sum_{j=1}^k y_{ij} \log(\hat{y}_{ij})$, where $y_{ij}$ is the true probability of example $i$ belonging to class $j$, and $\hat{y}_{ij}$ is the predicted probability. To minimize the loss function, we use gradient descent, which iteratively updates the weights and biases in the direction of the negative gradient of the loss with respect to these parameters: $W \leftarrow W - \alpha \frac{\partial \mathcal{L}}{\partial W}$ and $b \leftarrow b - \alpha \frac{\partial \mathcal{L}}{\partial b}$, where $\alpha$ is the learning rate.

To implement the backpropagation algorithm, which efficiently computes the gradients of the loss with respect to the weights and biases, we need to apply the chain rule of differentiation. Starting from the output layer, we calculate the gradient of the loss with respect to the activations: $\delta_2 = \frac{\partial \mathcal{L}}{\partial \hat{y}} \odot \sigma'(z_2)$, where $\odot$ denotes element-wise multiplication and $\sigma'$ is the derivative of the activation function. We then compute the gradients with respect to the weights and biases of the output layer: $\frac{\partial \mathcal{L}}{\partial W_2} = a_1 \delta_2^T$ and $\frac{\partial \mathcal{L}}{\partial b_2} = \delta_2$. Propagating the gradients back to the hidden layer, we calculate $\delta_1 = (W_2 \delta_2) \odot \sigma'(z_1)$ and obtain the gradients for the hidden layer's weights and biases: $\frac{\partial \mathcal{L}}{\partial W_1} = x \delta_1^T$ and $\frac{\partial \mathcal{L}}{\partial b_1} = \delta_1$. This process can be extended to FFNs with multiple hidden layers by recursively applying the chain rule.
   - ## PyTorch Implementation
