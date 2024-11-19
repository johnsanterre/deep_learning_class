# Week 3: Hands-on Feed Forward Networks

## Overview
This week provides hands-on experience with Feed Forward Networks (FFNs), focusing on practical implementation and understanding through interactive exercises.

## Learning Objectives
By the end of this week, students should be able to:
- Implement basic FFNs from scratch
- Understand forward and backward propagation through hands-on coding
- Debug common issues in FFN implementation
- Visualize network behavior and learning process
- Apply FFNs to simple real-world problems

## Topics Covered
1. **Hands-on FFN Implementation**
   - Building neurons and layers
   - Implementing forward propagation
   - Coding backward propagation
   - Creating the training loop

2. **Interactive Learning**
   - Step-by-step network construction
   - Visualization of learning process
   - Real-time parameter updates
   - Error analysis and debugging

3. **Practical Exercises**
   - Simple classification tasks
   - Basic regression problems
   - Network behavior analysis
   - Performance optimization 


PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab. It provides a flexible and dynamic approach to building and training neural networks, making it a popular choice among researchers and practitioners in the field of data science. PyTorch's core features include automatic differentiation, GPU acceleration, and a rich set of libraries for various deep learning tasks.

The fundamental data structure in PyTorch is the tensor, which is similar to NumPy's ndarray but can be efficiently computed on GPUs. Tensors are multi-dimensional arrays that can hold data of various types, such as integers, floats, and booleans. PyTorch provides a wide range of tensor operations, including element-wise arithmetic, matrix multiplication, and broadcasting. These operations can be performed on both CPUs and GPUs, enabling efficient computation of large-scale models.

One of the key advantages of PyTorch is its dynamic computational graph, which allows for more flexible and intuitive model development. In contrast to static graph frameworks like TensorFlow, PyTorch's computational graph is defined on-the-fly during the forward pass of the model. This enables easier debugging, as well as the ability to modify the graph structure during runtime. The dynamic graph also facilitates the implementation of complex architectures, such as recursive neural networks and attention mechanisms.

PyTorch's autograd module is responsible for automatic differentiation, which is crucial for training deep learning models. Autograd keeps track of the operations performed on tensors and constructs a computational graph in the background. During the backward pass, it efficiently computes the gradients of the loss function with respect to the model parameters using the chain rule of differentiation. This allows for seamless optimization of the model using gradient-based methods, such as stochastic gradient descent (SGD) or its variants, like Adam and RMSprop. The optimization process is defined as:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta J(\theta_t)$$

where $\theta$ represents the model parameters, $\eta$ is the learning rate, and $\nabla_\theta J(\theta_t)$ denotes the gradient of the loss function $J$ with respect to the parameters at iteration $t$.

PyTorch also provides a rich ecosystem of libraries and extensions that cater to various deep learning tasks. For example, torchvision offers pre-trained models and utilities for computer vision tasks, such as image classification, object detection, and semantic segmentation. Similarly, torchtext provides tools for natural language processing, including text preprocessing, vocabulary management, and dataset handling. These libraries, along with the extensive community support and a wide range of third-party packages, make PyTorch a versatile and powerful framework for implementing deep learning models in data science applications.
   -## TensorFlow Implementation

TensorFlow is an open-source software library developed by Google for numerical computation and large-scale machine learning. It utilizes data flow graphs, where nodes represent mathematical operations and edges represent multidimensional data arrays (tensors) that flow between the nodes. TensorFlow provides a flexible ecosystem for implementing and deploying machine learning models across various platforms, including CPUs, GPUs, and TPUs.

The core concept in TensorFlow is the computational graph, which defines the flow of data and the operations performed on that data. To build a computational graph, you start by creating placeholders for input data using the `tf.placeholder()` function. These placeholders allow you to feed data into the graph during runtime. Next, you define the mathematical operations and transformations that need to be applied to the input data. TensorFlow provides a wide range of built-in functions and classes for common operations, such as `tf.matmul()` for matrix multiplication, `tf.nn.conv2d()` for 2D convolution, and `tf.nn.relu()` for the rectified linear unit activation function.

Once the computational graph is defined, you can create a TensorFlow session using `tf.Session()`. The session is responsible for executing the graph and allocating resources for the computations. You can feed input data into the placeholders using the `feed_dict` parameter of the `session.run()` method. This method runs the specified operations in the graph and returns the output tensors. It is important to close the session using `session.close()` to release the allocated resources when you are done with the computations.

TensorFlow also provides high-level APIs, such as Keras and Estimators, which abstract away many of the low-level details and provide a more user-friendly interface for building and training machine learning models. Keras is a high-level neural networks API that allows you to define models using a sequential or functional API. It supports various layer types, such as dense layers (`tf.keras.layers.Dense()`), convolutional layers (`tf.keras.layers.Conv2D()`), and recurrent layers (`tf.keras.layers.LSTM()`). Keras also provides utilities for model compilation, training, and evaluation. Estimators, on the other hand, are a high-level API for distributed training and deployment of machine learning models. They encapsulate the model architecture, training, and evaluation logic, making it easier to scale models to large datasets and deploy them in production environments.

TensorFlow supports automatic differentiation, which is crucial for training machine learning models. It allows you to compute gradients of scalar functions with respect to their parameters efficiently. The `tf.GradientTape()` context manager records operations performed inside its scope, enabling automatic differentiation. By calling the `tape.gradient()` method, you can compute the gradients of a target tensor with respect to the watched tensors. This feature is particularly useful for implementing optimization algorithms, such as gradient descent, which update the model parameters based on the computed gradients to minimize a loss function. TensorFlow provides various optimization classes, such as `tf.train.GradientDescentOptimizer()` and `tf.train.AdamOptimizer()`, that handle the parameter updates based on the computed gradients.
   -## Neural Network Frameworks: A Comparative Analysis

The rapid advancement of deep learning has led to the development of various neural network frameworks, each with its unique approach to building and training models. These frameworks provide high-level APIs and abstractions, enabling data scientists and machine learning practitioners to focus on model design and experimentation rather than low-level implementation details. In this section, we will compare and contrast the approaches taken by popular neural network frameworks, highlighting their strengths and differences.

TensorFlow, developed by Google, is one of the most widely adopted neural network frameworks. It follows a graph-based approach, where computations are represented as a directed acyclic graph (DAG). In TensorFlow, the graph is defined using a static computation graph, which means that the entire graph is constructed before execution. This allows for efficient optimization and parallelization of computations. TensorFlow provides a comprehensive ecosystem with extensive documentation, pre-trained models, and a large community support. It supports both eager execution and graph-based execution, catering to different development preferences. The mathematical foundation of TensorFlow lies in tensor algebra, where computations are performed on multidimensional arrays called tensors. For example, a simple matrix multiplication in TensorFlow can be expressed as:

$$
\mathbf{C} = \mathbf{A} \cdot \mathbf{B}
$$

where $\mathbf{A}$ and $\mathbf{B}$ are input tensors, and $\mathbf{C}$ is the resulting output tensor.

PyTorch, developed by Facebook, takes a different approach compared to TensorFlow. It emphasizes dynamic computation graphs and eager execution, allowing for more flexibility and easier debugging. In PyTorch, the graph is defined dynamically during runtime, enabling immediate evaluation and modification of computations. This dynamic nature makes it more intuitive for researchers and developers who prefer a more imperative programming style. PyTorch's autograd module automatically computes gradients for backpropagation, simplifying the process of training neural networks. It also provides a rich set of optimized tensor operations, making it efficient for numerical computations. PyTorch's mathematical operations are similar to NumPy, a popular scientific computing library in Python. For instance, element-wise addition of two tensors in PyTorch can be performed as:

$$
\mathbf{C} = \mathbf{A} + \mathbf{B}
$$

where $\mathbf{A}$ and $\mathbf{B}$ are input tensors, and $\mathbf{C}$ is the resulting output tensor.

Keras, a high-level neural network library, prioritizes simplicity and ease of use. It provides a user-friendly API for building and training neural networks, making it accessible to beginners and rapid prototyping. Keras follows a declarative approach, where the model architecture is defined using a sequence of layers. It abstracts away many low-level details, allowing users to focus on the overall structure of the model. Keras supports multiple backend engines, including TensorFlow and Theano, enabling seamless integration with different frameworks. It offers a wide range of built-in layers, activation functions, and optimization algorithms, facilitating the creation of complex neural network architectures with minimal code. Keras also provides utilities for data preprocessing, model evaluation, and visualization, enhancing the end-to-end workflow of building and training models.

3. **Code Examples**


**Classification**
```python
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions
y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")
```
**Regression**

```python
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(1000, 3)
y = 3 * X[:, 0] + 2 * X[:, 1] + X[:, 2] + np.random.randn(1000) * 0.1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate the model
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)

# Predict
predictions = model.predict(X_test_scaled)
```
   - Scaling to larger datasets
   - Best practices for model development

## Required Reading
- Deep Learning Book (Goodfellow et al.) - Chapter 6: Deep Feedforward Networks
- PyTorch Documentation - Neural Networks Tutorial
- TensorFlow Documentation - Keras Sequential API Guide

## Additional Resources
- Interactive visualizations of neural networks
- Code examples in each framework
- Debugging guides for common issues
