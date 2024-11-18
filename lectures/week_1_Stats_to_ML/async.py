"""Asynchronous learning materials for week_1_Stats_to_ML"""

## Linear Regression "The Old Way"


import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# Data: Simple relationship between x (independent) and y (dependent) variables
x = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5])

# Model: Linear Regression
model = LinearRegression().fit(x, y)

# Predict
x_pred = np.array([6]).reshape(-1, 1)
y_pred = model.predict(x_pred)
print("Predicted value:", y_pred[0])

"""
This code block demonstrates a basic implementation of linear regression using scikit-learn.
Here's what's happening step by step:

1. We import the necessary libraries:
   - numpy for numerical operations
   - LinearRegression from scikit-learn for the regression model
   - stats from scipy for statistical calculations

2. We create a simple dataset:
   - x: Independent variable with values [1,2,3,4,5], reshaped to a column vector (-1,1)
   - y: Dependent variable with values [2,4,5,4,5]
   The data suggests a somewhat linear relationship with some noise

3. We create and train the linear regression model:
   - Initialize a LinearRegression object
   - Fit it to our data using the .fit() method
   
4. We make a prediction:
   - Create a new input value x_pred = 6
   - Use the trained model to predict the corresponding y value
   - Print the predicted value

This example serves as a foundation for understanding how machine learning models
work by showing the basic workflow: data preparation, model creation, training,
and prediction.
"""


# Get the slope (coefficient) and the intercept
slope = model.coef_[0]

# Calculate predictions and residuals
residuals = y - model.predict(x)  # Fixed extra space around minus operator

# Degrees of freedom
df = len(y) - 1 - 1  # number of examples - predictors - 1  # Fixed typo in comment

# Mean squared error
mse = np.sum(residuals**2) / df  # Sum of squared residuals / DF  # Fixed spacing around division

# Standard error of the slope coefficient
SE_slope = np.sqrt(mse / np.sum((x - np.mean(x)) ** 2))

# t-statistic for the slope
t_stat = slope / SE_slope

# Calculate the p-value
p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))

"""
This code block performs statistical analysis on the linear regression model:

1. Extracts the slope coefficient from the fitted model

2. Calculates residuals by subtracting predicted values from actual y values
   Residuals represent the differences between observed and predicted values

3. Calculates degrees of freedom (df) as:
   number of samples - number of predictors - 1
   This accounts for the constraints in the model

4. Computes Mean Squared Error (MSE):
   - Squares the residuals to make them positive
   - Sums them up and divides by degrees of freedom
   MSE measures the average squared prediction error

5. Calculates Standard Error of the slope:
   - Uses MSE and the spread of x values
   - Measures uncertainty in the slope estimate

6. Computes t-statistic for the slope:
   - Ratio of slope to its standard error
   - Measures how many standard errors the slope is from zero

7. Determines p-value using t-distribution:
   - Two-tailed test (multiply by 2)
   - Uses cumulative distribution function (cdf)
   - Measures statistical significance of the slope

This analysis helps determine if there is a significant linear relationship
between x and y variables in the regression model.
"""

#slope, intercept, t_stat, p_value
# plot graph

## Random Forest "The New Old Way"
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Step 1: Generate Synthetic Data
# 10 samples and 15 features
X = np.random.rand(1000, 15000)  # Random features
y = np.random.randint(0, 2, 1000)  # Binary target variable

# Step 2: Random Forest Model
model = RandomForestClassifier(n_estimators=10)

# Step 3: Cross-Validation
# Using 3-fold cross-validation
scores = cross_val_score(model, X, y, cv=10)

print("Cross-Validation Scores:", scores)
print("Mean Score:", np.mean(scores))

"""
This code block demonstrates a Random Forest classification approach:

1. Data Generation:
   - Creates synthetic dataset with 1000 samples and 15000 features
   - Features are random values between 0 and 1
   - Target variable is binary (0 or 1)

2. Model Setup:
   - Uses RandomForestClassifier from scikit-learn
   - Sets up ensemble with 10 decision trees (n_estimators=10)
   - Random forests combine multiple decision trees to reduce overfitting
   
3. Model Evaluation:
   - Implements 10-fold cross-validation
   - Splits data into 10 parts, trains on 9 and tests on 1
   - Repeats process 10 times with different splits
   - Cross-validation helps assess model generalization
   
4. Results Output:
   - Prints individual scores from each fold
   - Calculates and prints mean score across all folds
   - Scores represent classification accuracy

This approach represents a traditional machine learning method,
contrasting with the deep learning approach that follows.
Random forests are known for:
- Good performance on many types of data
- Robustness to overfitting
- Ability to handle high-dimensional data
- Built-in feature importance metrics
"""


## Deep Learning "The Current Fancy Way"

import torch
import numpy as np

# Generate synthetic data
X = np.random.rand(1000, 10).astype(np.float32)  # 1000 samples, 10 features
y = np.random.randint(0, 2, 1000).astype(np.float32)  # Binary target variable (0 or 1)

# Convert numpy arrays to torch tensors
X_torch = torch.from_numpy(X)
y_torch = torch.from_numpy(y).view(-1, 1)  # Reshape for binary classification

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(10, 5)  # Input layer with 10 features, hidden layer with 5 neurons
        self.fc2 = torch.nn.Linear(5, 5)  # Input layer with 10 features, hidden layer with 5 neurons
        self.fc3 = torch.nn.Linear(5, 1)   # Output layer with 1 neuron
    
    def forward(self, x):  # Added blank line before method definition
        x = torch.nn.functional.relu(self.fc1(x))  # ReLU activation for hidden layer
        x = torch.nn.functional.relu(self.fc2(x))  # ReLU activation for hidden layer
        x = torch.sigmoid(self.fc3(x))  # Sigmoid activation for output layer
        return x

# Initialize the model
model = SimpleNN()

# Loss and optimizer
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(20):
    # Forward pass
    outputs = model(X_torch)
    loss = criterion(outputs, y_torch)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/200], Loss: {loss.item():.4f}')

"""
This code block implements a basic neural network using PyTorch for binary classification:

1. Data Preparation:
   - Creates synthetic dataset with 1000 samples and 10 features using NumPy
   - Features are random values between 0 and 1 (float32 type)
   - Target variable is binary (0 or 1) as float32
   - Converts NumPy arrays to PyTorch tensors for GPU compatibility
   - Reshapes target variable to match network output shape

2. Neural Network Architecture (SimpleNN class):
   - Three fully connected (Linear) layers:
     * Input layer: 10 features → 5 neurons
     * Hidden layer: 5 neurons → 5 neurons 
     * Output layer: 5 neurons → 1 neuron (binary classification)
   - ReLU activation functions between layers for non-linearity
   - Sigmoid activation at output for probability between 0-1
   - Inherits from nn.Module for PyTorch functionality

3. Model Training Setup:
   - Binary Cross Entropy Loss (BCELoss)
     * Appropriate for binary classification
     * Measures difference between predicted and actual values
   - Adam optimizer with learning rate 0.001
     * Adaptive learning rate optimization
     * Good default choice for many problems
     * Handles sparse gradients well

4. Training Loop:
   - Runs for 20 epochs (complete passes through dataset)
   - For each epoch:
     * Forward pass: Get predictions from model
     * Calculate loss between predictions and targets
     * Backward pass: Compute gradients
     * Zero gradients to prevent accumulation
     * Update weights using optimizer
   - Prints loss every 20 epochs to monitor training

This implementation demonstrates core deep learning concepts:
- Tensor operations
- Neural network layers and activations
- Backpropagation and gradient descent
- Loss functions and optimization

The architecture is intentionally simple for demonstration but could be
expanded with:
- Batch processing for larger datasets
- Dropout layers for regularization
- Additional hidden layers
- Learning rate scheduling
- Early stopping
- Validation set evaluation
"""
