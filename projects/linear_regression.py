#!/usr/bin/env python3
"""
Linear Regression Implementation
- Implementation from scratch
- Using numpy
- Using scikit-learn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class LinearRegressionFromScratch:
    """
    Linear Regression implementation from scratch.
    """
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """Train the model on the given data."""
        # Initialize parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for _ in range(self.n_iterations):
            # Linear equation: y_pred = X * weights + bias
            y_pred = np.dot(X, self.weights) + self.bias
            
            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store cost
            cost = (1/(2*n_samples)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return np.dot(X, self.weights) + self.bias
        
    def score(self, X, y):
        """Calculate the coefficient of determination R^2."""
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - y_pred)**2)
        return 1 - (ss_residual / ss_total)

def generate_sample_data(n_samples=100, noise=10):
    """Generate sample data for linear regression."""
    np.random.seed(0)
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X.squeeze() + 1 + np.random.randn(n_samples) * noise
    return X, y

def plot_results(X, y, custom_pred, sklearn_pred, title="Linear Regression Comparison"):
    """Plot the results of different linear regression implementations."""
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')
    plt.plot(X, custom_pred, color='red', linewidth=2, label='Custom implementation')
    plt.plot(X, sklearn_pred, color='green', linewidth=2, label='Scikit-learn')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_regression_comparison.png')
    plt.close()

def main():
    # Generate sample data
    X, y = generate_sample_data(n_samples=100, noise=2)
    X_reshaped = X.reshape(-1, 1)  # Reshape for scikit-learn
    
    # Train custom model
    custom_model = LinearRegressionFromScratch(learning_rate=0.01, n_iterations=1000)
    custom_model.fit(X_reshaped, y)
    custom_pred = custom_model.predict(X_reshaped)
    custom_mse = mean_squared_error(y, custom_pred)
    custom_r2 = custom_model.score(X_reshaped, y)
    
    # Train scikit-learn model
    sklearn_model = LinearRegression()
    sklearn_model.fit(X_reshaped, y)
    sklearn_pred = sklearn_model.predict(X_reshaped)
    sklearn_mse = mean_squared_error(y, sklearn_pred)
    sklearn_r2 = sklearn_model.score(X_reshaped, y)
    
    # Print results
    print("Custom Implementation:")
    print(f"Weights: {custom_model.weights}, Bias: {custom_model.bias}")
    print(f"MSE: {custom_mse:.4f}, R²: {custom_r2:.4f}")
    print("\nScikit-learn Implementation:")
    print(f"Weights: {sklearn_model.coef_}, Bias: {sklearn_model.intercept_}")
    print(f"MSE: {sklearn_mse:.4f}, R²: {sklearn_r2:.4f}")
    
    # Plot results
    try:
        plot_results(X, y, custom_pred, sklearn_pred)
        print("\nResults plotted and saved as 'linear_regression_comparison.png'")
    except Exception as e:
        print(f"Could not generate plot: {str(e)}")

if __name__ == "__main__":
    # If matplotlib and scikit-learn are not installed, the script will still run
    # but will skip the plotting and scikit-learn comparison
    try:
        main()
    except ImportError as e:
        print(f"Error: {str(e)}")
        print("Please install required libraries: pip install numpy matplotlib scikit-learn")
