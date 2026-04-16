"""XGmodel: Logistic Regression model for Expected Goals (xG) prediction.

This module implements an extended Gradient-based logistic regression model
for binary classification tasks, with support for feature normalization,
PCA dimensionality reduction, and L2 regularization.
"""

from .model import model
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt

class GradianDescentModel(model):
    """Logistic Regression model with advanced features for xG prediction.
    
    This class extends the base model with gradient descent optimization,
    feature normalization, PCA reduction, and L2 regularization capabilities.
    """
    
    def __init__(self, lr: float = 0.01, num_iters: int = 100,
                 random_init_weight: bool = True, normalize_data: bool = True,
                 regularization: bool = False, Lambda: int = 1, reduce_dimension: bool = False,
                 reduction_size: Optional[int] = None, **kwargs) -> None:
        """Initialize the XGmodel with hyperparameters.
        
        Args:
            lr (float, optional): Learning rate for gradient descent. Defaults to 0.01.
            num_iters (int, optional): Number of iterations for training. Defaults to 100.
            random_init_weight (bool, optional): Whether to randomly initialize weights. Defaults to True.
            normalize_data (bool, optional): Whether to normalize input features. Defaults to True.
            regularization (bool, optional): Whether to apply L2 regularization. Defaults to False.
            Lambda (int, optional): Regularization parameter. Defaults to 1.
            reduce_dimension (bool, optional): Whether to apply PCA reduction. Defaults to False.
            reduction_size (int, optional): Target dimension after PCA reduction. Defaults to None.
            
        Raises:
            ValueError: If lr is not positive or num_iters is not positive.
        """
        # Validate input parameters
        if lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {lr}")
        if num_iters <= 0:
            raise ValueError(f"Number of iterations must be positive, got {num_iters}")
        if reduction_size is not None and reduction_size <= 0:
            raise ValueError(f"Reduction size must be positive, got {reduction_size}")
            
        super().__init__()
        
        # Optimization parameters
        self.lr: float = lr
        self.num_iters: int = num_iters
        
        # Initialization parameters
        self.random_init_weight: bool = random_init_weight
        
        # Feature preprocessing parameters
        self.normalize_data: bool = normalize_data
        self.reduce_dimension: bool = reduce_dimension
        self.reduction_size: Optional[int] = reduction_size
        
        # Regularization parameters
        self.regularization: bool = regularization
        self.Lambda: int = Lambda
        
        # Attributes to store normalization statistics
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.w: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, Y: np.ndarray, show: bool = False) -> None:
        """Train the logistic regression model using gradient descent.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features).
            Y (np.ndarray): Training labels of shape (n_samples,) or (n_samples, 1).
            show (bool, optional): Whether to display loss curve plot. Defaults to False.
            
        Raises:
            ValueError: If X and Y have incompatible shapes.
            TypeError: If X or Y are not numpy arrays.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        if not isinstance(Y, np.ndarray):
            raise TypeError(f"Y must be numpy array, got {type(Y)}")
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X and Y must have same number of samples. Got X: {X.shape[0]}, Y: {Y.shape[0]}")
        
        # Feature preprocessing: Normalization
        if self.normalize_data or self.reduce_dimension:
            X = self.featureNormalization(X)

        # Feature preprocessing: Dimensionality reduction
        if self.reduce_dimension:
            if self.reduction_size is None:
                raise ValueError("reduction_size must be specified when reduce_dimension is True")
            X, total_variance = self.PCA(X, self.reduction_size)
            print(f"Roughly {round(total_variance*100, 2)}% of the total variation in the dataset was captured.")

        # Add bias term to features
        X = self.addBias(X)
        C_history: list = []
        m: int = X.shape[1]
        
        # Initialize weights
        if self.random_init_weight:
            # Random initialization from uniform distribution [0, 1)
            self.w = np.random.rand(m, 1)
        else:
            # Zero initialization
            self.w = np.zeros((m, 1))
        
        # Gradient descent optimization loop
        pbar = tqdm(range(self.num_iters), desc="Training XGmodel")

        for i in pbar:
            # Compute cost and gradient
            C, grad = self.costFunction(self.w, X, Y)
            C_history.append(C)
            
            # Update weights using gradient descent
            self.w = self.w - self.lr * grad
            
            # Update progress bar with current loss
            pbar.set_description(f"Loss: {round(C, 2)}")

        # Display loss curve if requested
        if show:
            plt.figure(figsize=(10, 6))
            plt.plot(C_history, linewidth=2)
            plt.xlabel("Number of Iterations")
            plt.ylabel("Loss (Binary Cross-Entropy)")
            plt.title("Training Loss Over Time")
            plt.grid(True, alpha=0.3)
            plt.show()


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on new data.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple of (predicted_labels, probabilities)
                - predicted_labels: Binary predictions (0 or 1) of shape (n_samples,)
                - probabilities: Prediction probabilities of shape (n_samples, 1)
                
        Raises:
            RuntimeError: If model has not been trained.
            TypeError: If X is not a numpy array.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        
        # Check if model has been trained
        if self.w is None:
            raise RuntimeError("Model must be trained before making predictions. Call fit() first.")
        
        # Apply same preprocessing as during training
        if self.normalize_data or self.reduce_dimension:
            # Check that normalization statistics are available
            if self.mean is None or self.std is None:
                raise RuntimeError("Model must be trained before prediction")
            
            # Normalize features using training statistics
            X = (X - self.mean) / self.std
        
        # Apply PCA if reduction was used during training
        if self.reduce_dimension:
            if self.reduction_size is None:
                raise RuntimeError("reduction_size must be set for PCA")
            X, _ = self.PCA(X, self.reduction_size)

        # Add bias term
        X = self.addBias(X)
        
        # Compute prediction probabilities
        probs = self._sigmoid(X @ self.w)
        
        # Apply threshold for binary classification
        preds = (probs >= 0.5).astype(int).flatten()
        
        return preds, probs
            



    def featureNormalization(self, X: np.ndarray) -> np.ndarray:
        """Normalize features using Z-score standardization.
        
        Computes mean and standard deviation on the input data and stores them
        for later use during prediction. Formula: X_normalized = (X - mean) / std
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Normalized features of same shape as input.
            
        Raises:
            TypeError: If X is not a numpy array.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        
        # Compute normalization statistics
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, ddof=1)  # Use Bessel's correction (ddof=1)
        
        # Avoid division by zero: replace zero std with 1
        self.std[self.std == 0] = 1.0
        
        # Apply normalization
        X_norm = (X - self.mean) / self.std

        return X_norm


    def addBias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term (column of ones) to the feature matrix.
        
        Prepends a column of ones to the input features to represent the bias/intercept term
        in the linear model. This bias column will always be multiplied by the same weight.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            
        Returns:
            np.ndarray: Extended features of shape (n_samples, n_features+1) with bias as first column.
            
        Raises:
            TypeError: If X is not a numpy array.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got {X.ndim}D")
        
        # Create bias column (all ones)
        bias_column = np.ones((X.shape[0], 1))
        
        # Concatenate bias column with original features
        X_extended = np.column_stack((bias_column, X))

        return X_extended
    
    def costFunction(self, w: np.ndarray, X: np.ndarray, Y: np.ndarray) -> Tuple[float, np.ndarray]:
        """Compute cost (loss) and gradient for logistic regression.
        
        Computes binary cross-entropy loss with optional L2 regularization.
        Formula without regularization: J = -1/m * sum(Y*log(h) + (1-Y)*log(1-h))
        With L2 regularization: J = J + (Lambda/(2m)) * ||w||^2 (excluding bias term)
        
        Args:
            w (np.ndarray): Weight vector of shape (n_features+1, 1).
            X (np.ndarray): Input features of shape (n_samples, n_features+1) (includes bias).
            Y (np.ndarray): Target labels of shape (n_samples,) or (n_samples, 1).
            
        Returns:
            Tuple[float, np.ndarray]: Tuple of (cost, gradient)
                - cost: Scalar loss value
                - gradient: Weight gradients of shape (n_features+1, 1)
                
        Raises:
            ValueError: If input shapes are incompatible.
        """
        # Input validation
        if not isinstance(w, np.ndarray) or not isinstance(X, np.ndarray) or not isinstance(Y, np.ndarray):
            raise TypeError("All inputs must be numpy arrays")
        
        # Number of training samples
        m = X.shape[0]
        
        if m == 0:
            raise ValueError("Training set cannot be empty")
        
        # Compute predictions
        h = self._sigmoid(X @ w)
        
        # Reshape Y to column vector if needed
        Y = Y.reshape(-1, 1)
        
        # Compute cost and gradient
        if self.regularization:
            # L2 regularization: exclude bias term (weight 0)
            w_regularized = np.vstack((np.array([0]), w[1:, :]))
            
            # Regularization penalty term
            regularization_penalty = self.Lambda * (w_regularized.T @ w_regularized) / (2 * m)
            
            # Binary cross-entropy loss with regularization
            C = (1 / m) * np.sum((-Y) * np.log(h) - ((1 - Y) * np.log(1 - h))) + regularization_penalty
            
            # Gradient with regularization
            grad = (X.T @ (h - Y) + self.Lambda * w_regularized) / m
        else:
            # Binary cross-entropy loss without regularization
            C = (1 / m) * np.sum((-Y) * np.log(h) - ((1 - Y) * np.log(1 - h)))
            
            # Gradient without regularization
            grad = (X.T @ (h - Y)) / m
        
        # Ensure C is a scalar
        C = float(np.squeeze(C))
        
        return C, grad


    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation function.
        
        Sigmoid function is defined as: σ(z) = 1 / (1 + e^(-z))
        Maps any real value to range (0, 1), suitable for binary classification.
        
        Args:
            z (np.ndarray): Input array of any shape.
            
        Returns:
            np.ndarray: Sigmoid output of same shape as input, values in (0, 1).
            
        Note:
            For numerical stability with large negative values, clipping may be applied.
        """
        # Apply sigmoid function with numerical stability
        # Clip z to prevent overflow in exp
        z_clipped = np.clip(z, -500, 500)
        g = 1 / (1 + np.exp(-z_clipped))
        return g
    

    def PCA(self, X: np.ndarray, reduction_size: int = 2) -> Tuple[np.ndarray, float]:
        """Apply Principal Component Analysis for dimensionality reduction.
        
        Performs PCA by:
        1. Normalizing features
        2. Computing covariance matrix
        3. Finding eigenvalues and eigenvectors
        4. Selecting top k principal components
        5. Projecting data onto reduced dimensional space
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features).
            reduction_size (int, optional): Target number of dimensions. Defaults to 2.
            
        Returns:
            Tuple[np.ndarray, float]: Tuple of (reduced_data, explained_variance_ratio)
                - reduced_data: Dimensionality-reduced features of shape (n_samples, reduction_size)
                - explained_variance_ratio: Fraction of total variance retained (value in [0, 1])
                
        Raises:
            ValueError: If reduction_size is invalid or larger than number of features.
            TypeError: If X is not a numpy array.
        """
        # Input validation
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")
        if reduction_size <= 0:
            raise ValueError(f"reduction_size must be positive, got {reduction_size}")
        if reduction_size > X.shape[1]:
            raise ValueError(f"reduction_size ({reduction_size}) cannot exceed number of features ({X.shape[1]})")
        
        # Normalize features (applies Z-score normalization)
        X = self.featureNormalization(X)
        
        # Compute covariance matrix (with Bessel's correction)
        covariance_matrix = np.cov(X, ddof=1, rowvar=False)
        
        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        
        # Sort by eigenvalues in descending order (importance)
        order_of_importance = np.argsort(eigenvalues)[::-1]
        sorted_eigenvalues = eigenvalues[order_of_importance]
        sorted_eigenvectors = eigenvectors[:, order_of_importance]

        # Compute explained variance ratio
        explained_variance = sorted_eigenvalues / np.sum(sorted_eigenvalues)

        # Project data onto principal components
        X_reduced = np.matmul(X, sorted_eigenvectors[:, :reduction_size])

        # Compute total explained variance for selected components
        total_explained_variance = float(np.sum(explained_variance[:reduction_size]))

        return X_reduced, total_explained_variance
