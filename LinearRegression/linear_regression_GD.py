import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes, load_boston, fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

'''
Class to implement Multiple Linear Regression Model.

1. Split the dataset into training and test sets
2. Scale the data
3. Create W matrix(weights/theta) of shape (num_features+[1], 1)
5. Compute the cost function
6. Compute the Gradients with respect to theta(Weights)
7. Update the weights using Gradient Descent(W_prev = W_new - learning_rate * Gradients)
8. Repeat the above steps until convergence
9. Return the weights
'''

class LinearRegression_GD(object):
    def __init__(self, lr:float = 1e-4, iteration:int = 2000, alpha:float=0.1) -> None:
        super().__init__()
        self.lr = lr
        self.iteration = iteration
        self.alpha = alpha
        self.W = None
    
    def _forward_pass(self, X:np.ndarray) -> np.ndarray:
        '''
        Forward pass of the input data and weights
        args:
            X: np.ndarray(n,m), Input data
        return:
            y_pred: np.ndarray
        '''
        return X @ self.W

    def loss(self, y_pred:np.ndarray, y_true:np.ndarray)->np.float32:
        '''
        Compute the loss function
        args:
            y_pred: np.ndarray
            y_true: np.ndarray
        return:
            loss: np.float32
        '''
        return np.sum((y_pred-y_true)**2)/y_pred.shape[0]
    
    def _l2_regulaization(self)->np.ndarray:
        '''
        Compute the l2 regularization term

        L2 Regularization is given by:
        L2 = lambda/2 * ||W||^2

        NOTE: We do not regularize bias term, therefore we do not include the first element of self.W in the regularization
        return:
            l2_reg: np.ndarray(1,1)
        '''
        return self.alpha * np.linalg.norm(self.W[1:],ord=2)
    
    def _gradient_penalty(self):
        '''
        Compute the gradient penalty term for penalizing the model to get a regularizing effect.

        gradient of L2 regularization term is given by:
        dReg/dW = lambda * W

        return:
            gradient_penalty: np.ndarray(m,1)
        '''
        gradient_penalty = np.asarray(self.alpha * self.W[1:])

        # Insert Zero for the bias term in order to avoid regularization of the bias term
        gradient_penalty = np.insert(gradient_penalty, 0, 0, axis=0)

        return gradient_penalty

    def _gradient_descent(self, X:np.ndarray, y:np.ndarray, y_pred:np.ndarray)->None:
        '''
        Gradient Descent function
        args:
            X: np.ndarray(n,m), Input data
            y: np.ndarray(n,1), Output data
        return:
            None
        '''
        gp = self._gradient_penalty()/X.shape[0]
        dldw = 2*np.dot(X.T, (y_pred - y))/X.shape[0] # Gradient of the loss function with respect to the weights(2*X.T*(y_pred-y)/n)
        dldw = dldw + gp # Add the gradient penalty term to the gradient of the loss function
        self.W = self.W - self.lr * dldw # Update the weights
    
    def fit(self, X:np.ndarray, y:np.ndarray):
        '''
        Train the model using Gradient Descent
        args:
            X: np.ndarray(n,m), Input data
            y: np.ndarray(n,1), Output data
        return:
            W: np.ndarray(m,1), Trained weights
            b: np.ndarray(1,1), Trained bias/intercept
        '''
        ones = np.ones((X.shape[0],1)) 
        X = np.append(ones, X, axis=1) # Add a column of ones to X for the bias term

        # Initialize the weights
        self.W = np.random.normal(size=(X.shape[1], 1))
        print(self.W.shape)
        print(X.shape)

        # Iterate over the number of iterations
        for i in range(self.iteration+1):
            y_pred = self._forward_pass(X)
            loss = self.loss(y_pred, y) + self._l2_regulaization()/X.shape[0]
            self._gradient_descent(X, y, y_pred)

            if i%200 == 0:
                print(f'Iteration: {i}, Loss: {loss}')
        
        return self.W[1:], self.W[0]
    def predict(self, X:np.ndarray)->np.ndarray:
        '''
        Predict the output of the model
        NOTE: self.W matrix contains the bias term as the first element and the weights as the rest of the elements
        args:
            X: np.ndarray(n,m), Input data
        return:
            y_pred: np.ndarray(n,1)
        '''
        return np.dot(X, self.W[1:]) + self.W[0]
    
    @property
    def weights(self)->np.ndarray:
        return self.W[1:]
    
    @property
    def bias(self)->np.ndarray:
        return self.W[0]

if __name__ == "__main__":
    data = fetch_california_housing()
    X = data.data
    y = data.target

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f'X_train shape: {X_train.shape}')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Create a Linear Regression Model
    linear_reg = LinearRegression_GD(lr=0.01, iteration=2000, alpha=3.0)
    w, bias = linear_reg.fit(X_train, np.expand_dims(y_train, axis=1))
    print(f'Trained Weights: {w}')
    print(f'Trained Bias: {bias}')

    # Test the model
    y_pred = linear_reg.predict(X_test)
    print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
    print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
    # print(f'Mean Squared Log Error: {mean_squared_log_error(y_test, y_pred)}')

    # Print the predicted values and actual values side by side
    for y_pred, y_actual in zip(y_pred[:10], y_test[:10]):
        print(f'Predicted Value: {y_pred}, Actual Value: {y_actual}')