import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc

'''
Class to implement Logistic Regression Model.

'''

class LogisticRegression(object):
    def __init__(self, lr:float=0.01, iteration:int = 2000, alpha:float=1.0) -> None:
        super().__init__()
        self.lr = lr
        self.iteration = iteration
        self.alpha = alpha
        self.W = None
        self.b = None

    def _sigmoid(self, x:np.ndarray) -> np.ndarray:
        '''
        Sigmoid function
        args:
            x: np.ndarray, Input array
        return:
            y: np.ndarray, sigmoid of x
        '''
        return 1/(1+np.exp(-x))
    
    def _forward_pass(self, X:np.ndarray) -> np.ndarray:
        '''
        Forward pass of the input data and weights
        args:
            X: np.ndarray(n,m), Input data
        return:
            y_pred: np.ndarray
        '''
        return self._sigmoid(np.dot(X, self.W) + self.b)
    
    def loss(self, y_pred:np.ndarray, y_true:np.ndarray)->np.float32:
        '''
        Compute Cross Entropy Loss

        Cross Entropy loss is given as:
        CE = -sum(y * log(y') + (1-y) * log(1-y'))/n

        args:
            y_pred: np.ndarray, predicted values
            y_true: np.ndarray, true values
        return:
            loss: np.float32
        '''
        return -np.sum(y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))/y_pred.shape[0]
    
    def _l2_regularization(self):
        '''
        Compute the l2 regularization term

        L2 Regularization is given by:
        L2 = lambda/2 * ||W||^2

        return:
            l2_reg: np.ndarray(1,1)
        '''
        
        return self.alpha * np.linalg.norm(self.W, ord=2)
    
    def _gradient_penalty(self):
        '''
        Compute the gradient penalty term for penalizing the model to get a regularizing effect.

        gradient of L2 regularization term is given by:
        dReg/dW = lambda * W

        return:
            gradient_penalty: np.ndarray(m,1)
        '''
        gradient_penalty = np.asarray(self.alpha * self.W)
        return gradient_penalty

    def _gradient_descent(self,X:np.ndarray, y:np.ndarray, y_pred:np.ndarray)->None:
        '''
        Gradient Descent function
        args:
            X: np.ndarray(n,m), Input data
            y: np.ndarray(n,1), Output data
        '''
        gp = self._gradient_penalty()/X.shape[0] # Compute the gradient penalty

        # Calculate the gradient of the loss function with respect to the weights and bias
        dldw = np.dot(X.T, (y_pred - y))/X.shape[0]
        dldw = dldw + gp # Add the gradient penalty to the gradient of the loss function
        dldb = np.sum(y_pred - y)/X.shape[0]

        # Update the weights and bias
        self.W = self.W - self.lr * dldw
        self.b = self.b - self.lr * dldb
    
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
        # Initialize the weights and bias
        self.W = np.random.normal(size=(X.shape[1], 1))
        self.b = np.random.normal(size=(1, 1))

        # Iterate over the number of iterations
        for i in range(self.iteration+1):
            y_pred = self._forward_pass(X) # Forward pass
            loss = self.loss(y_pred, y) + self._l2_regularization()/X.shape[0] # Compute the log loss
            self._gradient_descent(X, y, y_pred) # Update the weights and bias using gradient descent

            if i % 200 == 0:
                print(f'Iteration: {i}, Loss: {loss}')
        
        return self.W, self.b
    
    def predict(self, X:np.ndarray, threshold:float = 0.5)->np.ndarray:
        '''
        Predict the output on new data 
        args:
            X: np.ndarray(n,m), Input data
            threshold: float, Threshold for the prediction
        return:
            y_pred: np.ndarray(n,1), Predicted values
        '''
        assert 0 < threshold < 1, 'Threshold must be between 0 and 1'
        y_pred = self._forward_pass(X)
        return np.where(y_pred > 0.5, 1, 0)
    
    @property
    def weights(self):
        return self.W
    
    @property
    def bias(self):
        return self.b

if __name__ == "__main__":
    # Load the dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target

    # Split the data into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'Training data shape: {X_train.shape}')
    print(f'Testing data shape: {X_test.shape}')

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Train the model
    log_reg = LogisticRegression(lr=0.1, iteration=4000, alpha=0.1)
    W, b = log_reg.fit(X_train, y_train[:, np.newaxis])
    print(f'Weights: \n{W}')
    print(f'Bias: {b}\n')

    # Predict the output on the test data
    y_pred = log_reg.predict(X_test)
    print(f'Accuracy: {(accuracy_score(y_test, y_pred)*100).round(2)}%')
    print(f'Classification Report: \n{classification_report(y_test, y_pred)}')
    print(f'Confusion Matrix: \n{confusion_matrix(y_test, y_pred)}')

    # Plot Confusion matrix
    plt.figure(figsize=(10,10))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.xticks(np.arange(2), ['Benign', 'Malignant'])
    plt.yticks(np.arange(2), ['Benign', 'Malignant'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Compute the ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f'ROC AUC: {roc_auc}')

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()
