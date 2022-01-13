import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize, Bounds, LinearConstraint
from scipy.sparse import data

from sklearn.datasets import make_moons, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel, linear_kernel
from sklearn.metrics import accuracy_score


'''
Class to implement the SVM algorithm using Lagrange multipliers.

References:
1. https://www.svm-tutorial.com/2014/11/svm-understanding-math-part-1/
2. https://www.youtube.com/watch?v=_PwhiWxHK8o&t=560s (MIT open course on SVM)
3. https://www.youtube.com/watch?v=hCOIMkcsm_g&list=PLLssT5z_DsK-h9vYZkQkYNWcItqhlRJLN&index=70 (Andrew Ng)
4. Mathematics for Machine Learning by Marc Peter Deisenroth, A. Aldo Faisal and Cheng Soon Ong- Chapter 12
5. https://machinelearningmastery.com/a-gentle-introduction-to-method-of-lagrange-multipliers/
'''

class SVM:
    def __init__(self, C:float = 1.0, kernel:str='rbf', degree:int =2, gamma:float = 1.0) -> None:
        self.C = C
        self.w = None
        self.b = None
        self.alpha = None
        self.X = None
        self.y = None
        self.small_number = 1e-8
        self.degree = degree
        self.gamma = gamma
        self.kernel = kernel
        self._supported_kernel = ['rbf', 'linear', 'poly']
    
    def _kernel(self, X1:np.ndarray, X2:np.ndarray):
        '''
        The function defines the kernel function
        args:
            X: np.ndarray, feature matrix
            d: int, degree of the polynomial kernel
            gamma: float, gamma parameter for the RBF kernel
        returns:
            float, kernel value
        raises:
            ValueError, if the kernel is not supported
        '''
        if self.kernel == 'rbf':
            return rbf_kernel(X1,X2, gamma=self.gamma)
        
        elif self.kernel == 'linear':
            return linear_kernel(X1,X2)

        elif self.kernel == 'poly':
            return polynomial_kernel(X1, X2, degree=self.degree, gamma=self.gamma)

        else:
            raise ValueError(f'Kernel not supported, please use one of the following: {self._supported_kernel}')

    def _lagrangian_dual(self, alpha:np.ndarray)->np.float:
        '''
        The fucntion defines lagrangian dual.
        The lagrangian dual for SVM is defined as:
            L = 1/2 * sum(i=0,n)(sum(j=0,n)(alpha_i * alpha_j * y_i * y_j * K(x_i, x_j))) + sum(i=0,n)(alpha_i),
                where K is the kernel function.
                alpha_i is the Lagrange multiplier for the ith instance.
                y_i is the label for the ith instance.
                x_i is the ith instance.

        subjet to constraints:
            1. 0 < alpha_i < C for all i
            2. sum(0,n)(y_i * alpha_i) = 0
        
        args:
            alpha: np.ndarray, lagrange multipliers
            X: np.ndarray, dataset  
            y: np.ndarray, labels
        returns:
            float, lagrangian dual
        '''
        alpha=alpha.reshape(-1,1)
        self.y = self.y.reshape(-1,1)
        L = (1/2) * np.sum((alpha @ alpha.T) * (self.y @ self.y.T) * self._kernel(self.X, self.X)) - np.sum(alpha)
        return L
    
    def _constraint_dual(self)-> LinearConstraint:
        '''
        The function defines the constraints for lagrangian dual.
        The constraint is:
            sum(0,n)(y_i * alpha_i) = 0
            which is equivalent to:
            y.T @ alpha = 0
        args:
            y: np.ndarray, labels
        returns:
            LinearConstraint, constraint object
        '''
        # A = label matrix of shape(1,n)
        # alpha = lagrange multipliers of shape(1,n)
        #NOTE: The ub and lb are set to [0] which indicates that the constraint is a equality constraint
        #NOTE: the alpha will be taken from minimize() function's x argument which here is Alpha(lagrangian multipliers)

        return LinearConstraint(A=self.y, lb=[0], ub=[0])
    
    def _constraint_bounds(self, n:int)-> Bounds:
        '''
        The function defines the bounds for lagrangian dual.
        The bounds are:
            0 <= alpha_i <= C for all i
        args:
            n: int, number of training instances
        returns:
            Bounds, constraint object
        '''
        # lb = lower bound of shape(1,n)
        # ub = upper bound of shape(1,n) for all alpha_i 
        # where 0 <= alpha_i <= C
        return Bounds(lb=np.zeros(n), ub=self.C*np.ones(n))
    
    def _optimize_alphas(self)->np.ndarray:
        '''
        This function optimizes the lagrangian dual for Alphas.
        args:
            X: np.ndarray, dataset
            y: np.ndarray, labels
        returns:
            np.ndarray, lagrange multipliers
        '''
        n, _ = self.X.shape
        # initialize random alpha values of size n and multiply by C so that it gets bounded by 0 and C
        alpha_initial = np.random.rand(n)*self.C

        # constraints and bounds for optimization
        constraints = self._constraint_dual()
        bounds = self._constraint_bounds(n)

        # optimize the lagrangian dual
        options = {
            'disp': True,
            'maxiter': 100,
            # 'ftol': 0.01,
            # 'eps': 0.01
            }

        res = minimize(self._lagrangian_dual, alpha_initial, method='SLSQP', constraints=[constraints], bounds=bounds, options=options)

        # return the lagrangian multipliers
        self.alpha = res.x

        return self.alpha

    
    def _get_support_vector_indices(self):
        '''
        This function returns the support vectors indices.
        '''
        sv = (self.alpha > self.small_number) * (self.alpha < self.C).flatten()
        if len(sv) == 0:
            return None
        return sv
    
    def _get_w(self)->np.ndarray:
        '''
        This function calculates the weight vector.
        returns:
            np.ndarray, weight vector
        '''
        sv = self._get_support_vector_indices()
        if sv is None:
            return None
        self.w = np.dot(self.X[sv,:].T, self.alpha[sv]*self.y[sv])
        return self.w.T
    
    def _get_b(self)-> np.float:
        '''
        This function calculates the bias term.
        returns:
            np.float, bias term
        '''
        # calculate the bias term
        sv = self._get_support_vector_indices()
        if sv is None:
            return None
        self.b = np.mean(self.y[sv, np.newaxis] - self.alpha[sv] * self.y[sv, np.newaxis] * self._kernel(self.X[sv],self.X[sv]))

        return self.b


    def fit(self, X:np.ndarray, y:np.ndarray)->None:
        '''
        This function fits the model to the dataset.
        args:
            X: np.ndarray, dataset
            y: np.ndarray, labels
        '''
        y[y==0] = -1
        self.X = X
        self.y = y
        self.alpha = self._optimize_alphas()
        self.w = self._get_w()
        self.b = self._get_b()


    def predict(self, X_test:np.ndarray)->np.ndarray:
        '''
        This function predicts the labels for the test dataset.
        args:
            X: np.ndarray, Test dataset
        returns:
            np.ndarray, predicted labels
        '''
        sv = self._get_support_vector_indices()
        if sv is None:
            return None
        y_pred = np.sum((self.alpha[sv, np.newaxis] @ self.y[sv].T) @ self._kernel(X_test, self.X[sv]).T, axis=0)
        predicted_labels = np.sign(y_pred + self.b)
        predicted_labels[predicted_labels==-1]= 0
        return predicted_labels

    @property
    def support_vectors_(self)->np.ndarray:
        '''
        This function returns the support vectors.
        returns:
            np.ndarray, support vectors
        '''
        assert self.X is not None, "Dataset is not fitted yet, Call fit() first"
        sv = self._get_support_vector_indices()
        return self.X[sv]
    
    @property
    def weights_(self)->np.ndarray:
        '''
        This function returns the weights.
        returns:
            np.ndarray, weights
        '''
        assert self.X is not None, "Dataset is not fitted yet, Call fit() first"
        return self.w
    
    @property
    def bias_(self)->np.float:
        '''
        This function returns the bias.
        returns:
            np.float, bias
        '''
        assert self.X is not None, "Dataset is not fitted yet, Call fit() first"
        return self.b

# A utility function to plot the decision boundary
# This function is taken from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/ac5dcd03a40a08a8af7e1a67ade37f28cf88db43/ML/algorithms/svm/utils.py#L31
# And modified to plot support vectors
def plot_contour(X, y, svm:SVM, title:str):
    # plot the resulting classifier
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)

    # plt the points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], c='black', s=80, marker='x')
    plt.title(title)
    plt.show()

def main():
    np.random.seed(42)

    # data = make_moons(n_samples=200, noise=0.1, random_state=42)
    data = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2)
    X = data[0]
    y = data[1]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

    # Create a SVM classifier
    svm = SVM(C=1.0, kernel='rbf', degree=2, gamma=1.0)
    svm.fit(X_train, y_train)

    predicted = svm.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, predicted)}')

    if svm.kernel == 'linear':
        title = f'Linear Kernel, C={svm.C}'
    elif svm.kernel == 'rbf':
        title = f'RBF Kernel, C={svm.C}, gamma={svm.gamma}'
    elif svm.kernel == 'poly':
        title = f'SVM with C={svm.C}, kernel={svm.kernel}, degree={svm.degree}, gamma={svm.gamma}'
    plot_contour(X_train, y_train, svm, title)

if __name__ == "__main__":
    main()