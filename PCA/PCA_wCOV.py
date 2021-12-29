import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt

from sklearn.datasets import load_digits, load_wine
from sklearn.preprocessing import StandardScaler

'''
Class to perform PCA on a dataset with covariance matrix

1. Compute the row-wise mean of the dataset
2. Subtract the mean from each row
3. Compute the covariance matrix
4. Compute the eigenvalues and eigenvectors of the covariance matrix
5. Sort the eigenvalues and eigenvectors in descending order
6. Compute the cumulative variance explained by the eigenvalues
7. Compute the number of dimensions needed to explain 95% of the variance
8. Project the dataset onto the new subspace
9. Plot the projected dataset
'''

class PCA(object):
    def __init__(self,n_components:int=2) -> None:
        self.n_components = n_components
        self.fitted_covariance_matrix = None
        self.fitted_eigenvalues = None
        self.fitted_eigenvectors = None
        self.fitted_components = None
        self.fitted_explained_variance = None

    def _get_covariance_matrix(self, data:np.ndarray):
        # Compute the covariance matrix
        self.fitted_covariance_matrix = np.cov(data.T)
        return self.fitted_covariance_matrix

    def _calculate_eigens(self):
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eig_val, eig_vec = np.linalg.eig(self.fitted_covariance_matrix)
        idx = np.argsort(eig_val)[::-1]
        self.fitted_eigenvalues = eig_val[idx]
        self.fitted_eigenvectors = eig_vec[:,idx]
        return self.fitted_eigenvalues, self.fitted_eigenvectors

    def _project_data(self, data:np.ndarray):
        # Project the dataset onto the new subspace
        self.fitted_components = data @ self.fitted_eigenvectors[:,:self.n_components]
        return self.fitted_components

    def _calculate_explained_variance(self):
        # Compute the cumulative variance explained by the eigenvalues
        self.fitted_explained_variance = np.abs(self.fitted_eigenvalues) / np.sum(self.fitted_eigenvalues)
        return self.fitted_explained_variance

    def fit(self, data:np.ndarray) -> None:
        # Fit the PCA model
        self._get_covariance_matrix(data)
        self._calculate_eigens()
        self._calculate_explained_variance()
        self._project_data(data)

    def fit_transform(self, data:np.ndarray) -> np.ndarray:
        # Fit and transform the PCA model
        self.fit(data)
        return self.fitted_components

if __name__ == "__main__":
    # Load the dataset
    wine = load_wine()
    data = wine.data
    print(data.shape)
    labels = wine.target
    print(labels.shape)

    # Standardize the dataset
    scaler = StandardScaler()
    data = scaler.fit_transform(data)

    # Create a PCA object
    pca = PCA(n_components=3)
    pca.fit(data)
    print(pca.fitted_components.shape)
    print(pca.fitted_explained_variance.shape)
    print(pca.fitted_eigenvectors[:,:3].shape)

    # Plot the projected data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca.fitted_components[:,0], pca.fitted_components[:,1], pca.fitted_components[:,2], c=labels)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.show()

    # Plot the cumulative variance explained
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(pca.fitted_explained_variance)
    plt.show()

    # Plot covariance matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(pca.fitted_covariance_matrix)
    plt.show()