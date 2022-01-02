import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer, load_digits, load_wine

from sklearn.decomposition import PCA

from tqdm import tqdm

'''
TODO: Can be op
Class to perform KNN classification on a dataset

1. Split the dataset into training and test sets
2. Scale the data
3. Compute the distance between each point in the training set and the test point
4. Sort the distances
5. Compute the majority vote of the k nearest neighbors
6. Return the prediction
'''
class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def train(self, X, Y):
        self.X_train = X
        self.y_train = Y

    def _compute_euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))
    
    def predict(self, X_test:np.ndarray)-> np.ndarray:
        y_pred = np.zeros(X_test.shape[0])
        for i,x in enumerate(X_test):
            distances = [self._compute_euclidean_distance(x, x_train) for x_train in self.X_train]
            k_nearest_indices = np.argsort(distances)[:self.k].astype(int)
            k_nearest_labels = self.y_train[k_nearest_indices]
            y_pred[i] = np.argmax(np.bincount(k_nearest_labels))
        return y_pred

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        n_correct = np.sum(y_pred == y_test)
        return n_correct/len(y_test) * 100.0


if __name__ == "__main__":

    data = load_wine()
    X = data.data
    Y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print(f'X-Train set shape:{X_train.shape}\nX- Test set shape: {X_test.shape}\nY-Train set shape: {y_train.shape}\nY-Test set shappe: {y_test.shape}')


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # pca = PCA(n_components=3)
    # X_train = pca.fit_transform(X_train)
    # X_test = pca.fit_transform(X_test)

    # print(f'rX-Train set shape:{X_train.shape}\nrX- Test set shape: {X_test.shape}')

    accuracies = {'k':[],'accuracy':[]}
    for k in tqdm(range(0,20, 2)):
        k = 1 if k == 0 else k
        knn = KNN(k)
        knn.train(X_train, y_train)
        accuracies['k'].append(k)
        accuracies['accuracy'].append(knn.accuracy(X_test, y_test))

    fig, ax = plt.subplots()
    ax.plot(accuracies['k'], accuracies['accuracy'], '-o')
    ax.set_xlabel('K')
    ax.set_ylabel('Accuracy')
    ax.set_title('K vs Accuracy')

    for k, acc in zip(accuracies['k'], accuracies['accuracy']):
        ax.annotate(f'({k},{acc:.2f})', (k, acc+0.01))
    plt.show()

    knn = KNN(k=10)
    knn.train(X_train, y_train)
    print(knn.accuracy(X_test, y_test))


    y = knn.predict(X_test[2].reshape(1, -1))
    print(f'Predicted label: {y}, Actual label: {y_test[2]}')


