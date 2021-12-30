from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.datasets import make_blobs, make_classification, make_swiss_roll, make_moons  

import sys

# Create a class for k-means clustering algorithm
class KMeansClustering(object):
    def __init__(self, K:int, max_iter:int = 200) -> None:
        super().__init__()
        self.K = K
        self.max_iter = max_iter
        self.num_datapoints, self.num_feat = X.shape
        self.fitted_centroids = None
        self.inertia = 0

    def init_centroids(self, X:np.ndarray) -> np.ndarray:
        # centroids = np.zeros(shape=(self.K, self.num_feat))
        # for k in range(self.K):
        #     centroid = X[np.random.randint(1,len(X))]
        #     centroids[k] = centroid
        # return centroids

        centroids = []
        centroids.append(X[np.random.randint(1,len(X))])
        for _ in range(self.K-1):
            distances = []
            for x in X:
                d = sys.maxsize
                for i in range(len(centroids)):
                    temp_distance = np.sqrt(np.sum((x - centroids[i])**2))
                    if temp_distance < d:
                        d = temp_distance
                distances.append(d)
            distances = np.array(distances)
            max_idx = np.argmax(distances)
            centroids.append(X[max_idx])
            distances = []
        return np.array(centroids)
    
    def create_clusters(self, X:np.ndarray, centroids:np.ndarray) -> List[list]:
        clusters = [[] for _ in range(self.K)] # Create K empty clusters
        for p_idx, p in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum((p - centroids)**2, axis=1))) # Find closest centroid for each point using Euclidian distance
            clusters[closest_centroid].append(p_idx) # assign each data point_idx to the cluster(Centroid)
        return clusters
    
    def update_centroid(self, X:np.ndarray, clusters:List[list])-> np.ndarray:
        centroids = np.zeros(shape=(self.K, self.num_feat))
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis=0)
            centroids[idx] = new_centroid
        return centroids

    def plot_cluster(self, centroids, x, y):
        plt.scatter(x[:,0], x[:,1], c=y, s=50, cmap='viridis')
        plt.scatter(centroids[:,0], centroids[:,1], c='black', s=100, alpha=0.7, marker='x')
        plt.show()
    
    def plot_3d_cluster(self, centroids, x, y):
        ax = plt.axes(projection='3d')
        ax.scatter3D(x[:,0], x[:,1], x[:,2], c=y, s=20, alpha =0.3,cmap='viridis')
        ax.scatter3D(centroids[:,0], centroids[:,1], centroids[:,2], c='black', s=100, alpha=1.0, marker='o')
        plt.show()

    def get_y_label(self, clusters:List[list], X:np.ndarray):
        y_label = np.zeros(shape=(self.num_datapoints))
        for idx, cluster in enumerate(clusters):
            for point_idx in cluster:
                y_label[point_idx] = idx
        return y_label

    def predict(self, X:np.ndarray):
        pass

    def fit(self, X:np.ndarray):
        centroids = self.init_centroids(X)
        for i in range(self.max_iter):
            clusters = self.create_clusters(X, centroids)
            prev_centroids = centroids
            centroids = self.update_centroid(X, clusters)
            print(f'Centroids at iter {i+1}: {centroids[0]}')

            diff = prev_centroids - centroids
            if diff.any() < 0.0001:
                break

        self.fitted_centroids_ = centroids

        y_label = self.get_y_label(clusters, X)

        if self.num_feat == 2:
            self.plot_cluster(centroids,X, y_label)
        elif self.num_feat == 3:
            self.plot_3d_cluster(centroids, X, y_label)
        
        return y_label
        

if __name__ == "__main__":
    np.random.seed(45)
    K = 3
    num_of_features = 3
    num_of_samples = 1000
    X, _ = make_blobs(n_samples=num_of_samples, centers=K, n_features=num_of_features, cluster_std=2.0, random_state=1)
    # X, _ = make_classification(n_samples=num_of_samples, n_features=num_of_features, n_redundant=0, n_informative=2, n_classes=K, n_clusters_per_class=1)
    # X, _ = make_moons(n_samples=num_of_samples, noise=0.1)

    kmeans = KMeansClustering(K, max_iter=30)
    y_label = kmeans.fit(X)
    
    