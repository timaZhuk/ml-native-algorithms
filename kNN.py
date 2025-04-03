# Dataset and library imports
import numpy as np
from sklearn import datasets
# for splitting the data into train and test
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import Counter

# color mapping list
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

#--Dataset
iris = datasets.load_iris()
# X - feature vectors (X1, X2, X3, X4)
# y - classes [0, 1, 2]
X, y = iris.data, iris.target

# ---split the dataset into traning and test sets (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1234)

# print(X_train.shape)  # (120, 4) , 120-samples, 4-features
# print(y_train.shape) # (120,)  , 120-classes for 120 samples

# plot X1, X2 features

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolors='k', s=20)
plt.show()

# eucleadean distance 
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

# ------kNN class

class KNN:

    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # X - could be a test data
        # we loop through samples and apply _predict method
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # x_train vector [x1, x2, x3, x4] features vector
        # 1 compute distances (from point x to x_train points)
        distances = [euclidean_distance(self.x, x_train)for x_train  in self.X_train]


        # 2 get k nearest samples, labels
        # get indices of k-nearest samples (by distance): sort array for k-closest
        # distances
        k_indices = np.argsort(distances)[:self.k]
        # get labels for k-nearest points
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 3 majority vote, most common class label
        # chose most common label in a list
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    


# --- kNN class implementation
clf = KNN(k=3)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = np.sum(predictions == y_test)/len(y_test)
print(acc)
# 0.96,66
