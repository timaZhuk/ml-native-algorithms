import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# creating random data for Linear regression approximation
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

#print(X_train.shape) # (80, 1) -- row = samples = 80, column = 1-feature
# print(y_train.shape) # (80,)


#plot the data
# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0], y, color="b", marker = "o", s = 30)
# plt.show()

#split data to training and test 80% / 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Linear regression class

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters=1000 ):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias

            # derivatives of loss function
            dw = (1/n_samples) *np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # update components
            self.weights -=self.lr*dw
            self.bias -=self.lr *db

# using test dataset for predictions
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


# MSE function
def mse(y_true, y_predicted):
    return np.mean((y_true-y_predicted)**2)

# Using LinearRegression class
regressor = LinearRegression()

regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

# calculate MSE error=loss
mse_value = mse(y_test, predicted)
print(mse_value)

# plot the model results
y_pred_line = regressor.predict(X)
cmap = plt.get_cmap('viridis')

fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s=10)
plt.plot(X, y_pred_line, color='black', linewidth=2, label="Prediction")
plt.show()