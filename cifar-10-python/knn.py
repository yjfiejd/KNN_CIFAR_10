import numpy as np

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def compute_distances_no_loops(self, X):
        """
                公式: (x - y) ^2 = x^2 + y^2 -2xy
                1 - np.sum( (self.X_train)**2, axis=1)
                    计算训练集列方向的和 得到列方向的距离数组 即 y^2
                4 - 计算 xy 矩阵相乘
                5 - 计算 x^2 + y^2 - 2xy 的值
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        test_sum = np.sum(np.square(X), axis=1)
        train_sum = np.sum(np.square(self.X_train), axis=1)
        inner_product = np.dot(X, self.X_train.T)
        dists = np.sqrt(-2*inner_product + test_sum.reshape(-1, 1) + train_sum)
        return dists

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            closest_y = []
            y_indicies = np.argsort(dists[i, :], axis=0)
            closest_y = self.y_train[y_indicies[: k]]
            y_pred[i] = np.argmax(np.bincount(closest_y))
        return y_pred


