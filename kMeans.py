import numpy as np
import csv
import sys

from copy import deepcopy
from scipy.spatial.distance import pdist, euclidean, cityblock

class KNearestNeighbors:
    def __init__ (self, k, n, features):
        self.k = int(k)
        self.n = n
        self.features = features


    def fit(self, X, metric):
        self.centroids = []
        centers = np.random.randn(self.k, self.features) * (np.std(X, axis = 0)) + (np.mean(X, axis = 0))
        centers_old = np.zeros(centers.shape)

        clusters = np.zeros(self.n)
        distances = np.zeros((self.n, self.k))
        error = 1

        while error != 0:
            for i in range(self.k):
                for j in range(X.shape[0]):
                    if metric == 'euclidean':
                        distances[j,i] = euclidean(X[j], centers[i])
                    else:
                        distances[j,i] = cityblock(X[j], centers[i])

            centers_old = deepcopy(centers)
            clusters = np.argmin(distances, axis = 1)
            for i in range(self.k):
                centers[i] = np.mean(X[clusters == i], axis=0)

            error = pdist(centers - centers_old, metric=metric)

        self.centroids = centers


    def predict(self, X, metric):
        if metric == 'euclidean':
            distances = [euclidean(X, centroid) for centroid in self.centroids]
        else:
            distances = [cityblock(X, centroid) for centroid in self.centroids]
        return distances.index(min(distances))


    def calculate_accuracy(self, label, prediction):
        return np.mean(label == prediction)


def main ():
    if len(sys.argv) != 5 or '--dataset' not in sys.argv or '--dist' not in sys.argv:
        print("Invalid command")
        print("Make sure command is of type: " +
        "python3 kMeans.py --dataset /path/to/data/filename.csv --dist <cityblock|euclidean>")
        return

    filePath = sys.argv[sys.argv.index('--dataset') + 1]
    metric = sys.argv[sys.argv.index('--dist') + 1]

    if metric != 'cityblock' and metric != 'euclidean':
        print("Invalid command, Make sure dist parameter is either cityblock or euclidean")
        return

    try:
        with open(filePath)	as csvfile:
            reader = csv.reader(csvfile)
            data_list = list(reader)
            data_list.pop(0)

            X = np.array(data_list, dtype = np.float64)
            y = X[:, X.shape[1] - 1]
            y = y.astype(int)
            X = X[:, :-1]

            kMeans = KNearestNeighbors(2, X.shape[0], X.shape[1])
            kMeans.fit(X, metric)

            y_pred = []
            for row in X:
                y_pred.append(kMeans.predict(row, metric))

            misMatchIn1 = 0
            sameIn1 = 0
            count1 = np.count_nonzero(y_pred)
            misMatchIn0 = 0
            sameIn0 = 0
            count0 = len(y_pred) - count1

            for i in range(len(y_pred)):
                if (y_pred[i] == 1):
                    if (y[i] == y_pred[i]): sameIn1 += 1
                    else : misMatchIn1 += 1
                else:
                    if (y[i] == y_pred[i]): sameIn0 += 1
                    else : misMatchIn0 += 1

            print("(%) 1's in first cluster", sameIn1 * 100 / count1)
            print("(%) 1's in second cluster", 100 - (sameIn1 * 100 / count1))

            errorIn1 = min(misMatchIn1, sameIn1) / count1
            errorIn0 = min(misMatchIn0, sameIn0) / count0

            print('error', errorIn1+errorIn0)

    except IOError as e:
        print("Couldn't open the file (%s). Check file path, value of dist again." % e)


if __name__ == "__main__":
    main()