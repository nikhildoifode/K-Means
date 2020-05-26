# Assumption:
It is assumed that in the dataset, last column is for label.

# How to run (cityblock = Manhattan Distance; euclidean = Euclidean Distance):
$ python3 kMeans.py --dataset /path/to/data/filename.csv --dist <cityblock|euclidean>

# Example:
$ python3 kMeans.py --dataset Breast_cancer_data.csv --dist cityblock