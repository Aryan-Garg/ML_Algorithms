'''
Aryan Garg
B19153
Lab 7
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from scipy.optimize import linear_sum_assignment

print('''
Lab Assignment 7
Clustering Algorithms

''')
# Read training and testing data
train_data = pd.read_csv("mnist-tsne-train.csv")
test_data = pd.read_csv("mnist-tsne-test.csv")

# Separate input(params) and output(label) 
X = np.array(train_data[['dimention 1', 'dimension 2']])
y = np.array(train_data['labels'])
test_x = np.array(test_data[['dimention 1', 'dimention 2']])
test_y = np.array(test_data['labels'])

# Helper functions
def purity(y, y_pred):
    # Computes purity score for K-Means clusters
    conm = metrics.cluster.contingency_matrix(y, y_pred)
    row_i, col_i = linear_sum_assignment(-conm)
    purity_score = (conm[row_i, col_i].sum())/np.sum(conm)
    return purity_score

def plotter(title,X,y):
    # Makes scatter-plot of clusters
    plt.title(title)
    plt.grid(True)
    plt.scatter(X[:,0], X[:,1], c = y, cmap='tab20')
    plt.show()
    

print("Q1 K-Means")
# 1 a) K-Means model training...
kmeans = KMeans(n_clusters = 10).fit(X)
y_kmeans = kmeans.predict(X)

plt.title("Training data-points clusters(K-Means)")
plt.grid(True)
plt.scatter(X[:,0], X[:,1], c = y_kmeans, cmap='tab20')
centers = kmeans.cluster_centers_
plt.scatter(centers[:,0], centers[:,1], c ='black')
plt.show()

# 1 b) Purity-score
print("Purity score(training):",purity(y, y_kmeans))

# 1 c) Testing the model...
test_y_pred = kmeans.predict(test_x)

plt.title("Test data predictions(K-Means)")
plt.grid(True)
plt.scatter(test_x[:,0], test_x[:,1], c = test_y_pred, cmap='tab20')
centers_ = kmeans.cluster_centers_
plt.scatter(centers_[:,0], centers_[:,1], c = 'black')
plt.show()

# 1 d)
print("Purity score(testing):",purity(test_y, test_y_pred))

print()
print("Q2 GMM")
# 2 a) Training GMM model...
gmm = GaussianMixture(n_components=10).fit(X)
labels = gmm.predict(X)
plotter("GMM clustering on training data",X,labels)

# 2 b)
print("Purity-score(training):",purity(y, labels))

# 2 c)
test_labels = gmm.predict(test_x)
plotter("Test data predictions(GMM)",test_x,test_y_pred)

# 2 d)
print("Purity-score(testing):",purity(test_y, test_labels))

print()
print("Q3 DBSCAN")
# Q3 a) DBSCAN training and plotting...
dbscan_m = DBSCAN(eps = 5, min_samples=10)
db_labels = dbscan_m.fit_predict(X,y)

plotter("Training data clusters (DBSCAN)", X, db_labels)

# b)
print("Purity-score(training):",purity(y,db_labels))

# c)
dbscan_m.fit(test_x)
db_test_labels = dbscan_m.labels_
plotter("Testing data clusters (DBSCAN)", test_x, db_test_labels)

# d)
print("Purity-score(testing):",purity(test_y, db_test_labels))

