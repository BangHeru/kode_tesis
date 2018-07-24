print(__doc__)

from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AdapAffinityPropagation
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn import datasets
import pandas
import numpy as np
# #############################################################################
# Generate sample data

""" names = ['Sequence Name','mcg', 'gvh', 'alm', 'mit', 'erl','pox','vac','nuc']
dataset = pandas.read_csv('yeast.data', names=names, delim_whitespace=True)
array = dataset.values
X = array[:,0:7]
name = array[:,8]
indek = {'CYT':0, 'NUC':1, 'MIT':2, 'ME3':3, 'ME2':4, 'ME1':5, 'EXC':6, 'VAC':7, 'POX':8, 'ERL':9}
target = []
for kode in name:
        target.append(indek[kode])

labels_true = np.array(target)
 """

"""
centers = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
X, labels_true = make_blobs(n_samples=15, centers=centers, cluster_std=0.5,
                            random_state=0)
"""
#print labels_true
"""
X = [
        [5.1, 3.5, 1.4, 0.2], 
        [4.9, 3, 1.4, 0.2],
        [7, 3.2, 4.7, 1.4],
        [6.3, 3.3, 6, 2.5]
        ]
labels_true = [1, 1, 2, 3]
"""
"""
X = [
      [1.4, 2.1],
      [-0.06, -1.4],
      [1.4, -1.0],
      [1.8, 1.2]
]
"""
labels_true = [1, 2, 2, 1]

data_wine = datasets.load_wine()
digits = datasets.load_digits()
iris = datasets.load_iris()

#X = digits.data
#labels_true = digits.target

X = iris.data
labels_true = iris.target


#X = data_wine.data
#labels_true = data_wine.target

#print(X)
# #############################################################################
# Compute Affinity Propagation
#af = AffinityPropagation(preference=-50).fit(X)
#af = AffinityPropagation().fit(X)
af = AdapAffinityPropagation().fit(X)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

n_clusters_ = len(cluster_centers_indices)

print('Estimated number of clusters: %d' % n_clusters_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
#score between 0.0 and 1.0. 1.0 stands for perfectly homogeneous labeling

print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
#score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
#score between 0.0 and 1.0. 1.0 stands for perfectly complete labeling

print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
#Similarity score between -1.0 and 1.0. Random labelings have an ARI close to 0.0. 1.0 stands for perfect match.

print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
#The AMI returns a value of 1 when the two partitions are identical(ie perfectly matched).

print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels, metric='sqeuclidean'))
#The best value is 1 and the worst value is -1.

#print("exemplars : ", cluster_centers_indices)
print("iteration : ", af.n_iter_)
# #############################################################################

# Plot result
#data = pandas.DataFrame(af.SM)
#data.to_csv('similarity.csv')
#print data


import matplotlib.pyplot as plt
from itertools import cycle

plt.close('all')
plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

