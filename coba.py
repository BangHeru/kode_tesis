import json
from pprint import pprint
from sklearn import datasets
import csv
import pandas
import numpy as np
from sklearn.utils import as_float_array, check_array
#ata_wine = datasets.load_wine()
#digits = datasets.load_digits()
iris = datasets.load_iris()


#data_wine = datasets.load_wine()
#data = json.load(open('contoh_json.json'))

#pprint(data)

#pprint(data_wine.target)

"""
data = [
        [3, 4, 3, 2], 
        [4, 3, 5, 1],
        [3, 5, 3, 3],
        [2, 1, 3, 3],
        [1, 1, 3, 2]
        ]
target = [1, 1, 1, 2, 2]
print(data)
print(data[1])
print(target)
"""

X = iris.data
labels_true = iris.target

#print X
yeast_data = []

"""
with open('yeast.data', 'r') as file:
        #reader = csv.DictReader(file)
        reader = csv.reader(file,  quotechar='|')
        for row in reader:
                yeast_data.append(row)
                #print ', '.join(row)

for i in range(0, 5):
        print(yeast_data[i])

print(len(yeast_data))

#print(yeast_data[:,4])


#data_yeast = np.array(yeast_data)
#print(data_yeast[1])
"""

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
names = ['Sequence Name','mcg', 'gvh', 'alm', 'mit', 'erl','pox','vac','nuc']
dataset = pandas.read_csv('yeast.data', names=names, delim_whitespace=True)


# shape
#print(dataset.shape)

# head
#print(dataset.head(5))

array = dataset.values
X = array[:,0:7]
labels_true = array[:,8]
Y = array[:,4]
indek = {'CYT':0, 'NUC':1, 'MIT':2, 'ME3':3, 'ME2':4, 'ME1':5, 'EXC':6, 'VAC':7, 'POX':8, 'ERL':9}
target = []
#print (X)

for kode in labels_true:
        target.append(indek[kode])
        #print indek[kode]
target = np.array(target)
#print(labels_true)
print(target)