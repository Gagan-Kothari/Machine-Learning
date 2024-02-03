import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\6. Clustering\2. Heirarchial\Mall_Customers.csv')
x = data.iloc[:,3:].values

# using dendrograms : scipy is required
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distances")
plt.show()

# Training the model wtih 5 categories
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean' , linkage='ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0, 0],x[y_hc == 0, 1], c = 'red' , s = 20 , label = 'Cat 1')
plt.scatter(x[y_hc == 1, 0],x[y_hc == 1, 1], c = 'blue' , s = 20 , label = 'Cat 2')
plt.scatter(x[y_hc == 2, 0],x[y_hc == 2, 1], c = 'green' , s = 20 , label = 'Cat 3')
plt.scatter(x[y_hc == 3, 0],x[y_hc == 3, 1], c = 'yellow' , s = 20 , label = 'Cat 4')
plt.scatter(x[y_hc == 4, 0],x[y_hc == 4, 1], c = 'black' , s = 20 , label = 'Cat 5')
plt.legend()
plt.xlabel("Annual Salary (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Heirarchial Clustering")
plt.show()

# Training model with 3 categories:

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=3, affinity='euclidean' , linkage='ward')
y_hc = hc.fit_predict(x)

plt.scatter(x[y_hc == 0, 0],x[y_hc == 0, 1], c = 'red' , s = 20 , label = 'Cat 1')
plt.scatter(x[y_hc == 1, 0],x[y_hc == 1, 1], c = 'blue' , s = 20 , label = 'Cat 2')
plt.scatter(x[y_hc == 2, 0],x[y_hc == 2, 1], c = 'green' , s = 20 , label = 'Cat 3')
plt.legend()
plt.xlabel("Annual Salary (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Heirarchial Clustering")
plt.show()