import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\6. Clustering\1. K-Means\Mall_Customers.csv')
x = data.iloc[:,3:].values

from sklearn.cluster import KMeans
# we will run the program with different amounts of clusters, [ ELBOW METHOD ]
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state=42) # init = ''  is essential to escape random initialisation trap
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) # inertia gives the wcss value

plt.plot(range(1,11),wcss,color = 'red')
plt.show()

# from graph we can see the number of clusters is 5

kmeans = KMeans(n_clusters=5, init = 'k-means++',random_state=42)
y_kmeans = kmeans.fit_predict(x)                                            ################ fit_predict: trains and returns value of the dependant variable created

print(y_kmeans)

plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],color = 'red', s=20 , label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],color = 'blue', s=20 , label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],color = 'green', s=20 , label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1],color = 'black', s=20 , label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1],color = 'yellow', s=20 , label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color = 'orange', s = 100)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()                          ### labels the clusters
plt.show()