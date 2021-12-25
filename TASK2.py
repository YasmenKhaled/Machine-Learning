# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 18:54:12 2021

@author: YASMEN KHALED
"""

import pandas as pd
import matplotlib.pyplot as plt
dataset=pd.read_csv("Wuzzuf_Jobs.csv")
#factorize YearExp feature
dataset['YearsExp']=pd.factorize(dataset['YearsExp'])[0]

#apply K-means job title & companies
x=dataset.iloc[:, [0,1]].values
df=pd.DataFrame(x)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])
x[:, 1] = labelencoder_x.fit_transform(x[:, 1])


from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()



#from kneed import KneeLocator
#k1 = KneeLocator(range(1, 20), wcss, curve="convex", direction="decreasing")
#print(k1.elbow)


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(x)
y_kmeans == 0
# Visualising the clusters
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 10, c = 'green', label = 'Cluster 3')


plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clusters of Jobs')
plt.xlabel('Title')
plt.ylabel('Company')
plt.legend()
plt.show()