import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans
from scipy.ndimage.measurements import label 
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


data = pd.read_csv("C:\\Users\\evil1\\Desktop\\Machine Learning\\5. clustering\\1.1 K-means clustering dataset.csv")
x = data.iloc[:,3:].values


# elbow method to find the optimal number of cluster
wcss = []
for i in range(1,11):
    cluster  = KMeans(n_clusters=i,init="k-means++",random_state=42)
    cluster.fit(x)
    wcss.append(cluster.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow Method")
plt.xlabel("Numbers Of Cluster")
plt.ylabel("WCSS")
plt.show()


cluster  = KMeans(n_clusters=5,init="k-means++",random_state=42)
y_predict =cluster.fit_predict(x)

plt.scatter(x[y_predict==0,0],x[y_predict==0,1],s=80,c="red",label="Cluster 1")
plt.scatter(x[y_predict==1,0],x[y_predict==1,1],s=80,c="blue",label="Cluster 2")
plt.scatter(x[y_predict==2,0],x[y_predict==2,1],s=80,c="green",label="Cluster 3")
plt.scatter(x[y_predict==3,0],x[y_predict==3,1],s=80,c="cyan",label="Cluster 4")
plt.scatter(x[y_predict==4,0],x[y_predict==4,1],s=80,c="magenta",label="Cluster 5")
plt.scatter(cluster.cluster_centers_[:,0],cluster.cluster_centers_[:,1],s=60,c="yellow",label="Centroids")
plt.title("Cluster Of Customer")
plt.xlabel("Annual  Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()