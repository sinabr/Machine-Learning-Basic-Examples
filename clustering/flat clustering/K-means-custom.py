import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans

x = np.array([[1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11],
    [1,3],[2,3],[4,2],[3,5],[8,10],[7,5],[3,1],[6,8]])


# x[:,0] -> all the 0th elements in x
# plt.scatter(x[:,0],x[:,1],s=150,linewidths=5)
# plt.show()

colors = 10 * ['g','r','o','b','k']

class K_Means:

    def __init__(self,k=2,tol=0.001,max_iter=300):
        self.max_iter = max_iter
        self.tol = tol
        self.k = k


    def fit(self,data):
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature_set in data:
                # Simply Euclidean Distances from Centroids
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                # class (cluster) Id of which the point is closest to
                classification = distances.index(min(distances))
                # add the point (sample) to the cluster
                self.classifications[classification].append(feature_set)

            prev_centroids = dict(self.centroids)


            for classification in self.classifications:
                # Mean of the points in the cluster
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            optimized = True

            for c in self.centroids:

                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]

                # print(np.sum((current_centroid - original_centroid)/original_centroid*100.0))
                if np.sum((current_centroid - original_centroid)/original_centroid*100.0) > self.tol:
                    optimized = False
            
            if optimized: 
                break

        


    def predict(self,data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

clf = K_Means()
clf.fit(x)

for centroid in clf.centroids:
    plt.scatter(clf.centroids[centroid][0],clf.centroids[centroid][1],
        marker='o' , color='k',s=150,linewidths=5)

for classification in clf.classifications:
    color = colors[classification]
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0],featureset[1],marker='x',color=color,s=50,linewidths=5)

unknowns = np.array([[1,3],[2,3],[4,2],[3,5],[8,10],[7,5],[3,1],[6,8]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0],unknown[1],marker='*',color=colors[classification],s=20,linewidths=5)


plt.show()