from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

dataset = {'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]}

new_features = [5,7]


def knn(data,predict,k=3):
    if len(data) >= k:
        warnings.warn('K is set to value less than total voting groups, idiot!')
        
    distances = []
    for group in data:
        for features in data[group]:
            # Euclidean Distance With Numpy
            eucdist = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([eucdist,group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]


    return vote_result


result = knn(dataset,new_features,3)
print(result)

[[plt.scatter(ii[0],ii[1],s=100,color=i) for ii in dataset[i]] for i in dataset]

plt.scatter(new_features[0],new_features[1],color='g',s=100)

plt.show()