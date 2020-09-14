import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random


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


df = pd.read_csv('breast-cancer-wisconsin.data',names=['id','m1','m2','m3','m4','m5','m6','m7','m8','m9','class'])
df.replace('?',-99999 ,inplace=True)
df.drop(['id'],axis=1,inplace=True)

#  Convert the Data to Floats -> Get Values Out of the DF and convert to python lists
data = df.astype(float).values.tolist()

#print(data[:5])
random.shuffle(data)
#print(data[:5])
# So : Shuffle is 'Inplace'

test_size = 0.2
train_set = {2:[],4:[]}
test_set = {2:[],4:[]}
train_data = data[:-int(test_size*len(data))]
test_data = data[-int(test_size*len(data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])

for i in test_data:
    test_set[i[-1]].append(i[:-1])


correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote = knn(train_set,data,k=5)
        if group == vote:
            correct += 1

        total += 1

print('Accuracy :'  )
print(correct/total)