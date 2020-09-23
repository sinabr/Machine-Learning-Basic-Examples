import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import preprocessing 
from sklearn.model_selection import cross_validate

df = pd.read_excel('titanic.xls')



df.drop(['body','name','ticket','boat','sex'],1,inplace=True)

# print(df.columns)

df.apply(pd.to_numeric , errors='ignore')
df.fillna(0,inplace=True)

# There are non-numeric data -> We'll transform it to numeric data here

def handle_non_numeric_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            # Unique non-repetative elements
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int , df[column]))

    return df

df = handle_non_numeric_data(df)

x = np.array(df.drop(['survived'],1).astype(float))

# Without Scaling -> Much lower accuracy
x = preprocessing.scale(x)

y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(x)

correct = 0 

for i in range(len(x)):
    predict_me = np.array(x[i].astype(float))
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct+=1


print(correct/len(x))