from sklearn import svm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

data = pd.read_csv('./data.csv')

print(data.info())
print(data.describe())
print(data.describe(include=['O']))
print(data.head())
print(data.tail())

data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})

features_mean= list(data.columns[2:12])
features_se= list(data.columns[12:22])
features_worst=list(data.columns[22:32])

data.drop('id',axis=1,inplace=True)

corr = data[features_mean].corr()
plt.figure(figsize=(14,14))
sns.heatmap(corr, annot=True)
plt.show()

features_remain = ['radius_mean','texture_mean', 'smoothness_mean','compactness_mean','symmetry_mean', 'fractal_dimension_mean']

train , test = train_test_split(data, test_size = 0.3)
train_X = train[features_remain]
train_y = train['diagnosis']
test_X = test[features_remain]
test_y = test['diagnosis']
train_X2 = train.iloc[:,1:]
test_X2 = test.iloc[:,1:]

ss = StandardScaler()
train_X = ss.fit_transform(train_X)
test_X = ss.transform(test_X)
train_X2 = ss.fit_transform(train_X2)
test_X2 = ss.transform(test_X2)

model = svm.SVC()
model.fit(train_X, train_y)
prediction = model.predict(test_X)
print(metrics.accuracy_score(test_y,prediction))

model2 = svm.LinearSVC()
model2.fit(train_X,train_y)
prediction2 = model2.predict(test_X)
print(metrics.accuracy_score(test_y,prediction2))

model3 = svm.SVC()
model3.fit(train_X2,train_y)
prediction3 = model3.predict(test_X2)
print(metrics.accuracy_score(test_y,prediction3))

model4 = svm.LinearSVC()
model4.fit(train_X2,train_y)
prediction4 = model3.predict(test_X2)
print(metrics.accuracy_score(test_y,prediction4))