from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression

import  matplotlib.pyplot as plt
iris = datasets.load_iris()
# print(iris.keys())
# print(iris.data[30:],iris.target[30:])
# print(iris['data'].shape)
x = iris['data'][:,3:]
# print(x)
y = (iris['target']==2).astype(np.int)
# print(y)
# Train a logistic regression classifier
clf = LogisticRegression()
clf.fit(x,y)
# print(clf.predict(([[1.6,0,4,3]])))


x_new = np.linspace(0,3,1000).reshape(-1,1)
# print(x_new)
y_prob = clf.predict_proba(x_new)
plt.plot(x_new,y_prob[:,1:],"g-",label="viginica")
plt.show()