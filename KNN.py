from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
irs = datasets.load_iris()
features = irs.data
labels =  irs.target
# /\print(features,labels)
cof = KNeighborsClassifier()
cof.fit(features,labels)
pred = cof.predict([[0,0,1,1]])
print(pred)