import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets ,linear_model
from sklearn.metrics import mean_squared_error

dib = datasets.load_diabetes()
# print(dib.keys())
dib_X = dib.data[:,np.newaxis ,2]
# dib_X = dib.data
# print(dib_X)
dib_X_Ts = dib_X[-30:]
dib_Y_Ts = dib.target[-30:]

dib_X_Tr = dib_X[:-30]
dib_Y_Tr = dib.target[:-30]
# print(dib_Y_Tr)



co = linear_model.LinearRegression()
co.fit(dib_X_Tr, dib_Y_Tr)
# co.predict(dib_X_Ts,dib_Y_Ts)
y_predicted = co.predict(dib_X_Ts)
ab = mean_squared_error(y_predicted,dib_Y_Ts)
print(ab)
print(co.coef_)
print(co.intercept_)
# plt.scatter(dib_X_Ts,dib_Y_Ts)
# plt.plot(dib_X_Ts,y_predicted )
# plt.show()