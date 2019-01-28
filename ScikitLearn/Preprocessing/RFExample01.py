import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# https://blog.csdn.net/manjhOK/article/details/80367624
# #############################################################################
# Generate sample data
X = np.sort(7 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() * 20
# #############################################################################
# Add noise to targets
noise = 9 * (0.5 - np.random.rand(40))
y_noise = y + noise
# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=10)
svr_lin = SVR(kernel='linear', C=10)
rf = RandomForestRegressor()
y_SVM_rbf = svr_rbf.fit(X, y_noise).predict(X)
y_SVM_lin = svr_lin.fit(X, y_noise).predict(X)
y_RF_rf = rf.fit(X, y_noise).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y_noise, color='b', label='Input data')
plt.plot(X, y_SVM_rbf, color='y', lw=lw, label='SVM_RBF C=10 ')
plt.plot(X, y_SVM_lin, color='c', lw=lw, label='SVM_Linear C=10')
plt.plot(X, y_RF_rf, color='r', lw=lw, label='RandomFrest_RF')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
