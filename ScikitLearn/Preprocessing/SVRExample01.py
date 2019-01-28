import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

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
svr_rbf = SVR(kernel='rbf', C=1)
svr_lin = SVR(kernel='linear', C=1)
svr_poly = SVR(kernel='poly', C=1, degree=2)
y_rbf = svr_rbf.fit(X, y_noise).predict(X)
y_lin = svr_lin.fit(X, y_noise).predict(X)
y_poly = svr_poly.fit(X, y_noise).predict(X)

# #############################################################################
# Look at the results
lw = 2
plt.scatter(X, y_noise, color='darkorange', label='Input data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF C=1 ')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear C=1')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial C=1')
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()