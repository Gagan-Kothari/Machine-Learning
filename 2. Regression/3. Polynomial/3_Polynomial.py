# During salary Negotiation, the interviewee asks for 160000 as he was offered that for the job,
# using the given data set, predict if he was lying considering he had the position for quite a while (assume the value to be between 6 and 7)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'Python/Machine Learning/2. Regression/3. Polynomial/Position_Salaries.csv')
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1]


# Linear Regression
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(x,y)

my_test = np.array([6.5])
my_test = my_test.reshape(1,len(my_test))

lin_my_pred = lin_regressor.predict(my_test)

# Polynomial Regression

from sklearn.preprocessing import PolynomialFeatures
poly_regressor = PolynomialFeatures(degree = 4) # Higher degrees make the graph even more fit but very high degrees ruin fitting
x_poly = poly_regressor.fit_transform(x)

poly_lin_reg = LinearRegression()
poly_lin_reg.fit(x_poly,y)

poly_my_test = np.array([6.5])
poly_my_test = my_test.reshape(1,len(my_test))
poly_my_pred = poly_regressor.fit_transform(my_test)

# Printing the predicted values using linear and polynomial regression

print('Linear Regression: ',lin_my_pred)
print("Polynomial Linear Regression: ",poly_lin_reg.predict(poly_my_pred))

plt.subplot(1,2,1)
plt.scatter(x,y,color = 'red')
plt.scatter(my_test,lin_my_pred,color = 'green')
plt.plot(x,lin_regressor.predict(x),color = 'blue')

plt.subplot(1,2,2)
plt.scatter(x,y,color = 'red')
plt.scatter(my_test,poly_lin_reg.predict(poly_my_pred),color= 'green')
plt.plot(x,poly_lin_reg.predict(x_poly),color = 'blue')

plt.show()