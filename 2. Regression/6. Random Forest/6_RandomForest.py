import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\2. Regression\6. Random Forest\Position_Salaries.csv')
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

y = np.array(y)

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0) # n_estimators = number of decision trees
regressor.fit(x,y)

my_test = np.array([6.5])
my_test = my_test.reshape(len(my_test),1)

print(regressor.predict(my_test))

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color = 'red')
plt.scatter(my_test,regressor.predict(my_test),color = 'green')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.show()