import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'Python/Machine Learning/2. Regression/5. Decision Tree/Position_Salaries.csv')
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values

y = np.array(y)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

my_test = np.array([6.5])
my_test = my_test.reshape(1,len(my_test))

print(regressor.predict(my_test))

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)

plt.scatter(x,y,color = 'red')
plt.scatter(my_test,regressor.predict(my_test),color = 'green')
plt.plot(x_grid,regressor.predict(x_grid),color = 'blue')
plt.show()