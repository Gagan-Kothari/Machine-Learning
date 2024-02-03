import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\2. Regression\4. Support Vector\Position_Salaries.csv')
x = data.iloc[:,1:-1].values
y = data.iloc[:,-1]
y = np.array(y).reshape(len(y),1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x = sc_x.fit_transform(x)
sc_y = StandardScaler()
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

my_test = np.array([6.5])
my_test = my_test.reshape(len(my_test),1)

my_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(my_test)).reshape(-1,1))
print(my_pred)

x_original = np.array(sc_x.inverse_transform(x))
y_original = np.array(sc_y.inverse_transform(y))

plt.scatter(x_original,y_original,color = 'red')
plt.scatter(my_test,np.array(sc_y.inverse_transform(regressor.predict(sc_x.transform(my_test)).reshape(-1,1))),color = 'green')
plt.plot(sc_x.inverse_transform(x), sc_y.inverse_transform(regressor.predict(x).reshape(-1,1)), color = 'blue')
plt.show()