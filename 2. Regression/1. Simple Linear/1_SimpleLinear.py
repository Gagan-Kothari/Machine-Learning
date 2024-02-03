import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\2. Regression\1. Simple Linear\Salary_Data.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train, y_train)

# y_pred = regression.predict(x_test)

plt.subplot(1, 2, 1)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.title("Train Set")
plt.xlabel("Years")
plt.ylabel("Salary")

plt.subplot(1, 2, 2)
plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regression.predict(x_train), color='blue')
plt.title("Test set")
plt.xlabel("Years")
plt.ylabel("Salary")

plt.show()