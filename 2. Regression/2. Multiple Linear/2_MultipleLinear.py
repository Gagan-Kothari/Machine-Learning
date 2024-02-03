import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\2. Regression\2. Multiple Linear\50_Startups.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[3])],remainder = 'passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print("\n")

############################################################# MY DATA TO PREDICT
my_test = np.array([1.0,0.0,0.0,160000,130000,300000])
my_test = my_test.reshape(1,len(my_test))

my_pred = regressor.predict(my_test)
print(my_pred)

print("\n")

############################################################ To get equation of linear regression
print(regressor.coef_)
print(regressor.intercept_)