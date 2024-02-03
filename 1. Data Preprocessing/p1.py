import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd

# Importing the data

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\2. Regression\1. Simple Linear\Salary_Data.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1]

# Taking care of missing data

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])

# Encoding Categorical Data

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting into Test and Train set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 1)

# Feature Scaling by Standardisation

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:,3:] = sc.fit_transform(x_train[:, 3:])
x_test[:,3:] = sc.transform(x_test[:,3:])