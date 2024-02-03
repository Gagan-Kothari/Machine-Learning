import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\4. Classification\6.Decision Tree Classification\Social_Network_Ads.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0) # CRITERION IS TO BE CHANGED AS LEARNED IN THEORY
classifier.fit(x_train,y_train)

# My Test
print(classifier.predict(sc.transform(np.array([[30,87000]]))))

y_pred = classifier.predict(x_test).reshape(-1,1)

print(np.concatenate((y_pred,y_test.reshape(-1,1)),1))

from sklearn.metrics import accuracy_score,confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(accuracy_score(y_test,y_pred))