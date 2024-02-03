import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# header = None : makes the data not consider the first line as header
#
dataset = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\7. Association Rule Learning\1. Apriori\Market_Basket_Optimisation.csv', header=None)

transactions = []

for i in range(7501):
    transactions.append([str(dataset.values[i][j]) for j in range(0,20)])

# Training the Apriori Model

from apyori import apriori
rules = apriori(transactions= transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
# min_length and max_length : product A with product B (Only Two elements) can be changed according to need
# min_lift = 3 is a general rule of thumb
# for min_confidence start with 0.8, if there not many great rules divide by two, if again not many good rules keep dividing by 2

results = list(rules)

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])

## SORTING ACCORDING TO LIFT

resultsinDataFrame = resultsinDataFrame.nlargest(n = 10, columns = 'Lift')

print(resultsinDataFrame)