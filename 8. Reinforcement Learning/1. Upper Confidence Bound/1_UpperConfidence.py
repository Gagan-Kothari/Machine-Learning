# The Data set is only for simulation, because this is a real-time process

import numpy as np;
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv(r'C:\Users\Gagan\Programming\Python\Machine Learning\8. Reinforcement Learning\1. Upper Confidence Bound\Ads_CTR_Optimisation.csv')

# Implementing UCB

import math

N = 10000      # number of user     ###################### this algo can find the best ad in 1000 rounds
d = 10         # number of ads
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d

total_reward = 0

for n in range(0,N):
    ad = 0
    max_upper_bound = 0
    for i in range(0,d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])                       # Confidence Interval
            upper_bound = average_reward + delta_i

        else:
            upper_bound = 1e400                                                                       # 1e400 is a very large value and 
            
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i

    ads_selected.append(ad)
    numbers_of_selections[ad] += 1

    reward = data.values[n][ad]
    sums_of_rewards[ad] += reward
    total_reward += reward

## Plotting the Histogram
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times ad was selected")
plt.show()