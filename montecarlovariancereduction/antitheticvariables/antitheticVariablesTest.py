#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test the comparison of the average errors we get with standard Monte-Carlo
and Antithetic Variables when valuating an European call option. 

In particular, we plot the two average errors for an incresing number of
simulations.

@author: Andrea Mazzon
"""

from compareStandardMCWithAV import compare

import matplotlib.pyplot as plt
import time


#process parameters
initialValue = 100
r = 0.05
sigma = 0.5

#option parameters
T = 3
strike = initialValue


#we want to test the two methods for different numbers of simulations
numbersOfSimulations = [10**k for k in range(3,6)] #[1000, 10000, 100000]

averageErrorsWithStandardMC = []
averageErrorsWithAV = []

start = time.time()
for numberOfSimulations in numbersOfSimulations:
    #the function compare returns a 2-uple. The first value is the average error of the
    #standard Monte-Carlo method, the second one the one with Antithetic variables
    averageErrorWithStandardMC, averageErrorWithAV = \
        compare(numberOfSimulations,initialValue, sigma, T, strike, r)
    averageErrorsWithStandardMC.append(averageErrorWithStandardMC) 
    averageErrorsWithAV.append(averageErrorWithAV)

end = time.time()

plt.plot(numbersOfSimulations,averageErrorsWithStandardMC, 'bo')
plt.plot(numbersOfSimulations,averageErrorsWithAV, 'ro')
plt.xlabel('Number of simulations')
plt.ylabel('Average error')
plt.title('Average errors for a call option, with standard M-C and Antithetic Variables')
plt.legend(['Standard Monte-Carlo', 'Antithetic Variables'])
plt.show()

print(end-start)
