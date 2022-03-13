#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we test how the use of control variates can make the approximation of the
valuation of an american call and put option better.

@author: Andrea Mazzon
"""

from controlVariates import AmericanOptionWithControlVariates
import matplotlib.pyplot as plt
from analyticformulas.analyticFormulas import blackScholesPricePut

#parameters for the model
initialValue = 1
r = 0.02
sigma = 0.7

#paramteres for the option
maturity = 3
strike = initialValue


maximumNumberOfTimes = 150

americanCV = []
americanBinomial =[]
european = []

evaluator = AmericanOptionWithControlVariates(initialValue, r, sigma, maturity, strike)

for numberOfTimes in range (2, maximumNumberOfTimes + 1):
    
    #note that we want the put price: we then select the entries [1]
    americanCV.append(evaluator.getAmericanPutPriceWithControlVariates(numberOfTimes))
    americanBinomial.append(evaluator.getAmericanPutPriceWithBinomialModel(numberOfTimes))
    european.append(evaluator.getEuropeanPutPriceWithBinomialModel(numberOfTimes))
  
#here as well 
blackScholesPrice = blackScholesPricePut(initialValue, r, sigma, maturity, strike)
blackScholesVector = [blackScholesPrice] * (maximumNumberOfTimes - 1)

plt.plot(americanCV)
plt.plot(americanBinomial)
plt.plot(european)
plt.plot(blackScholesVector)
plt.ylim([americanCV[-1]*94/100, americanCV[-1]*104/100])
plt.legend(('american, control variates','american, binomial model',
            'european, binomial model','european, analytic'))
plt.title("Control variates approximation of an American option for a BS model")
plt.show()      
