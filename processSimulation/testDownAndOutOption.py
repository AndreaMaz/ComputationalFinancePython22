#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It tests the pricing of a Knock-out (down-and-out) option, by using the 
Monte-Carlo simulation of the underlying process

@author: Andrea Mazzon
"""

import time
import numpy as np

from processSimulation.eulerDiscretizationForBlackScholesWithLogarithm import EulerDiscretizationForBlackScholesWithLogarithm
from processSimulation.knockOutOption import KnockOutOption
from analyticformulas.analyticFormulas import blackScholesDownAndOut



numberOfSimulations = 10000
seed = 20

timeStep = 0.001
finalTime = 3
maturity = finalTime

initialValue = 2
r = 0.0
sigma = 0.5 


strike = 2
lowerBarrier = 1.9

analyticPrice = blackScholesDownAndOut(initialValue, r, sigma, maturity, strike, lowerBarrier)

print("The analytic price is ", analyticPrice)

#Monte-Carlo

payoffFunction = lambda x : np.maximum(x-strike,0)



timeMCInit = time.time() 

eulerBlackScholes= EulerDiscretizationForBlackScholesWithLogarithm(numberOfSimulations, timeStep, finalTime,
                   initialValue, r, sigma)

processRealizations = eulerBlackScholes.getRealizations()

knockOutOption = KnockOutOption(payoffFunction, maturity, lowerBarrier)

timeNeededMC = time.time()  - timeMCInit

print("The Monte-Carlo price is ", knockOutOption.getPrice(processRealizations))
print("The time needed with Monte-Carlo is ", timeNeededMC)

