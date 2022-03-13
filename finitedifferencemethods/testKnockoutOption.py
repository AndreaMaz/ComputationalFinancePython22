#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
It tests the pricing of a Knock-out (down-and-out) option, both with 
Monte-Carlo simulation of the underlying process and PDEs, in the case of a
Black-Scholes model.

@author: Andrea Mazzon
"""

import math
import time
import numpy as np
from numpy import mean

from processSimulation.eulerDiscretizationForBlackScholesWithLogarithm import EulerDiscretizationForBlackScholesWithLogarithm
from processSimulation.knockOutOption import KnockOutOption
from analyticformulas.analyticFormulas import blackScholesDownAndOut
from implicitEuler import ImplicitEuler



maturity = 3

initialValue = 2
r = 0.0
sigma = 0.5 


strike = 2
lowerBarrier = 1

analyticPrice = blackScholesDownAndOut(initialValue, r, sigma, maturity, strike, lowerBarrier)

print("The analytic price is: {:.5}".format(analyticPrice))

#Monte-Carlo

numberOfSimulations = 10000
seed = 1897

timeStep = 0.01
finalTime = maturity

payoffFunction = lambda x : np.maximum(x-strike,0)

eulerBlackScholes= EulerDiscretizationForBlackScholesWithLogarithm(numberOfSimulations, timeStep, finalTime,
                       initialValue, r, sigma)

knockOutOption = KnockOutOption(payoffFunction, maturity, lowerBarrier)


#Implicit Euler
dx = 0.05

xmin = lowerBarrier
xmax = 13

dt = dx
tmax = 3

sigmaFunction = lambda x : sigma

functionLeft = lambda x, t : 0
functionRight = lambda x, t : x - strike * math.exp(-r * t)

implicitEulerSolver = ImplicitEuler(dx, dt, xmin, xmax, tmax, r, sigmaFunction, payoffFunction, functionLeft,
                                    functionRight)

errorsMonteCarlo = []
timesMonteCarlo = []

errorsImplicitEuler = []
timesImplicitEuler = []

numberOfTests = 50


for testIndex in range(numberOfTests):

    timeMCInit = time.time()

    processRealizations = eulerBlackScholes.getRealizations()

    MCprice = knockOutOption.getPrice(processRealizations)

    timesMonteCarlo.append(time.time()  - timeMCInit)
    errorsMonteCarlo.append(abs(MCprice-analyticPrice)/analyticPrice)

#Implicit Euler


    timeImplicitInit = time.time()

    IEprice = implicitEulerSolver.getSolutionForGivenTimeAndValue(tmax, initialValue)

    timesImplicitEuler.append(time.time()  - timeImplicitInit)
    errorsImplicitEuler.append(abs(IEprice-analyticPrice)/(analyticPrice))

print()
print("Average time needed with Monte-Carlo: {:.5}".format(mean(timesMonteCarlo)))
print("Average time needed with Implict Euler: {:.5}".format(mean(timesImplicitEuler)))

print()

print("Average error with Monte-Carlo: {:.5}".format(mean(errorsMonteCarlo)))
print("Average error with Implicit Euler: {:.5}".format(mean(errorsImplicitEuler)))