#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We test here the performances of three methods for the computation of the price
of a Cliquet option, in terms of the variance of the computed prices:
    - standard Monte-Carlo
    - Monte-Carlo with Antithetic variables
    - Monte-Carlo with Control variates.

We look at the variance since the analytic price is in general not known

@author: Andrea Mazzon
"""

import numpy as np
import time

from cliquetOption import CliquetOption
from cliquetOptionWithArrays import CliquetOptionWithArrays


from fasterControlVariatesCliquetBSWithArrays import FasterControlVariatesCliquetBSWithArrays
from controlVariatesCliquetBS import ControlVariatesCliquetBS

from fasterControlVariatesCliquetBS import FasterControlVariatesCliquetBS
from generateBSReturnsWithArrays import GenerateBSReturns

# processParameters
r = 0.2
sigma = 0.5

# option parameters
maturity = 4

numberOfTimeIntervals = 16

localFloor = -0.05
localCap = 0.3

globalFloor = 0
globalCap = numberOfTimeIntervals * 0.3

# Monte Carlo parameter

numberOfSimulations = 10000

# the object to generate the returns
generator = GenerateBSReturns(numberOfSimulations, numberOfTimeIntervals,
                              maturity, sigma, r)

# we want to compute the price with the standard Cliquet option implementation..
cliquetOption = CliquetOption(numberOfSimulations, maturity, localFloor, localCap, globalFloor, globalCap)


# ..with control variates..
cliquetWithControlVariates = \
    ControlVariatesCliquetBS(numberOfSimulations, maturity, numberOfTimeIntervals,
                             localFloor, localCap, globalFloor, globalCap, sigma, r)

# ..with the faster implementation of control variates..
fasterCliquetWithControlVariates = \
    FasterControlVariatesCliquetBS(numberOfSimulations, maturity, numberOfTimeIntervals,
                                   localFloor, localCap, globalFloor, globalCap, sigma, r)

# ..with the Cliquet option implementation that uses arrays..
cliquetWithArrays = \
    CliquetOptionWithArrays(numberOfSimulations, maturity, localFloor, localCap, globalFloor, globalCap)

# ..and with faster implementation of control variates, using arrays
fasterCliquetWithControlVariatesWithArrays = \
    FasterControlVariatesCliquetBSWithArrays(numberOfSimulations, maturity, numberOfTimeIntervals,
                                   localFloor, localCap, globalFloor, globalCap, sigma, r)

numberOfTests = 30

pricesStandard = []
pricesAV = []
pricesCV = []
pricesFasterCV = []
pricesStandardWithArrays = []
pricesFasterCVWithArrays = []

timesStandard = []
timesAV = []
timesCV = []
timesFasterCV = []
timesStandardWithArrays = []
timesFasterCVWithArrays = []

for k in range(numberOfTests):
    # first we do it via standard Monte-Carlo
    start = time.time()
    returnsRealizations = generator.generateReturns()
    priceStandardMC = cliquetOption.discountedPriceOfTheOption(returnsRealizations, r)
    end = time.time()
    pricesStandard.append(priceStandardMC)
    timesStandard.append(end - start)

    # then via Monte-Carlo with Antithetic variables
    start = time.time()
    returnsRealizationsAV = generator.generateReturnsAntitheticVariables()
    priceAV = cliquetOption.discountedPriceOfTheOption(returnsRealizationsAV, r)
    end = time.time()
    pricesAV.append(priceAV)
    timesAV.append(end - start)

    # the with control variates
    start = time.time()
    priceCV = cliquetWithControlVariates.getPriceViaControlVariates()
    end = time.time()
    pricesCV.append(priceCV)
    timesCV.append(end - start)

    # ..with the faster control variates
    start = time.time()
    priceFasterCV = fasterCliquetWithControlVariates.getPriceViaControlVariates()
    end = time.time()
    pricesFasterCV.append(priceFasterCV)
    timesFasterCV.append(end - start)

    # ..with control variates using arrays
    start = time.time()
    priceStandardWithArrays = cliquetWithArrays.discountedPriceOfTheOption(returnsRealizationsAV, r)
    end = time.time()
    pricesStandardWithArrays.append(priceStandardWithArrays)
    timesStandardWithArrays.append(end - start)

    # ..and finally with the faster control variates using arrays
    start = time.time()
    priceFasterCVWithArrays = fasterCliquetWithControlVariatesWithArrays.getPriceViaControlVariates()
    end = time.time()
    pricesFasterCVWithArrays.append(priceFasterCVWithArrays)
    timesFasterCVWithArrays.append(end - start)

print()
print("The variance of the prices using standard Monte-Carlo is ", np.var(pricesStandard))

print()
print("The variance of the prices using Antithetic variables is ", np.var(pricesAV))

print()
print("The variance of the prices using Control variates is ", np.var(pricesCV))

print()
print("The variance of the prices using the faster Control variates is ", np.var(pricesFasterCV))

print()
print("The variance of the prices using standard Monte-Carlo with arrays is ", np.var(pricesStandardWithArrays))

print()
print("The variance of the prices using the faster Control variates with arrays is ", np.var(pricesFasterCVWithArrays))

print()
print("The average elapsed time using standard Monte-Carlo is ", np.mean(timesStandard))

print()
print("The average elapsed time using Antithetic variables is ", np.mean(timesAV))

print()
print("The average elapsed time using Control variates is ", np.mean(timesCV))

print()
print("The average elapsed time using the faster Control variates is ", np.mean(timesFasterCV))

print()
print("The average elapsed time using standard Monte-Carlo with arrays is ", np.mean(timesStandardWithArrays))

print()
print("The average elapsed time using the faster Control variates with arrays is ", np.mean(timesFasterCVWithArrays))