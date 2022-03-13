#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we test the computation of the price of a knock-out option written
on a binomial model

@author: Andrea Mazzon
"""

#from europeanOption import EuropeanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart
from knockOutOption import KnockOutOption


initialValue = 100
decreaseIfDown = 0.9
increaseIfUp = 1.1
interestRate = 0.0
numberOfTimes = 7

myBinomialModelSmart = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 

myPayoffEvaluator = KnockOutOption(myBinomialModelSmart)

maturity = numberOfTimes - 1

payoff = lambda x : max(x-90,0)

lowerBarrier = 75
upperBarrier = 130


valuesOption = myPayoffEvaluator.getValuesPortfolioBackward(payoff, maturity, lowerBarrier, upperBarrier)
processRealizations = myBinomialModelSmart.getRealizations()

priceOfTheOption = myPayoffEvaluator.getInitialDiscountedValuePortfolio(payoff, maturity, lowerBarrier, upperBarrier)

print("The discounted price of the option computed going backward is ",
      priceOfTheOption)