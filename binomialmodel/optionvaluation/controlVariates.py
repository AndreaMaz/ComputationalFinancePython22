#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import math

from binomialmodel.optionvaluation.americanOption import AmericanOption
from binomialmodel.creationandcalibration.binomialModelSmart import BinomialModelSmart

from europeanOption import EuropeanOption

from analyticformulas.analyticFormulas import blackScholesPricePut

class AmericanOptionWithControlVariates:
    """
    This class is devoted to an application of control variates to the
    approximation of the price of an american put option under the
    Black-Scholes model, simulating a binomial model. 
    
    In particular, we rely on: 
        - the analytic price P_E of an european put under the Black-Scholes
         model
        - the Monte-Carlo price P_E^N of an european put under the Black-Scholes
          model, simulating a binomial model with N times
        - the Monte-Carlo price P_A^N of an american put under the Black-Scholes
          model, simulating a binomial model with N times
          
    Using the heuristic that P_A - P_A^N is approximately equal to P_E - P_E^N,
    we then approximate P_A as P_A^N + P_E - P_E^N
    
    Attributes
    ----------
    initialValue : float
        the initial value of the process
    r : float
        the risk free rate of the model, i.e., such that the risk free asset B
        has dynamics B(t)=exp(r t) 
    sigma: float
        the log-volatility of the Black-Scholes model
    maturity: float
        the maturity of the option
    strike: float
        the strike of the option

    Methods
    -------
    getAmericanCallAndPutPriceWithBinomialModel(numberOfTimes):
        It returns the approximated value of an european call/put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times
        
    getEuropeanCallAndPutPriceWithBinomialModel(numberOfTimes):
        It returns the approximated value of an european call/put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times
        
    getAmericanCallAndPutPriceWithControlVariates(self, numberOfTimes):
        It returns the approximated value of an american call/put option written
        on a Black-Scholes model, by using control variates
    """
    
    def __init__(self, initialValue, r, sigma, maturity, strike):
        """
         Attributes
         ----------
        initialValue : float
            the initial value of the process
        r : float
            the risk free rate of the model, i.e., such that the risk free asset B
            has dynamics B(t)=exp(r t) 
        sigma: float
            the log-volatility of the Black-Scholes model
        maturity: float
            the maturity of the option
        strike: float
            the strike of the option
        """
        
        self.initialValue = initialValue
        self.r = r
        self.sigma = sigma
        self.maturity = maturity
        self.strike = strike

    
    def getAmericanPutPriceWithBinomialModel(self, numberOfTimes):
        """
        It returns the approximated value of an american put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times

        Parameters
        ----------
        numberOfTimes : int
            the number of times where we simulate the binomial model.

        Returns
        -------
        priceAmericanPut : float
            the approximated price of the American put.

        """
        initialValue = self.initialValue
        r = self.r
        sigma = self.sigma
        maturity = self.maturity
        strike = self.strike
        
        increaseIfUp = math.exp(sigma * math.sqrt(maturity / numberOfTimes))
        decreaseIfDown = 1/increaseIfUp
 
        interestRate = math.exp(r * maturity / numberOfTimes) - 1
    
        binomialmodel = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
        payoffEvaluator = AmericanOption(binomialmodel)
        
        payoffPut = lambda x : max(strike - x,0)
        
        priceAmericanPut = payoffEvaluator.getValueOption(payoffPut, numberOfTimes - 1)
        
        return priceAmericanPut


    def getEuropeanPutPriceWithBinomialModel(self, numberOfTimes):
        """
        It returns the approximated value of an european put option written
        on a Black-Scholes model, by using a binomial model with numberOfTimes
        times

        Parameters
        ----------
        numberOfTimes : int
            the number of times where we simulate the binomial model.

        Returns
        -------
        priceEuropeanPut : float
            the approximated price of the European put.

        """
        
        initialValue = self.initialValue
        r = self.r
        sigma = self.sigma
        maturity = self.maturity
        strike = self.strike
        
        increaseIfUp = math.exp(sigma * math.sqrt(maturity / numberOfTimes))
        decreaseIfDown = 1/increaseIfUp
 
        interestRate = math.exp(r * maturity / numberOfTimes) - 1
    
        binomialmodel = BinomialModelSmart(initialValue, decreaseIfDown, increaseIfUp,
                                numberOfTimes, interestRate) 
    
        payoffEvaluator = EuropeanOption(binomialmodel)
        
        payoffPut = lambda x : max(strike - x,0)
        priceEuropeanPut = payoffEvaluator.evaluateDiscountedPayoff(payoffPut, numberOfTimes - 1)
        
        return priceEuropeanPut


    def getAmericanPutPriceWithControlVariates(self, numberOfTimes):
        """
        It returns the approximated value of an american put option written
        on a Black-Scholes model, by using control variates
        
        Parameters
        ----------
        numberOfTimes : int
            the number of times where we simulate the binomial model.

        Returns
        -------
        pricePut : float
            the approximated price of the put.

        """
        
        bSPutEuropean = blackScholesPricePut(self.initialValue, self.r, self.sigma, self.maturity, self.strike)
        
        binomialPutEuropean = self.getEuropeanPutPriceWithBinomialModel(numberOfTimes)
        
        binomialPutAmerican = self.getAmericanPutPriceWithBinomialModel(numberOfTimes)
        
        controlVariatePutAmerican = binomialPutAmerican + (bSPutEuropean - binomialPutEuropean)
    
        return controlVariatePutAmerican