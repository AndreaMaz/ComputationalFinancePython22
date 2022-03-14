#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AndreaMazzon
"""

import numpy as np
import math
from random import seed


class StandardEulerDiscretizationBlackScholes:
    """
    This is a class whose mail goal is to discretize and simulate a
    geometric Brownian motion under the martingale measure. That is, we simulate
    dX_t = r X_t dt + sigma X_t dW_t, t >=0.
     Attributes
    ----------
    numberOfSimulations : int
        the number of simulated paths.
    timeStep : float
        the time step of the time discretization.
    finalTime : float
        the final time of the time discretization.
    initialValue : float
        the initial value of the process.
    sigma : float
        the volatility of the process
    interestRate : float
        the interest rate (it gives the drift under the martingale measure)
    mySeed : int, optional
        the seed to the generation of the standard normal realizations

    Methods
    ----------
    getRealizations():
        It returns all the realizations of the process
    getRealizationsAtGivenTimeIndex(timeIndex):
        It returns the realizations of the process at a given time index
    getRealizationsAtGivenTime(time):
        It returns the realizations of the process at a given time
    getAverageRealizationsAtGivenTimeIndex(timeIndex):
        It returns the average realizations of the process at a given time index
    getAverageRealizationsAtGivenTime(time):
        It returns the average realizations of the process at a given time

    """

    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValue,
                 interestRate, sigma, mySeed=None):

        """


         Parameters
        ----------
        numberOfSimulations : int
            the number of simulated paths.
        timeStep : float
            the time step of the time discretization.
        finalTime : float
            the final time of the time discretization.
        initialValue : float
            the initial value of the process.
        sigma : float
            the volatility of the process
        r : float, optional
            the interest rate (it gives the drift under the martingale measure)
        mySeed : int, optional
            the seed to the generation of the standard normal realizations

        Returns
        -------
        None.

        """
        self.numberOfSimulations = numberOfSimulations
        self.timeStep = timeStep
        self.finalTime = finalTime
        self.initialValue = initialValue
        self.mySeed = mySeed
        self.interestRate = interestRate
        self.sigma = sigma

        # we generate all the paths for all the simulations
        self.__generateRealizations()

    def __generateRealizations(self):
        numberOfTimes = math.ceil(self.finalTime / self.timeStep) + 1

        # times on the rows
        self.realizations = np.zeros((numberOfTimes, self.numberOfSimulations))
        self.realizations[0] = [self.initialValue] * self.numberOfSimulations

        seed(self.mySeed)

        standardNormalRealizations = np.random.standard_normal((numberOfTimes, self.numberOfSimulations))

        for timeIndex in range(1, numberOfTimes):
            pastRealizations = self.realizations[timeIndex- 1]
            self.realizations[timeIndex] = pastRealizations + self.interestRate * pastRealizations * self.timeStep \
                                                 + self.sigma * pastRealizations * math.sqrt(self.timeStep) * \
                                                 standardNormalRealizations[timeIndex]  # the Brownian motion

    def getRealizations(self):
        """
        It returns all the realizations of the process

        Returns
        -------
        array
            matrix containing the process realizations. The n-th row contains
            the realizations at time t_n

        """
        return self.realizations

    def getRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the realizations of the process at a given time index

        Parameters
        ----------
        timeIndex : int
             the time index, i.e., the row of the matrix of realizations.

        Returns
        -------
        array
            the vector of the realizations at given time index

        """

        return self.realizations[timeIndex]

    def getRealizationsAtGivenTime(self, time):
        """
        It returns the realizations of the process at a given time

        Parameters
        ----------
        time : float
             the time at which the realizations are returned

        Returns
        -------
        array
            the vector of the realizations at given time

        """

        indexForTime = round(time / self.timeStep)
        return self.realizations[indexForTime]

    def getAverageRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the average realizations of the process at a given time index

        Parameters
        ----------
        time : int
             the time index, i.e., the row of the matrix of realizations.

        Returns
        -------
        float
            the average of the realizations at given time index

        """

        return np.average(self.realizations(timeIndex))

    def getAverageRealizationsAtGivenTime(self, time):
        """
        It returns the average realizations of the process at a given time

        Parameters
        ----------
        time : int
             the time at which the realizations are returned

        Returns
        -------
        float
            the average of the realizations at given time

        """

        return np.average(self.getRealizationsAtGivenTime(time))
