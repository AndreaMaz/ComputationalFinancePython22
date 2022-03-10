#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

import abc
import matplotlib.pyplot as plt


class BinomialModel(metaclass=abc.ABCMeta):
    """
    This is an abstract class whose main goal is to get the main features of a binomial model,
    like the evolution of its average or of its realizations.
    It can be extended by classes providing the way to construct it

    ...

    Attributes
    ----------
    initialValue : float
        the initial value S(0) of the process
    decreaseIfDown : float
        the number d such that S(j+1)=dS(j) with probability 1 - q
    increaseIfUp : float
        the number u such that S(j+1)=uS(j) with probability q
    numberOfTimes : int
        the number of times at which the process is simulated, initial time
        included
    interest rate : double
        the interest rate rho such that the risk free asset B follows the dynamics
        B(j+1) = (1+rho)B(j)
    riskNeutralProbabilityUp : double
        the risk neutral probability q =(1+rho-d)/(u-d) such that
        P(S(j+1)=S(j)*u) = q, P(S(j+1)=S(j)*d) = 1 - q,
        u > rho+1, d<1
    realizations, [double, double]
        a matrix containing the realizations of the process



    Methods
    -------
    generateRealizations()
        It generates the realizations of the process
    getRealizations()
        It returns the realizations of the process.
    getDiscountedAverageAtGivenTime(timeIndex)
        It returns the average of the process at time timeIndex discouted at time 0
    getEvolutionDiscountedAverage()
        It returns the evolution of the average of the process discounted at time 0
    printEvolutionDiscountedAverage()
        It prints the evolution of the average of the process discounted at time 0
    plotEvolutionDiscountedAverage()
        It plots the evolution of the average of the process discounted at time 0
    getPercentageOfGainAtGivenTime(timeIndex)
        It returns the percentage that (1+rho)^(-j)S(j)>S(0)
    getEvolutionPercentageOfGain()
        It returns the evolution of the percentage that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1
    printEvolutionPercentagesOfGain()
        It prints the evolution of the percentage that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1
    plotEvolutionPercentagesOfGain()
        It plots the evolution of the percentage that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1
    """

    # Python specific syntax for the constructor
    def __init__(self, initialValue, decreaseIfDown, increaseIfUp,
                 numberOfTimes,
                 interestRate=0,  # it is =0 if not specified
                 ):
        """
        Parameters
        ----------
        initialValue : float
            the initial value S(0) of the process
        decreaseIfDown : float
            the number d such that S(j+1)=dS(j) with probability 1 - q
        increaseIfUp : float
            the number u such that S(j+1)=uS(j) with probability q
        numberOfTimes : int
            the number of times at which the process is simulated, initial time
            included
        interest rate : float
            the interest rate rho such that the risk free asset B follows the dynamics
            B(j+1) = (1+rho)B(j)

         """
        self.initialValue = initialValue
        self.decreaseIfDown = decreaseIfDown
        self.increaseIfUp = increaseIfUp
        self.interestRate = interestRate
        self.numberOfTimes = numberOfTimes
        self.riskNeutralProbabilityUp = (1 + interestRate - decreaseIfDown) / (increaseIfUp - decreaseIfDown)
        # we generate the realizations once for all, during the call to the constructor
        self.realizations = self.generateRealizations()

    # note the syntax: this is an abstract method, whose implementation will be
    # given in the derived classes. In our case, we will see the implementation
    # with a pure Monte Carlo method and a smarter one.
    @abc.abstractmethod
    def generateRealizations(self):
        """It generates the realizations of the process.
        """

    def getRealizations(self):
        """
        It returns the realizations of the process.

        Returns
        -------
        list
            The matrix hosting the realizations of the process.
        """
        return self.realizations

        # this method is abstract as well: we will see indeed two different ways to

    # compute the average of the process, depending on the way the process is
    # generated

    @abc.abstractmethod
    def getDiscountedAverageAtGivenTime(self, timeIndex):
        """It computes the average of the realizations of the process at
        timeIndex, discounted at time 0

        Returns
        -------
        None.
        """

    def getEvolutionDiscountedAverage(self):
        """
        Returns
        -------
        list
            A vector representing the evolution of the average of the process
            discounted at time 0.
        """

        return [self.getDiscountedAverageAtGivenTime(timeIndex)
                for timeIndex in range(self.numberOfTimes)]

    def printEvolutionDiscountedAverage(self):
        """
        It prints the evolution of the average of the process discounted at time 0
        """
        evolutionDiscountedAverage = self.getEvolutionDiscountedAverage();

        print("The evolution of the average value of the discounted process is the following:")
        print()
        # note the syntax to tell the program we want to print three decimal digits,
        # separating the strings with a comma
        print('\n'.join('{:.3}'.format(discountedAverage) for discountedAverage
                        in evolutionDiscountedAverage))
        print()

    def plotEvolutionDiscountedAverage(self):
        """
        It plots the evolution of the average of the process discounted at time 0
        """
        evolutionDiscountedAverage = self.getEvolutionDiscountedAverage();

        plt.plot(evolutionDiscountedAverage)
        plt.xlabel('Time')
        plt.ylabel('Discounted average')
        plt.title('Evolution of the discounted average of the process')
        plt.show()

    # Also the way we compute this percentage depends on how we construct
    # Binomial model: for this reason, this is an abstract method.
    @abc.abstractmethod
    def getPercentageOfGainAtGivenTime(self, timeIndex):
        """
        It returns the percentage that (1+rho)^(-timeIndex)S(timeIndex)>S(0)

        Parameters
        ----------
        timeIndex : int
            The time at which we want to get the percentage
        Returns
        -------
        float
            The percentage that (1+rho)^(-timeIndex)S(timeIndex)>S(0)
        """

    def getEvolutionPercentageOfGain(self):
        """
        Returns
        -------
        list
            A list representing the evolution of percentage that (1+rho)^(-j)S(j)>S(0), for
            j going from 1 to self.numberOfTimes - 1 .

        """

        return [self.getPercentageOfGainAtGivenTime(timeIndex)
                for timeIndex in range(self.numberOfTimes)]

    def printEvolutionPercentagesOfGain(self):
        """
        It prints the evolution of the percentage that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1
        """
        percentagesPath = self.getEvolutionPercentageOfGain();

        print("The path of the percentage evolution  is the following:")
        print()
        print('\n'.join('{:.3}'.format(prob) for prob in percentagesPath))
        print()

    def plotEvolutionPercentagesOfGain(self):
        """
        "It plots the evolution of the percentage that (1+rho)^(-j)S(j)>S(0), for
        j going from 1 to self.numberOfTimes - 1
        """
        percentagesPath = self.getEvolutionPercentageOfGain();
        plt.plot(percentagesPath)
        plt.xlabel('Time')
        plt.ylabel('Percentage')
        plt.title('Evolution of 100*Q(S(j+1)>(1+rho)^jS(0)))')
        plt.show()