#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

#This module is based on page 143 of the script: it solves an optimization problem to find the optimal emission abatement
from scipy.optimize import minimize, Bounds
import numpy as np



def maximizeExpectedWealth(lossFunction, consumptionFunction, probabilityFunctions,
                                               ambiguityAversionFunction, initialWealth, weightsExperts, probabilitiesGivenTippingPoint,
                                               lossesGivenTippingPoints, discountFactor):
    """
    It returns the analytical value of an european call option written
    on a Black-Scholes model.

    Parameters
    ----------
    lossFunction: function
        a vectorized, increasing, convex function
    consumptionFunction: function
        the function giving the cost of emission abatement
    probabilityFunctions: array of functions
        the i-th element of this array is a decreasing function, indicating the probability
        of the tipping point as a function of the abatement according expert i
    ambiguityAversionFunction: function
        an increasing, convex function indicating ambiguity aversion
    weightsExperts: vector of positive numbers summing up to 1
        its i-th element indicates the "trust" on the i-th expert
    probabilitiesGivenTippingPoint: array
        gives the probability of the extreme climate events, conditional to the fact
        that the tipping point happens
    lossesGivenTippingPoints: array
        the losses caused by the extreme climate events, conditional to the fact
        that the tipping point happens
    discountFactor: float
        the factor by which we discount future losses

    Returns
    -------
     float:
        the optimal emission abatement to solve the minimization problem at page 143 of the script.

        """

    #this is the function that we want to optimize
    def computeExpectedWealthForEmissionsReduction(emissionsReduction):

        costOfTheEmissionAbatement = lossFunction(consumptionFunction(emissionsReduction))

        # note that this is applied to a vector!
        lossesByExtremeEvents = lossFunction(lossesGivenTippingPoints)

        # the losses weighted by their probabilities, CONDITIONAL TO THE TIPPING POINT TO HAPPEN:
        #in order to get the expectation, we have to multiply by the probability that the tipping point
        #happen
        expectedLossConditionalToTippingPoint = np.dot(lossesByExtremeEvents, probabilitiesGivenTippingPoint)
        #note that probabilityFunctions(emissionsReduction) is an array! The element i is the probability
        #according to expert i
        expectedLossForExperts = probabilityFunctions(emissionsReduction) * expectedLossConditionalToTippingPoint

        #we now take into consideration ambiguity aversion
        expectedLossForExpertsWithAmbiguityAversionFunction = ambiguityAversionFunction(expectedLossForExperts)

        expectedLossModelUncertainty = np.dot(expectedLossForExpertsWithAmbiguityAversionFunction, weightsExperts)

        return (costOfTheEmissionAbatement + discountFactor * expectedLossModelUncertainty)

    #the bounds for the optimal abatement
    bounds = Bounds(0, 10)

    #we now perform the minimization
    res = minimize(computeExpectedWealthForEmissionsReduction, x0 = 0.5, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True}, bounds=bounds)

    return res.x[0]

