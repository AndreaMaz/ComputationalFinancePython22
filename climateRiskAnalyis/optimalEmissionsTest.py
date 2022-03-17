#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Andrea Mazzon
"""

#This module is the test of the model described at page 143 of the script
import math
import numpy as np

from optimalEmissions import maximizeExpectedWealth

lossFunction = lambda y : y**1.2 #note that it is convex!
consumptionFunction = lambda y : y #the cost

#the probability functions for the experts: note that they are in (0,1) for positive abatement
probabilityFunctions = lambda a : np.array([0.5*(1-math.atan(0.5*a)), 0.5*(1-math.atan(0.5*a)), 0.5*(1-math.atan(0.5*a)),
                                   0.5*(1-math.atan(0.5*a)),  0.5*(1-math.atan(0.5*a))])

ambiguityAversionFunction = lambda x : x**1.5 #note that it is convex!

initialWealth = 4

weightsExperts = [1/5,1/5,1/5,1/5,1/5]
# weightsExperts = [1/4,1/4,1/8,1/8,1/4]
#weightsExperts = [0,0,1,0,0]


probabilitiesGivenTippingPoint = [1/2, 2/3, 2/3]

lossesGivenTippingPoints = np.array([3.8, 3.8, 3.9])

discountFactor = 0.9

optimalAbatement = maximizeExpectedWealth(lossFunction, consumptionFunction, probabilityFunctions,
                                           ambiguityAversionFunction, initialWealth, weightsExperts, probabilitiesGivenTippingPoint,
                                           lossesGivenTippingPoints, discountFactor)

print()
print("The optimal abatement is ", optimalAbatement)
