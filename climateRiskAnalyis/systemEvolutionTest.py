#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
In this class we do some first tests for the evolution of values of firms putting some money in green investements
and having some carbon emissions.

@author: Andrea Mazzon
"""
import numpy as np
import math

import matplotlib.pyplot as plt

from systemEvolution import SystemEvolution


numberOfSimulations = 10000
numberOfTimes = 301
finalTime = 3
timeStep = finalTime / (numberOfTimes - 1)


numberOfCompanies = 5

initialValues = 20
sigma = 0.2

impactOfNaturalDisaster = 1

#all the firms put the same amount of money on green investements every time
matrixOfGreenInvestmentsInTime = np.full((numberOfTimes, numberOfCompanies), 5)

emissions = [1, 3, 5, 10, 50]
#instead, they differentiate their carbon production
matrixOfCarbonProductionInTime = np.transpose(np.array([np.full((numberOfTimes), emissions[0]),
                                                        np.full((numberOfTimes), emissions[1]),
                                                        np.full((numberOfTimes), emissions[2]),
                                                        np.full((numberOfTimes), emissions[3]),
                                                        np.full((numberOfTimes), emissions[4])]))


#0.02 is also the average of the drivers for the impact of green investements and carbon production on the reputation
costFunction = lambda x : 0.02 * x

productionFunction = lambda x : 0.02 * x


systemGenerator = SystemEvolution(numberOfSimulations, timeStep, finalTime, initialValues,
                 sigma, matrixOfGreenInvestmentsInTime, matrixOfCarbonProductionInTime,
                 costFunction, productionFunction, impactOfNaturalDisaster)


#plot of a path of realizations per firm
realizations = systemGenerator.getRealizations()

for firmIndex in range(0,numberOfCompanies):
    plt.plot(realizations[:,0,firmIndex])

plt.legend(emissions)
plt.title("Evolution of a path of values of firms, identified by emissions")
plt.show()

#plot of the evolution of the average

evolutionAverage = systemGenerator.getEvolutionAverage()

evolutionAverageTransposed = np.transpose(evolutionAverage)

for firmIndex in range(0,numberOfCompanies):
    plt.plot(evolutionAverageTransposed[firmIndex])

plt.legend(emissions)
plt.title("Evolution of the average values of firms, identified by emissions")
plt.show()

plt.show()