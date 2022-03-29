#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: AndreaMazzon
"""

import numpy as np
import math
from random import seed

from processSimulation.eulerDiscretizationForBlackScholesWithLogarithm import EulerDiscretizationForBlackScholesWithLogarithm


class SystemEvolution:
    """
    This class simulates the value of a system of firms in the market identified by carbon emissions and green investements.
    The evolution of the value of a firm depends indeed (also) on such quentities. In particular, our simple "toy" model
    is based on the following assumptions:
    - Green investments have a cost, identified by a function c: c(G_t) is the cost of the green investements G_t
      at time t.
    - On the other hand, due to the fact that investors are getting more and more aware about climate change issues,
      they may favour firms investing on green technologies: this reputational advantage for firms which are heavily
      into green investements is identified by a term Psi_t G_t, where Psi_t is the value at time t of a Geometric
      Brownian motion.
    - Carbon emissions give an immediate production income, identified by a function p: p(C_t) is the production income
      given by a carbon emission C_t at time t.
    - On the other hand, again due the increasing awareness of investors with respect to climate related topics, firms
      with high carbon emissions are exposed to reputational risk. This results in a loss -Phi_t C_t, where Phi is a
      Geometric Brownian motion.
    - Moroever, firms which rely too much on carbon emissions might be seriously impacted by taxes decided by some
      decision maker in order to favour the transition towards a greener economy. The impact is (for now! This can be
      made better) equal to -C_t.
    - Finally, all the firms can be occasionally hit by some natural disasters, with an impact D_t at time t which is
      an exogenous parameter of the model. The probability that a calamity happens in a small interval of time close to
      time t is an increasing function of the average of ALL emissions (i.e., of ALL firms) from time 0 to t.

      Based on the observations/assumptions above, we model the evolution of the values X_i, i=1,...,m of m firms, as
      dX^i_t = (-c(G^i_t) + Psi_t G^i_t + p(C^i_t) - Phi_t C^i_t) dt + \sigma X^i_t dB_t - - C^i_t dN^T_t - D_tdN^D_t,
      where B is a Brownian motion, sigma>0 and N^T, N^D are jump processes: N^T indicates tha arrival of a new tax
      and N^D of a new natural disaster.
      The probability that N^D happens in an interval of time dt is atan(X^1_t+...X^m_t)dt
      (which is indeed in (0,1)): the higher the emissions, the higher the temperature, the higher the probability of
      extreme climate events.
      The probability that N^T happens in an interval of time dt is atan(2*(X^1_t+...X^m_t))dt:
      a decision maker has to promptly react if the total emissions are too high.

     Attributes
    ----------
    numberOfSimulations : int
        the number of simulated pathsfor any single firm
    timeStep : float
        the time step of the time discretization.
    finalTime : float
        the final time of the time discretization.
    initialValues : array
        the initial values of the firms.
    sigma : float
        the volatilities of the values of the firms (supposed to be the same for all)
    numberOfCompanies : int
        the number of the firms
    matrixOfGreenInvestmentsInTime : array
        a matrix representing the evolution of the money put in green investements for any company: the i-th row represents
        all the investements of the companies at fixed time t_i
    matrixOfCarbonEmissionInTime : two-dimensional array
        a matrix representing the evolution of the emissions for any company: the i-th row represents all the emissions
        of the companies at fixed time t_i
    costFunction : function (it has to be vectorized!)
        the function c such that c(G_t) is the cost of green investements G_t, where G_t is the NEW money put in green investements
        at time t
    productionFunction : function (it has to be vectorized!)
        the function p such that p(C_t) is the productivity of the NEW emissions C_t
    impactOfNaturalDisaster : float
        the impact of a natural disaster for the firms
    mySeed : int, optional
        the seed to the generation of the random realizations
    realizations: three-dimensional array
        a three dimnensional array representing the realizations of the evolution of the values of the firms: the
        i,j,k element is the value of the firm j, at time i for the k-th simulation
    stochasticDriverForBrandDisruption : two-dimensional array
        a matrix representing the values of the Geometric Brownian motion process Phi such that -Phi_t C^i_t is the
        impact of reputational disruption for firm i due to its new emissions C^i_t
    stochasticDriverForBrandEnhancement : two-dimensional array
        a matrix representing the values of the Geometric Brownian motion process Psi such that Psi_t G^i_t is the
        impact of reputational advantage for firm i due to its new green investements G^i_t

    Methods
    ----------

    getRealizations():
        It returns all the values of all the firms
    getRealizationsAtGivenTimeIndex(timeIndex):
        It returns the realizations of the values of the firms at a given time index. This is a two dimensional matrix.
    getRealizationsAtGivenTime(time):
        It returns the realizations of the value of the firms at a given time. This is a two dimensional matrix
    getAverageRealizationsAtGivenTimeIndex(timeIndex):
        It returns the average realizations of the values of all the firms at a given time index
    getAverageRealizationsAtGivenTime(time):
        It returns the average realizations of the values of all the firms at a given time

    """

    def __init__(self, numberOfSimulations, timeStep, finalTime, initialValues,
                 sigma, matrixOfGreenInvestmentsInTime, matrixOfCarbonEmissionInTime,
                 costFunction, productionFunction, impactOfNaturalDisaster, mySeed=None):

        """


         Parameters
        ----------
        numberOfSimulations : int
            the number of simulated paths.
        timeStep : float
            the time step of the time discretization.
        finalTime : float
            the final time of the time discretization.
        initialValues : array
            the initial capitals of the firms.
        sigma : array
            the volatilities of the processes
        interestRate : float
            the interest rate
        mySeed : int, optional
            the seed to the generation of the standard normal realizations
        getEvolutionAverage(self):
            it returns the evolution of the average realizations of the values of all the firms

        Returns
        -------
        None.

        """
        self.numberOfSimulations = numberOfSimulations
        self.timeStep = timeStep
        self.finalTime = finalTime
        self.initialValues = initialValues
        self.sigma = sigma

        #the number of columns of this matrix
        self.numberOfCompanies =  len(matrixOfCarbonEmissionInTime[0])
        self.matrixOfGreenInvestmentsInTime = matrixOfGreenInvestmentsInTime
        self.matrixOfCarbonEmissionInTime = matrixOfCarbonEmissionInTime
        self.costFunction = costFunction
        self.productionFunction = productionFunction
        self.impactOfNaturalDisaster = impactOfNaturalDisaster
        self.mySeed = mySeed

        # we generate all the paths for all the simulations
        self.__generateRealizations()

    def __generateRealizations(self):

        #we generate all the realizations (i.e., all the simulated trajectoris) of the stochastic
        #processes that will drive the (positive and negative) impact of brand reputation and the
        #(negative) impact of emissions related taxes
        self.__generateStochasticDriverForBrandEnhancement()
        self.__generateStochasticDriverForBrandDisruption()

        numberOfTimes = math.ceil(self.finalTime / self.timeStep) + 1

        # realizations[i,j,k] is the value of firm j at time t_i for the k-th simulation. Having the simulations as
        # last dimension can for example help computing the averages (see the methods to get the averages)
        self.realizations = np.zeros((numberOfTimes,self.numberOfCompanies, self.numberOfSimulations))
        self.realizations[0] = np.full((self.numberOfCompanies, self.numberOfSimulations), self.initialValues)


        randomNumberGenerator = np.random.RandomState(self.mySeed)

        #these are used in order to simulate the Brownian motions for the diffusion part

        standardNormalRealizations = randomNumberGenerator.standard_normal((numberOfTimes, self.numberOfSimulations))

        #they are used instead to simulate the events "new tax" and "new natural disaster": such event happen when
        #the realizations of random variables uniformly distributed in (0,1) are smaller than the probability
        #of such events to happen. Same idea as for the Monte-Carlo implementation of the binomial model
        uniformlyDistributedRealizationsForNaturalDisaster = randomNumberGenerator.uniform(0,1,(numberOfTimes, self.numberOfSimulations))
        uniformlyDistributedRealizationsForNewLaw = randomNumberGenerator.uniform(0,1,(numberOfTimes, self.numberOfSimulations))

        
        for timeIndex in range(1, numberOfTimes):

            #this is a two-dimensional array: it contains realizations at past time for any firm and any simulation
            pastRealizations = self.realizations[timeIndex- 1]

            #the i-th element of this array is new money put in green investements at this fixed time
            #from the i-th company
            vectorOfGreenInvestmentsInTime = self.matrixOfGreenInvestmentsInTime[timeIndex]

            #the i-th element of this array is the carbon emission at this fixed time of the i-th company
            vectorOfCarbonEmissionInTime = self.matrixOfCarbonEmissionInTime[timeIndex]

            # we now construct: diffusion (next line, this is easy) drift and jumps

            #CONSTRUCTION OF THE DIFFUSION TERM
            diffusion = self.sigma * pastRealizations

            #CONSTRUCTION OF THE JUMP TERM

            #the average of all the carbon emissions of all companies up to the current time: this identifies the
            #probability of a new climate disaster in a small interval close to the current time
            averagePastCarbonEmissions = np.mean(self.matrixOfCarbonEmissionInTime[0:timeIndex + 1])

            #we want these probabilities to be proportional to the time interval of the discretization
            probabilityOfNewLaw = math.atan(2*averagePastCarbonEmissions)*self.timeStep
            probabilityOfNaturalDisaster = math.atan(averagePastCarbonEmissions)*self.timeStep

            #same idea as in the Monte-Carlo implementation of the binomial model: a new law (stating a new tax punishing
            #carbon emissions) is established at current time t_i in the j-th simulation if u_{i,j} < probability_i,
            #where u_{i,j} is an element of the matrix of realizations of a random variable uniformly distributed in (0,1)
            #and probability_i is the probability that the new tax is established close to time t_i.
            #This is then a vector whose j-th element is FALSE if the tax is NOT established at time t_i in the j-th simulation
            #and TRUE if the tax IS established at time t_i in the j-th simulation
            isNewLawEstablished = uniformlyDistributedRealizationsForNewLaw[timeIndex] < probabilityOfNewLaw

            # same thing for the natural disaster
            doesNaturalDisasterHappen = uniformlyDistributedRealizationsForNaturalDisaster[timeIndex]<probabilityOfNaturalDisaster


            #here we want that jumpsForNewLaw[i,j] = isNewLawEstablished[j]*carbonEmissionOfFirm[i]: at every iteration of
            #this kind of hidden for loop, a new row of jumpsForNewLaw is constructed
            jumpsForNewLaw = [np.where(isNewLawEstablished, 0, 0) for carbonEmissionOfFirm in vectorOfCarbonEmissionInTime]

            #now we want to do the same thing. In this case impactOfNaturalDisaster is a number, so we get a one-dimensional array
            jumpsForNaturalDisasterAsVector = np.where(doesNaturalDisasterHappen, 0, 0)

            #we then sum the above matrix and the above array: these are the jumps. If they are not zero, they  have
            #a negative impact on the increment of the values of the firms
            jumps = - (jumpsForNewLaw + jumpsForNaturalDisasterAsVector)

            #CONSTRUCTION OF THE DRIFT TERM

            # this is the matrix whose i-th row is (-c(G_t^i)+Psi_t(omega_1) G_t^i, -c(G_t^i)+Psi_t(omega_2) G_t^i,..., -c(G_t^i)+Psi_t(omega_n) G_t^i)
            partOfTheDriftFromGreenInvestments =\
                np.array([-self.costFunction(greenInvestmentsInTimeOfFirm) + self.stochasticDriverForBrandEnhancement[timeIndex]
                          * greenInvestmentsInTimeOfFirm
                                        for greenInvestmentsInTimeOfFirm in vectorOfGreenInvestmentsInTime])

            # this is the matrix whose i-th row is (P(C_t^i)-Phi_t(omega_1) C_t^i, P(C_t^i)-Phi_t(omega_2) G_t^i,..., P(C_t^i)-Phi_t(omega_n) G_t^i)
            partOfTheDriftFromEmissions =\
                np.array([self.productionFunction(carbonEmissionInTimeForFirm) -
                          self.stochasticDriverForBrandDisruption[timeIndex] * carbonEmissionInTimeForFirm
                                        for carbonEmissionInTimeForFirm in vectorOfCarbonEmissionInTime])

            drift = partOfTheDriftFromGreenInvestments + partOfTheDriftFromEmissions

            self.realizations[timeIndex] = pastRealizations +  drift * self.timeStep + \
                                           diffusion * math.sqrt(self.timeStep) * standardNormalRealizations[timeIndex] + \
                                           jumps


    #the two stochastic drivers are generated with same initial value, drift and volatility. Moreover, these are given here
    #as parameters that teh user cannot set. Should we change this?

    def __generateStochasticDriverForBrandEnhancement(self):
        simulator = EulerDiscretizationForBlackScholesWithLogarithm(self.numberOfSimulations, self.timeStep, self.finalTime,
                                                                    0.02, 0.0, 0.02, self.mySeed)
        self.stochasticDriverForBrandEnhancement = simulator.getRealizations()


    def __generateStochasticDriverForBrandDisruption(self):
        simulator = EulerDiscretizationForBlackScholesWithLogarithm(self.numberOfSimulations, self.timeStep, self.finalTime,
                                                                    0.02, 0.0, 0.02, self.mySeed)
        self.stochasticDriverForBrandDisruption = simulator.getRealizations()

    def getRealizations(self):
        """
        It returns all the values of all the firms

        Returns
        -------
        array
            three-dimensional matrix whose [i,j,k] entry is the value of firm j at time t_i for the k-th simulation

        """
        return self.realizations

    def getRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the realizations of the values of the firms at a given time index. This is a two dimensional matrix.

        Parameters
        ----------
        timeIndex : int
             the time index, i.e., the row of the matrix of realizations.

        Returns
        -------
        array
            a two-dimensional matrix whose [j,k] entry is the value of firm j  for the k-th simulation at time specified by timeIndex

        """

        return self.realizations[timeIndex]

    def getRealizationsAtGivenTime(self, time):
        """
        It returns the realizations of the value of the firms at a given time. This is a two dimensional matrix.

        Parameters
        ----------
        time : float
             the time at which we want to get the values

        Returns
        -------
        array
            a two-dimensional matrix whose [j,k] entry is the value of firm j  for the k-th simulation at time specified
            as input

        """

        indexForTime = round(time / self.timeStep)
        return self.realizations[indexForTime]


    def getAverageRealizationsAtGivenTimeIndex(self, timeIndex):
        """
        It returns the average realizations of the values of all the firms at a given time index

        Parameters
        ----------
        time : index
             the time index for which the averages are returned

        Returns
        -------
        array
            a one-dimensional vector whose i-th element is the average of the values of the firm i at time index specified in
            the input
        """

        realizationsAtGivenTime = self.getRealizationsAtGivenTimeIndex(timeIndex)
        average = [np.average(rowOfTheMatrixOfRealization) for rowOfTheMatrixOfRealization in
                   realizationsAtGivenTime]
        return average

    def getAverageRealizationsAtGivenTime(self, time):
        """
        It returns the average realizations of the values of all the firms at a given time

        Parameters
        ----------
        time : float
             the time at which the averages are returned

        Returns
        -------
        array
            a uno-dimensional vector whose i-th element is the average of the values of the firm i at time specified in
            the input
        """
        realizationsAtGivenTime = self.getRealizationsAtGivenTime(time)
        average = [np.average(rowOfTheMatrixOfRealization) for rowOfTheMatrixOfRealization in
                   realizationsAtGivenTime]
        return average

    def getEvolutionAverage(self):
        """
        It returns the evolution of the average realizations of the values of all the firms

        Parameters
        ----------
        Returns
        -------
        array
              a two-dimensional matrix whose i-th raw is the evolution of the values of the firm i
            """
        numberOfTimes = int(np.ceil(self.finalTime/self.timeStep))
        evolutionAverage = np.array([self.getAverageRealizationsAtGivenTimeIndex(timeIndex) for timeIndex in range(0,numberOfTimes)])
        return evolutionAverage