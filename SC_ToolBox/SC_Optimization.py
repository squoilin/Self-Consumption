# -*- coding: utf-8 -*-
"""
This scripts implements the optimization function
for the PV and battery sizes with the objective of 
minimizing LCOE.

Created on Fri Feb 26 22:59:57 2016

@author: Sylvain Quoilin, JRC
"""

from __future__ import division
import numpy as np
import scipy.optimize as optimize
import sys
sys.path.append('../')
from SC_functions import FinancialAnalysis,EnergyFlows


def SCoptim(CapacityFactorPV,CountryData,Inv,coef): 
    '''
    Main optimization function
    
    :param CapacityFactorPV: Yearly capacity factor, kWh/kWp
    :param CountryData: Dictionary with the financial variables of the considered country   
    :param Inv: Investment data. Defined as a dictionary with the fields 'FixedPVCost','PVCost_kW','FixedBatteryCost','BatteryCost_kWh','PVLifetime','BatteryLifetime','OM'
    :param coef: Coefficient of the empirical SSR function the number of coefficients depends on the version of the function
    
    :return: list with the optimal values [r_pv, r_bat, LCOE]
    '''
    
    # Checking that country data is a dictionary:
    if not isinstance(CountryData,dict):
        sys.exit('CountryData must be a dictionnary in the SCoptim function')
    
    # Definition of the objective functionS of the optimization
    def f_optim(c):
        '''
        Objective function. Returns the profitability of the system.
        Note that some global variables are used.
        
        Global variables: Demand,eta_inv,eta_bat,CapacityFactorPV,coef,CountryData,
                          Inv, country    
        
        '''
        [r_PV,r_bat] = c
        # if the ratios are negative, set them to zero and add a penalty to the objective function
        penalty_PV = - 1E10 * np.minimum(0,r_PV)
        penalty_bat = - 1E10 * np.minimum(0,r_bat)
        r_PV = np.maximum(0,r_PV)
        r_bat = np.maximum(0,r_bat)
        
        E = EnergyFlows(r_PV,r_bat,Demand,eta_inv,eta_bat,CapacityFactorPV,coef)
        F = FinancialAnalysis(E,CountryData,Inv)
        
        #print 'PV = ' + str(r_PV) + ', BAT = ' + str(r_bat) + ', PR = ' + str(PR) + ', LCOE = ' + str(LCOE)
        return F['LCOE'] + penalty_PV + penalty_bat
        
        
    def f_optim_0(c):
        '''
        Objective function. Returns the profitability of the system with r_bat = 0.
        Note that some global variables are used.
        
        Global variables: Demand,eta_inv,eta_bat,CapacityFactorPV,coef,CountryData,
                          Inv, country    
        
        '''
        [r_PV] = c
        r_bat = 0
        # if the ratios are negative, set them to zero and add a penalty to the objective function
        penalty_PV = - 1E10 * np.minimum(0,r_PV)
        penalty_bat = - 1E10 * np.minimum(0,r_bat)
        r_PV = np.maximum(0,r_PV)
        r_bat = np.maximum(0,r_bat)
        
        E = EnergyFlows(r_PV,r_bat,Demand,eta_inv,eta_bat,CapacityFactorPV,coef)
        F = FinancialAnalysis(E,CountryData,Inv)
        
        #print 'PV = ' + str(r_PV) + ', BAT = ' + str(r_bat) + ', PR = ' + str(PR) + ', LCOE = ' + str(LCOE)
        return F['LCOE'] + penalty_PV + penalty_bat    
    
    
    
    # Constraints:
    cons = ({'type': 'ineq', 'fun': lambda x: x[0]},
            {'type': 'ineq', 'fun': lambda x: x[1]})
    bnds = ((0, 10), (0, 10))
    
    # Hard coded system efficiencies:
    eta_inv = 0.96
    eta_bat = 0.92
    Demand = 3500
    
    # Since there are 3 discrete variants of the problem, the optimization is performed 3 times and the best one is selected
    # With PV and Battery:
    result = optimize.minimize(f_optim, [1.1,1.1], method='Nelder-Mead', tol=1e-5, constraints=cons, bounds=bnds).values()
    # With PV, without Battery
    result2 = optimize.minimize(f_optim_0, [1.1], method='Nelder-Mead', tol=1e-5, constraints=cons, bounds=bnds).values()
    # Without PV, without battery:
    E_0 = EnergyFlows(0,0,Demand,eta_inv,eta_bat,CapacityFactorPV,coef)
    F_0 = FinancialAnalysis(E_0,CountryData,Inv)
    # Selecting the best solution:
    if F_0 <= result2[3] and F_0 <= result[3]:
        r_PV = 0
        r_bat = 0
        LCOE = F_0
    elif result2[3] <= F_0 and result2[3] <= result[3]:
        r_PV = result2[4][0]
        r_bat = 0
        LCOE = result2[3]
    elif result[3] <= F_0 and result[3] <= result2[3]:
        r_PV = result[4][0]
        r_bat = result[4][1]
        LCOE = result[3]

    return [r_PV, r_bat, LCOE]  
    
    
