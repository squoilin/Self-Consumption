# -*- coding: utf-8 -*-
"""
Example script calling the EnergyFlows function.

Created on Fri Mar  4 01:27:31 2016

@author: Sylvain Quoilin (JRC)
"""

from __future__ import division
from SC_functions import EnergyFlows

# Inputs:
Demand = 3500  # kWh
r_PV = 0.8     # kWh/kWh
r_bat = 1.3    # kWh/MWh
eta_inv = 0.96
eta_bat = 0.92
CapacityFactorPV = 990   # kWh/kWp

# Coefficients of the 16 parameters SSR function:
coef = [ 32.60336587,  38.22038589,   0.85403284,   1.01873506,
        13.26810645,   2.0917934 ,  -4.7601832 ,  24.58864616,
         8.99814429,   1.74242786,   1.37884009,   1.22066461,
        34.31965513,   1.45866917,   0.37348925,  15.02694745]

E = EnergyFlows(r_PV,r_bat,Demand,eta_inv,eta_bat,CapacityFactorPV,coef)


print 'For a yearly consumption of ' + str(Demand) + ' kWh, a yearly PV generation of ' \
      + str(Demand*r_PV) + ' kWh, and a battery capacity of ' + str(Demand*r_bat/1000) \
      + ' kWh, the amount of self-consumed energy is ' + str(E['SC']) + ' kWh (SSR=' \
      + str(E['SSR']) + '%), the energy directly self-consumed (without passing through the battery) is ' \
      + str(E['SSR_0']*Demand/100) + ' kWh, and the amount fed to the grid is ' \
      + str(E['ToGrid']) + ' kWh'