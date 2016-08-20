# -*- coding: utf-8 -*-
"""
Example script for the optimization procedure:

- Loads country data for Germany 
- Imposes investment costs
- Performs an optimization of the system for different battery costs 
- plots the results

Created on Fri Mar  4 01:27:31 2016

@author: Sylvain Quoilin (JRC)
"""

from __future__ import division
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('data/')
from SC_Optimization import SCoptim
from SC_functions import FinancialAnalysis,EnergyFlows,load


# Selection of the country within the table:
country='Germany'
code = 'DE'

# Hard coded system parameters:
Demand = 3500
eta_inv = 0.96
eta_bat = 0.92

# Coefficients of the 16 parameters fSSR function:
coef = [ 32.60336587,  38.22038589,   0.85403284,   1.01873506,
        13.26810645,   2.0917934 ,  -4.7601832 ,  24.58864616,
         8.99814429,   1.74242786,   1.37884009,   1.22066461,
        34.31965513,   1.45866917,   0.37348925,  15.02694745]

# Coefficients of the 10 parameters fSSR function:
#coef = [ 32.60336587,  38.22038589,  0.85403284,  1.01873506,   7.3875943,
#   1.38969202,   1.30456212,  37.57288367,   1.33778432,   2.08175158]
   
# Investment parameters
Inv = {'FixedPVCost':0,'PVCost_kW':1500,'FixedBatteryCost':300,'BatteryCost_kWh':200,'PVLifetime':20,'BatteryLifetime':10,'OM':0.015}

# Load PV generation data:
PV_data = pd.read_excel('data/PVCapacityFactors.xlsx',index_col=0,skiprows=range(4))
CapacityFactorPV = float(PV_data.Generation[code])

# Load country data:
CountryData = load('pickle/regulations')

# Parametric study (varying battery cost)
costs = np.arange(100,250,2)  
LCOEs = []
BAT = []
PV=[]
LCOE_stor = []    

for c in costs:
    # Optimisation
    Inv['BatteryCost_kWh'] = c
    [r_PV, r_bat, LCOE] = SCoptim(CapacityFactorPV,CountryData.loc[country,:].to_dict(),Inv,coef)
    BAT.append(r_bat)    
    LCOEs.append(LCOE)
    PV.append(r_PV)
    
    # Recalculating the whole set of values with the optimum inputs:
    E = EnergyFlows(r_PV,r_bat,Demand,eta_inv,eta_bat,CapacityFactorPV,coef)
    F = FinancialAnalysis(E,CountryData.loc[country,:].to_dict(),Inv)
    LCOE_stor.append(F['LCOE_stor'])


# Plotting:
fig4 = plt.figure(2,figsize=[7,5])
plt.subplot(211)
plt.plot(costs,BAT,linestyle='--',linewidth=2, marker='o',label='Battery')
plt.plot(costs,PV,label='PV', marker='o',linewidth=2)
plt.ylabel('Relative PV/Battery sizes [-]')
plt.ylim(0,2.5)
plt.legend(fontsize=16)
plt.grid()
#remove xaxis:
ax = plt.gca()
ax.set_xticklabels([])
ax.yaxis.label.set_fontsize(16)

plt.subplot(212)
plt.plot(costs,LCOEs,label='LCOE [EUR/MWh]',linewidth=2, marker='o')
plt.plot(costs,LCOE_stor,linestyle='--',linewidth=2, marker='o',label='LCOS [EUR/MWh]')
plt.ylabel('LCOE [EUR/MWh]',fontsize=16)
plt.xlabel('Cost of storage [EUR/kWh]',fontsize=16)
plt.legend(loc=4,fontsize=16)
plt.grid()



