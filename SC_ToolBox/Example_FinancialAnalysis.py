# -*- coding: utf-8 -*-
"""
Example script for the use of the FinancialAnalysis function

- Loads country data for Germany 
- Imposes investment costs
- Performs a parametric study by varying r_PV and r_bat 
- plots the results

Created on Fri Feb 26 22:59:57 2016

@author: Sylvain Quoilin, JRC
"""

from __future__ import division
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import colormaps
cmaps = colormaps()
import sys
sys.path.append('data/')
from SC_functions import FinancialAnalysis,EnergyFlows

# Inputs:
r_PV = 1.34
r_bat = 1
Demand = 3500
eta_inv = 0.96
eta_bat = 0.92

# Selection of the country within the table:
country='Germany'
code = 'DE'

# Coefficients of the 16 parameters fSSR function:
coef = [ 32.60336587,  38.22038589,   0.85403284,   1.01873506,
        13.26810645,   2.0917934 ,  -4.7601832 ,  24.58864616,
         8.99814429,   1.74242786,   1.37884009,   1.22066461,
        34.31965513,   1.45866917,   0.37348925,  15.02694745]

coef = [ 32.60336587,  38.22038589,  0.85403284,  1.01873506,   7.3875943,
   1.38969202,   1.30456212,  37.57288367,   1.33778432,   2.08175158]
   
# Investment parameters
Inv = {'FixedPVCost':0,'PVCost_kW':1500,'FixedBatteryCost':300,'BatteryCost_kWh':200,'PVLifetime':20,'BatteryLifetime':10,'OM':0.015}

# Load PV generation data:
PV_data = pd.read_excel('data/PVCapacityFactors.xlsx',index_col=0,skiprows=range(4))
CapacityFactorPV = float(PV_data.Generation[code])

# Load country data:
CountryData = pickle.load(open('pickle/regulations.pickle','rb'))

# Parametric study: varying r_PV and r_bat:
ratios_PV = np.linspace(1,2,20)
ratios_bat = np.linspace(0.6,1.6,10)

# Defining grid:
PV,BAT = np.meshgrid(ratios_PV,ratios_bat)
LCOE = np.zeros(PV.shape)

# Loop over all array components:
for i in range(len(ratios_PV)):
    for j in range(len(ratios_bat)):
        r_PV = ratios_PV[i]
        r_bat = ratios_bat[j]
        # Compute the energy flows with the given coefficients and inputs values:
        E = EnergyFlows(r_PV,r_bat,Demand,eta_inv,eta_bat,CapacityFactorPV,coef)
        F = FinancialAnalysis(E,CountryData.loc[country,:].to_dict(),Inv)
        LCOE[j,i] = F['LCOE']
       
# 3D plot:
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(30,-110)
#ax.plot_surface(np.log(PV+1),np.log(BAT+1),SSR,cmap=cmaps[32],rstride=1, cstride=1)
ax.plot_surface(PV,BAT,LCOE,cmap=cmaps[32],rstride=1, cstride=1)
ax.set_xlabel('PV [kWh/kWh]')
ax.xaxis.label.set_fontsize(16)
ax.set_ylabel('BATTERY [kWh/MWh]')
ax.yaxis.label.set_fontsize(16)
plt.title('LCOE [EUR/kWh]',fontsize=16)

# Contour plot:
import matplotlib
import matplotlib.cm as cm

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

levels = np.arange(258,266,0.5)
fig3 = plt.figure(figsize=(8, 6))
CS = plt.contour(PV, BAT, LCOE*1000, colors='black', linewidths=1.,levels=levels)
CS2 = plt.contourf(PV, BAT, LCOE*1000, cmap=cm.Purples, alpha=0.5,levels=levels)
plt.grid()
plt.clabel(CS, inline=1, fontsize=10)
plt.title('LCOE [EUR/kWh]')
plt.xlabel('PV [kWh/kWh]')
plt.ylabel('BATTERY [kWh/MWh]')







