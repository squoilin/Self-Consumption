# -*- coding: utf-8 -*-
"""
Example script computing the battery size as a function of the 
self-sufficiency and self-consumption rates

Created on Fri Mar  4 01:27:31 2016

@author: Sylvain Quoilin (JRC)
"""

from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import colormaps
cmaps = colormaps()
from scipy.optimize import fsolve
from SC_functions import fSSR

# Version of the function with only r_bat as input variable (r_PV is provided as a parameter) for fsolve:
def SSR_fsolve(r_bat, *param):
    a, r_PV, SSR = param
    if r_PV == 0:
        return r_bat
    else:
        return SSR - fSSR(r_PV,r_bat,a)
      
# test the coefficients of the 16 parameters function:
coef = [ 32.60336587,  38.22038589,   0.85403284,   1.01873506,
        13.26810645,   2.0917934 ,  -4.7601832 ,  24.58864616,
         8.99814429,   1.74242786,   1.37884009,   1.22066461,
        34.31965513,   1.45866917,   0.37348925,  15.02694745]

# Writing the parameters for SSR_fsolve
param = (coef,1,50)   

# Defining the matrices (10x10 with steps of 10%):
BAT100 = np.zeros([11,11])
check = np.zeros([11,11])

for i in range(11):
    for j in range(11):
        SSR = i * 10
        SCR = j * 10
        if SCR == 0:
            r_PV = 1E9
        else:
            r_PV = float(SSR)/SCR
        param = (coef,r_PV,SSR)  
        temp, info, found, msg = fsolve(SSR_fsolve,1,args=param,full_output=True)
        if found == 1:
            BAT100[i,j] = temp
        else:
            BAT100[i,j]= np.nan
         
        # Check must contain the r_bat input if the fitting was successful
        check[i,j] = fSSR(r_PV,BAT100[i,j],coef)
        
BAT100 = np.maximum(0,BAT100)
BAT100 = np.nan_to_num(BAT100)

X,Y = np.meshgrid(np.arange(0,110,10),np.arange(0,110,10))

# Plotting the required battery size as a function of SSR and SCR:
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(15,290)
ax.plot_surface(X,Y,BAT100,cmap=cmaps[32],rstride=1, cstride=1)
plt.title('Predicted battery size')
ax.set_xlabel('SCR [-]')
ax.set_ylabel('SSR [-]')
ax.set_zlabel('Battery [kWh/MWh]')

BatterySize = pd.DataFrame(BAT100,index=['SSR = ' + str(num) for num in np.arange(0,110,10)],columns=['SCR = ' + str(num) for num in np.arange(0,110,10)])
print BatterySize
BatterySize.to_excel('outputs/battery size.xls')
