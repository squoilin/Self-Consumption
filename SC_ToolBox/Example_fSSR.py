# -*- coding: utf-8 -*-
"""
Example script calling the fitted SSR function.

The function is called two times with two different sets of coefficients
- The first set comprises 10 coefficients (first version)
- The second set comprises 16 coefficients (second version)

Created on Fri Mar  4 01:27:31 2016

@author: Sylvain Quoilin (JRC)
"""

from __future__ import division
from SC_functions import fSSR

# Inputs:
Demand = 3500  # kWh
r_PV = 0.8     # kWh/kWh
r_bat = 0.8   # kWh/MWh

# test the coefficients of the 16 parameters function:
coef16 = [ 32.60336587,  38.22038589,   0.85403284,   1.01873506,
        13.26810645,   2.0917934 ,  -4.7601832 ,  24.58864616,
         8.99814429,   1.74242786,   1.37884009,   1.22066461,
        34.31965513,   1.45866917,   0.37348925,  15.02694745]

coef16 = [ 32.603,  38.220,   0.854,   1.019,
        13.268,   2.092 ,  -4.760 ,  24.589,
         8.998,   1.742,   1.379,   1.221,
        34.320,   1.459,   0.373,  15.027]


# Coefficicents of the 10 parameters function:
coef10 = [ 32.60336587,  38.22038589,  0.85403284,  1.01873506,   7.3875943,
   1.38969202,   1.30456212,  37.57288367,   1.33778432,   2.08175158]
   
# Test function:
SSR16 = fSSR(r_PV,r_bat,coef16)
SSR10 = fSSR(r_PV,r_bat,coef10)

print 'For a yearly consumption of ' + str(Demand) + ' kWh, a yearly PV generation of ' \
      + str(Demand*r_PV) + ' kWh, and a battery capacity of ' + str(Demand*r_bat/1000) \
      + ' kWh, the amount of self-consumed energy is ' + str(SSR16*Demand/100) + ' kWh (SSR=' \
      + str(SSR16) + '%)'