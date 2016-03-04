# -*- coding: utf-8 -*-
"""
Example script for the regression process:

- Loads simulated SSR data with its inputs 
- Fits the data
- Calls the functions with the 10 parameters and 16 parameters versions
- plotting the results is optional

Created on Fri Mar  4 01:27:31 2016

@author: Sylvain Quoilin (JRC)
"""

import pickle
import numpy as np
from SC_Regression import *

# Load the data to be fitted:
[PV_hr, BAT_hr, SSR_hr] = pickle.load(open('pickle/SSR_curve_all.pickle','rb'))
    
coef = SCregression(PV_hr,BAT_hr,SSR_hr,show_plots=True)

# test the coefficients of the 16 parameters function:
coef_test = [ 32.60336587,  38.22038589,   0.85403284,   1.01873506,
        13.26810645,   2.0917934 ,  -4.7601832 ,  24.58864616,
         8.99814429,   1.74242786,   1.37884009,   1.22066461,
        34.31965513,   1.45866917,   0.37348925,  15.02694745]

# Test function:
SSR = fSSR2(0.8,0.8,coef_test)

# Coefficicents of the 10 parameters function:
coef10 = [ 32.60336587,  38.22038589,  0.85403284,  1.01873506,   7.3875943,
   1.38969202,   1.30456212,  37.57288367,   1.33778432,   2.08175158]
   
# Test function:
SSR10 = fSSR(1,1,coef10)