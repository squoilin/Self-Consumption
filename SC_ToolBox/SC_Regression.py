# -*- coding: utf-8 -*-
"""
Fitting of the bivariate SSR surface

The input is a pickle file with a list of 3 arrays:
PV: 2D numpy array of input values for the PV ratio
BAT: 2D numpy array of input values for the battery ratio
SSR: 2D numpy array of the self-sufficiency rate for the provided inputs

The fitting is performed in a recursive way: 
- First the SSR for PV=1 and without battery is fixed
- Then, the important 1D curves are fitted: 
    SSR vs PV for BAT=0
    SSR vs BAT for PV = 1
- Finally these curves are extended into a 2D surface which conserves them. 

@author: Sylvain Quoilin, JRC

February 2016
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pylab import colormaps
cmaps = colormaps()
from sklearn import metrics



def SCregression(PV,BAT,SSR,show_plots=True):
    def func_PV1(x, a, b, c):
        '''
        Function used to fit the coefficients (a,b,c) of the SSR curve for r_PV = 1
        :param x: Vector of battery capacities (relative)
        :global coef[0]: Value of SSR for r_PV = 1 and r_bat = 0
        '''
        r_bat = x
        return (coef[0] + a * np.tanh(b*r_bat)+ c* r_bat)
        
    def func_BAT0(x, a, b, c, d):
        '''
        Function used to fit the coefficients (a,b,c,d) of the SSR curve for r_bat = 0
        :param x: Vector of relative PV yearly generation
        '''
        r_PV = x
        return (a * np.tanh(b*r_PV)+ c* r_PV + d* r_PV**0.5)    
    
    def func_10param(x, a, b, c, d, e, f, g, h):
        '''
        First version of the fSSR function. This formulation presents a defect for
        r_bat = 0 and r_PV < 1
        :param x: Matrix of PV capacities (first column) and battery capacities (second column)
        :global coef[:4]: Value of the SSR coefficients for r_PV = 1
        '''    
        r_PV = x[:,0]
        r_bat = x[:,1]
        
        smaller = (r_PV < 1)
        larger = (r_PV >= 1)
        
        return  (coef[0] + coef[1] * np.tanh(coef[2]*r_bat)+ coef[3] * r_bat) + \
                ((a + b* r_bat) * np.tanh(c*(r_PV-1))+ d* (r_PV-1))* larger + \
                ((e) * np.tanh(f * (1 - r_PV)))* smaller
    
    def func_16param(x, a, b, c, d, e, f, g, h):
        '''
        Second version of the fSSR function, solving the defect at r_bat=0
        :param x: Matrix of PV capacities (first column) and battery capacities (second column)
        :global coef[:4]: Value of the SSR coefficients for r_PV = 1
        :global coef[4:8]: Value of the SSR coefficients for r_bat = 0    
        '''  
        r_PV = x[:,0]
        r_bat = x[:,1]
               
        smaller = (r_PV < 1)
        larger = (r_PV >= 1)
        
        W = np.minimum(0.4,np.maximum(0,r_bat))
        
        return  W/0.4 * \
                ((coef[0] + coef[1] * np.tanh(coef[2]*r_bat)+ coef[3] * r_bat) * (1 + smaller*(r_PV -1)) + \
                ((a + b* r_bat) * np.tanh(c*(r_PV-1))+ d* (r_PV-1))* larger + \
                (e * np.tanh(f * (1 - r_PV)))* smaller * r_PV) + \
                (0.4 - W)/0.4 * \
                np.maximum(r_PV, (coef_PV[0] * np.tanh(coef_PV[1]*r_PV)+ coef_PV[2]* r_PV + coef_PV[3]* r_PV**0.5) \
                * ( 1 + g*np.tanh(r_bat))+h*np.tanh(r_bat))
    
    # Defining the array of found coefficients, which will be filled progressively
    coef = np.zeros(16)
    
    # Constant term:
    coef[0] = SSR[5,0]
    print 'Contant term: SSR_0 = ' + str(coef[0])
    
    # reference curve at PV = 1
    popt_1dim, pcov_1dim = curve_fit(func_PV1, BAT[5,:], SSR[5,:])
    y_PV1 = func_PV1(BAT[5,:], popt_1dim[0], popt_1dim[1], popt_1dim[2])
    # fill the coefficent corresponding to that curve:
    coef[1:4] = np.array(popt_1dim)
    
    print '1D coefficient at PV=1: ' + str(popt_1dim[0]) + ', ' + str(popt_1dim[1]) + ', ' + str(popt_1dim[2])
    
    # reference curve at BAT = 0
    popt_1dim, pcov_1dim = curve_fit(func_BAT0, PV[:,0], SSR[:,0])
    y_BAT0 = func_BAT0(PV[:,0], popt_1dim[0], popt_1dim[1], popt_1dim[2],popt_1dim[3])
    # fill the coefficent corresponding to that curve:
    coef_PV = np.array(popt_1dim)
    coef[4:8] = np.array(popt_1dim)
    
    print '1D coefficient at BAT=0: ' + str(popt_1dim[0]) + ', ' + str(popt_1dim[1]) + ', ' + str(popt_1dim[2])  + ', ' + str(popt_1dim[3])
    
    # Formating the data in columns:
    xdata = np.array([PV.flatten(), BAT.flatten()]).transpose()
    ydata_ssr = SSR.flatten()
    
    # Fitting the bivariate function
    #popt, pcov = curve_fit(func_10param, xdata, ydata_eps, p0=[7.38, 2.08, 1.389, 1.304, 37.572,1.33, 0,0.1])
    popt, pcov = curve_fit(func_16param, xdata, ydata_ssr, p0=[7.38745733, 2.08175193,   1.38972295,   1.30460235, 1, 1, 1,0])
    
    # Filling the remaining coefficients
    coef[8:] = np.array(popt)
    
    # Checking the prediction:
    y = func_16param(xdata, popt[0], popt[1], popt[2],  popt[3], popt[4],  popt[5], popt[6], popt[7])

    SSR_pred = y.reshape(PV.shape)
    
    if show_plots:
        fig1 = plt.figure()
        #plt.scatter(ydata_eps,y)
        plt.scatter(ydata_ssr,SSR_pred.flatten())
        plt.title('Predicted vs real SSR value')
        R2 = metrics.r2_score(ydata_ssr,SSR_pred.flatten())
        print 'R2 = ' + str(R2)
        
        fig2 = plt.figure()
        pos = 5
        plt.plot(BAT[pos,:], SSR[pos,:],label='Original values')
        plt.plot(BAT[pos,:], SSR_pred[pos,:],label='Fitted values')
        plt.title('SSR vs BAT with PV = ' + str(PV[pos,0]))
        plt.legend(loc=2)
        
        fig3 = plt.figure()
        pos = 0
        plt.plot(PV[:,pos],SSR[:,pos],label='Original values' )
        plt.plot(PV[:,pos], SSR_pred[:,pos],label='Fitted values',linestyle='--')
        plt.title('SSR vs PV with BAT = ' + str(BAT[0,pos]))
        plt.legend(loc=4,fontsize=18)
        plt.grid()
        plt.ylim(0,50)
        plt.ylabel('SSR [%]',fontsize=18)
        plt.xlabel('PV [kWh/kWh]',fontsize=18)
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(15,260)
        #ax.plot_trisurf(SCR_pred.flatten(),SSR_pred.flatten(),BAT_hr.flatten(),cmap=cmaps[4])
        ax.plot_surface(PV,BAT,SSR_pred,cmap=cmaps[32],rstride=1, cstride=1)
        plt.title('Predicted battery size')
        ax.set_xlabel('SCR')
        ax.set_ylabel('SSR')
        ax.set_zlabel('Battery')
    
    
    print 'All Coefficients: ' + str(coef)
    
    return coef



def fSSR(r_PV,r_bat, a):
    '''
    SSR as a function of the PV and battery relative sizes (10 parameters version). 
    :param r_PV: Yearly PV generation divided by yearly demand (kWh/kWh)
    :param r_bat: Battery capacity divided by yearly demand (kWh,MWh)
    :param a: List of 10 empirical coefficients
    '''
    if r_PV == 0:
        SSR = 0
    elif r_PV < 1:
        SSR = r_PV * ((a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat) + (a[7] * np.tanh(a[8] * (1 - r_PV))))
    else:
        SSR = ((a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat) +  ((a[4] + a[9]* r_bat) * np.tanh(a[5]*(r_PV-1))+ a[6]* (r_PV-1)))
    
    return np.minimum(100,np.maximum(0,SSR))


def fSSR2(r_PV,r_bat, a):
    '''
    SSR as a function of the PV and battery relative sizes (16 parameters version). 
    :param r_PV: Yearly PV generation divided by yearly demand (kWh/kWh)
    :param r_bat: Battery capacity divided by yearly demand (kWh,MWh)
    :param a: List of 16 empirical coefficients
    '''
    
    # Weigthing factor to impose an accuracte, fitted curve at r_bat = 0:
    W = np.minimum(1,np.maximum(0,r_bat))
    
    if r_PV == 0:
        SSR = 0
    elif r_PV < 1:
        SSR = W * \
              (a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat + \
              (a[12] * np.tanh(a[13] * (1 - r_PV)))) * r_PV + \
              (1 - W) * \
              np.maximum(r_PV, (a[4] * np.tanh(a[5]*r_PV)+ a[6]* r_PV + a[7]* r_PV**0.5) \
              * ( 1 + a[14]*np.tanh(r_bat))+a[15]*np.tanh(r_bat))
    else:
        SSR = W * \
              (a[0] + a[1] * np.tanh(a[2]*r_bat)+ a[3] * r_bat + \
              (a[8] + a[9]* r_bat) * np.tanh(a[10]*(r_PV-1))+ a[11]* (r_PV-1)) + \
              (1 - W) * \
              np.maximum(r_PV, (a[4] * np.tanh(a[5]*r_PV)+ a[6]* r_PV + a[7]* r_PV**0.5) \
              * ( 1 + a[14]*np.tanh(r_bat))+a[15]*np.tanh(r_bat))   
    return np.minimum(100,np.maximum(0,SSR))   


