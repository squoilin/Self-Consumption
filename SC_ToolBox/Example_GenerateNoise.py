# -*- coding: utf-8 -*-
"""
This script uses the historical household monitoring data to calibrate a stochastic model of the noise
around the monthly-averaged daily profile.

This is an exemple, only selects one reference time series, and generates plots for two differents methods

Requires:
- The enload library
- TimeSeries.pickle: pickle with a dataframe containing the TimeSeries used for model calibration

Created on Fri Nov 13 12:35:18 2015

@author: Sylvain Quoilin

@author: sylvain
"""

from __future__ import division
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from SC_enload import *
from SC_functions import *
import pandas as pd

show_plots_markov = False
show_plots_psd = True



index_2014 = pd.DatetimeIndex(start='01/01/2014 00:00:00',end='31/12/2014 23:59:00',freq='15min')

[TimeSeries_hist, HouseInfo_hist] = load('pickle/TimeSeries_UK')

f = 4 # frequency (inverse of the time step)

# Takine one household as example:
load = TimeSeries_hist.iloc[:,7].values
mean = np.zeros(8760*f)

#Generate standard load profile for each month:

month_conso = np.zeros(12)
hourly_curve = {}
ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ndays_cum = np.cumsum(ndays)

for m in range(12):
    hourly_curve[m] = np.zeros(96)
    if m==0:
        start = 0
    else:
        start = ndays_cum[m-1]*24*f
    stop = ndays_cum[m]*f*24
    for h in range(24*f):
        vec = load[start+h:stop:24*f]
        #print index_2014[start+h*f:stop:24*f]
        if len(vec) != ndays[m]:
            print 'Wrong vector length'
        hourly_curve[m][h] = np.mean(vec)
    for d in range(ndays[m]):
        mean[start+d*24*f:start+(d+1)*24*f] = hourly_curve[m]
error = load - mean

log_error = np.log(load) - np.log(mean)
log_error = np.maximum(-3,log_error)


print  'mean log error : ' + str(np.mean(log_error))

#plotHeatmap(scale_vector(log_error,8760))
noise = addNoise(np.mean(log_error)*np.ones(35040),3,1,Lmin=-10)
#reconsitute signal:
load_noise = np.exp(noise + np.log(mean))

if show_plots_markov:
    plt.figure(1)
    plt.plot(*LDC_load(load),label='Load')      
    plt.plot(*LDC_load(mean),label='Mean')     
    plt.plot(*LDC_load(load_noise),label='simulated load') 
    plt.legend()  
    
    plt.figure(2)
    plt.plot(load,label='original load')
    plt.plot(load_noise,label='simulated load')
    plt.legend()
    
    plt.figure(4)
    plotBoxplot_h(log_error)
    
    plt.figure(5)
    plt.plot(*LDC_load(log_error,min=-3),label='log_error')   
    plt.plot(*LDC_load(noise,min=-3),label='noise')
    plt.legend()
    
    plt.figure(6)
    plt.hist(log_error)




##Sample new loads from load duration curve
curve = log_error
#curve = scale_vector(log_error,8760)
N = len(curve)
min = -3

#Generate spectrally colored loads
#Retrieve original load spectrum

curve_ldc = genLoadsFromLDC(LDC_load(curve,Lmin=min),N=N)
PSD = plt.psd(curve, Fs=1, NFFT=N, sides='twosided') #TODO: check the relation of NFFT, Fs, and Dt (Nyquist criterion)

#use sampled load that respects the marginal distribution with no spectrum
Sxx = PSD[0]
curve_psd = genLoadfromPSD(Sxx, curve_ldc, 1)

#Reconstruct original curve:
load_psd = mean*np.exp(curve_psd)



if show_plots_psd:
    fig = plt.figure(figsize=(14,3))
    plt.plot(curve, linewidth =.3,label='Original time series')
    plt.plot(-np.sort(-curve),color='red',linewidth=2,label='Duration curve')
    plt.legend()
    plt.title('Lognorm error between the orginal values and the averaged daily profiles')
    plt.xlim(0,35040)
    plt.xlim(xmin=0)
    
    
    plt.figure(9)
    plt.plot(curve)
    plt.plot(curve_psd)
    
    plt.figure(10)
    plt.plot(*LDC_load(load),label='Load duration curve (original)')      
    plt.plot(*LDC_load(mean),label='Averaged load duration curve')     
    plt.plot(*LDC_load(load_psd),label='Simulated load duration curve, PSD method') 
    plt.xlim(xmin=0)
    plt.legend()  
    
    #Compare LDC and PSD of simulated and 'real' loads
    fig = plt.figure(figsize=(16,3))
    plt.subplot(1, 2, 1)
    
    plt.plot(*LDC_load(curve_psd,Lmin=min),label='LDC of simulated noise') 
    plt.plot(*LDC_load(curve,Lmin=min),label='LDC of original noise') 
    plt.xlim(xmin=0);
    plt.legend()
    
    plt.subplot(1, 2, 2)
    psd1 = plt.psd(curve_psd,NFFT=1024,label='PSD from simulated noise');
    psd2 = plt.psd(curve,NFFT=1024,label='PSD from original noise'); 
    plt.legend()
    
    #TODO : Fix scaling of simulated load PSD
    #TODO : Filter to remove high frequencies -> Unrealistic loads
    
    
    #Compare LDC and PSD of simulated and 'real' loads
    fig = plt.figure(figsize=(16,3))
    plt.subplot(1, 2, 1)
    
    plt.plot(*LDC_load(load),label='LDC of simulated load') 
    plt.plot(*LDC_load(load_psd),label='LDC of sampled load') 
    plt.legend()
    plt.xlim(xmin=0);
    
    plt.subplot(1, 2, 2)
    psd1 = plt.psd(load,NFFT=1024,label='PSD of simulated load'); 
    psd2 = plt.psd(load_psd,NFFT=1024,label='PSD from original load'); 
    plt.legend()





