# -*- coding: utf-8 -*-
"""
This script uses the historical household monitoring data to calibrate a stochastic model of the noise
around the monthly-averaged daily profile.

Iterates over all houshole profiles

Requires:
- The enload library
- TimeSeries.pickle: pickle with a dataframe containing the TimeSeries used for model calibration

Created on Fri Nov 13 12:35:18 2015

@author: Sylvain Quoilin
"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from enload import *
#from SC_functions import *
import pickle
import pandas as pd


index_2014 = pd.DatetimeIndex(start='01/01/2014 00:00:00',end='31/12/2014 23:59:00',freq='15min')

[TimeSeries_hist, HouseInfo_hist] = pickle.load(open('pickle/TimeSeries_UK.pickle','rb'))

Noise = pd.DataFrame(index=TimeSeries_hist.index)

f = 4 # frequency (inverse of the time step)

ndays = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ndays_cum = np.cumsum(ndays)

for key in TimeSeries_hist.keys():
    #Example with the first household:
    load = TimeSeries_hist[key].values
    mean = np.zeros(8760*f)
    
    #Generate standard load profile for each month:
    month_conso = np.zeros(12)
    hourly_curve = {}

    
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
    

    N = len(log_error)
    min = -3
    
    #Generate spectrally colored loads
    #Retrieve original load spectrum
    noise_ldc = genLoadsFromLDC(LDC_load(log_error,min=min),N=N)
    PSD = plt.psd(log_error, Fs=1, NFFT=N, sides='twosided') #TODO: check the relation of NFFT, Fs, and Dt (Nyquist criterion)
    
    #use sampled load that respects the marginal distribution with no spectrum
    Sxx = PSD[0]
    noise_psd = genLoadfromPSD(Sxx, noise_ldc, 1)
    noise_psd.name = key
    
    #Reconstruct original curve:
    load_psd = mean*np.exp(noise_psd)
    
    Noise[key] = noise_psd.values
    
    #break
        
pickle.dump(Noise,open('outputs/Noise.pickle','w'))


