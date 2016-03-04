# -*- coding: utf-8 -*-
"""
Python library with helpful functions to analyze and simulate energy loads.
Copyright (c) 2015 Konstantinos Kavvadias

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import pandas as pd
import scipy.stats
import scipy.linalg
import scipy.interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys


def genLoad(curve,min=0,max=np.inf):
    N = len(curve)
    curve_ldc = genLoadsFromLDC(LDC_load(curve,min=min),N=N)
    PSD = plt.psd(curve, Fs=1, NFFT=N, sides='twosided') #TODO: check the relation of NFFT, Fs, and Dt (Nyquist criterion)

    #use sampled load that respects the marginal distribution with no spectrum
    Sxx = PSD[0]
    curve_psd = genLoadfromPSD(Sxx, curve_ldc, 1)

    return curve_psd



def makeTimeseries(data, year=2015):
    """Convert numpy array to a pandas series with a timed index.
    1D data only.            #TODO: generalize to 2D
    """
    startdate = pd.datetime(year, 1, 1, 0, 0, 0)
    enddate = pd.datetime(year, 12, 31, 23, 59, 59)
    if len(data) == 8760:
        freq = 'H'
    elif len(data) == 35040:
        freq = '15min'
    else:
        sys.error('Input vector length must be 8760 or 35040')
    date_list = pd.DatetimeIndex(start=startdate,end=enddate,freq=freq)
    out = pd.Series(data, index=date_list)
    return out


def genLoadsFromDailyMonthly(ML, DWL, DNWL, weight, year=2015):
    """Generate annual timeseries using monthly demand and daily profiles.
    Working days are followed by non-working days which produces loads with
    unrealistic temporal sequence, which means that they cannot be treated as
    timeseries.

    Keyword arguments:
    ML -- monthly load (size = 12)
    DWL -- daily load (working day) (size = 24). Have to be normalized (sum=1)
    DNWL -- daily load (non working day) (size = 24) Have to be normalized (sum=1)
    weight -- weighting factor between working and non working day (0 - 1)
    """
    if np.abs(DWL.sum() + DNWL.sum() - 2) > .01:
        print ('Daily Loads not normalized')
        return
    out = makeTimeseries(0, year)  # Create empty pandas with datetime index
    # Assumptions for working days per month
    Days = np.r_[31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    Days_NW = np.r_[8, 7, 8, 8, 8, 8, 8, 12, 8, 8, 8, 8]  # how many NW days per month ?
    Days = np.c_[Days - Days_NW, Days_NW]

    for month in xrange(12):  # TODO:  when weight<>0.5 there is a small error
        TempW = ML[month] * weight * Days[month, 0] / \
                (weight * Days[month, 0] + (1 - weight) * Days[month, 1]) / Days[month, 0]
        TempNW = ML[month] * (1 - weight) * Days[month, 1] / \
                 (weight * Days[month, 0] + (1 - weight) * Days[month, 1]) / Days[month, 1]
        for hour in xrange(24):
            out.ix[(out.index.month == month + 1) &  #months dont start from 0
                   (out.index.weekday < 5) &
                   (out.index.hour == hour)] = (TempW * DWL[hour])[0]
            out.ix[(out.index.month == month + 1) &
                   (out.index.weekday >= 5) &
                   (out.index.hour == hour)] = (TempNW * DNWL[hour])[0]
    return out


def genSinusLoad(A1a, A1b, A2a, A2b, A3a, A3b, noise=0):
    """Generate sinusoidal load with daily, weekly and yearly seasonality
    with optional gaussian noise
    """
    def sinusFunc(x, w, A1, A2):
        out = A1 * np.cos(2 * np.pi/w * x) + A2 * np.sin(2 * np.pi/w * x)
        return out

    x = np.arange(0, 8760)
    Daily = sinusFunc(x, 24, A1a, A1b)
    Weekly = sinusFunc(x, 168, A2a, A2b)
    Yearly = sinusFunc(x, 8760, A3a, A3b)
    Total = Daily + Weekly + Yearly + noise * np.random.randn(8760)
    out = makeTimeseries(Total)
    return out


def genCorrArrays(Na, hours, M):
    """ Generating correlated normal variates
    Assume one wants to create a vector of random variates Z which is
    distributed according to Z∼N(μ,Σ) where μ is the vector of means,
    and Σ is the variance-covariance matrix.
    http://comisef.wikidot.com/tutorial:correlateduniformvariates

    Na: number of vectors e.g (3)
    hours: vector size (e.g 8760 hours)
    M: correlation matrix

    Returns one array with random generated corellated variables size :
    (Na, hours) e.g. (3, 8760)
    """
    if Na != np.size(M, 0):  # rows of pars have to be the same size as rows and cols of M
        Y = -1
        print('Parameters and corr matrix dimensions do not agree.')
        pass

    newM = M.copy()  # changing an array element without copying it changes it globally!
    u = np.random.randn(hours, Na)
    if min(np.linalg.eig(M)[0]) < 0:  # is M positive definite?
        print 'Error'
        # Make M positive definite:
        la, v = np.linalg.eig(newM)
        la[la < 0] = np.spacing(1)  # Make all negative eigenvalues zero
        ladiag = np.diag(la)  # Diagonal of eigenvalues
        newM = np.dot(np.dot(v, ladiag), v.T)  # Estimate new M = v * L * v'

    # Transformation is needed to change normal to uniform (Spearman - Pearson)
    for i in np.arange(0, Na):  # 1:Na
        for j in np.arange(max(Na-1, i), Na):  # max(Na-1,i):Na
            if i != j:
                newM[i, j] = 2 * np.sin(np.pi * newM[i, j] / 6)
                newM[j, i] = 2 * np.sin(np.pi * newM[j, i] / 6)

    if min(np.linalg.eig(newM)[0]) <= 0:  # check again brute force #while
        print 'Error'
        pass
        # M[1:(Na+1):(Na*Na)] = 1
        # M = hyper_decomp(M)

    cF = scipy.linalg.cholesky(newM)
    Y = np.dot(u, cF).T
    Y = scipy.stats.norm.cdf(Y)  # remove if you produce random.rand?
    return Y


def genLoadsFromLDC(LDC, Y=None, N=8760):
    """ Generate loads based on a Inverse CDF, such as a Load Duration Curve
    Inverse transform sampling: Compute the value x such that F(x) = u.
    Take x to be the random number drawn from the distribution described by F.
    LDC: 2 x N vector of the x, y coordinates of an LDC function.
        x coordinates have to be normalized (max = 1 ==> 8760hrs )
    Y (optional): a vector of random numbers. To be used for correlated loads.
        If None is supplied a random vector (8760) will be created.

    Returns a vector with the same size as Y that follows the LDC distribution
    """
    # func_inv = scipy.interpolate.interp1d(LDC[0], LDC[1])
    # simulated_loads = func_inv(Y)
    # ------- Faster way:   # np.interp is faster but have to sort LDC
    if Y is None:  # if there is no Y, generate a random vector
        Y = np.random.rand(N)
    # if np.all(np.diff(LDC[0]) > 0) == False: #if sorted
    idx = np.argsort(LDC[0])
    LDC_sorted = LDC[:, idx].copy()
    simulated_loads = np.interp(Y, LDC_sorted[0], LDC_sorted[1])
    # no need to insert timed index since there is no spectral information
    return simulated_loads


def genLoadfromPSD(Sxx, x, dt=1):
    """
    Algorithm for generating samples of a random process conforming to spectral
    density Sxx(w) and probability density function p(x).
    This is done by an iterative process which 'shuffles' the timeseries till
    convergence of both power spectrum and marginal distribution is reached.
    Also known as "Iterated Amplitude Adjusted Fourier Transform (IAAFT)"
    Adopted from:
    J.M. Nichols, C.C. Olson, J.V. Michalowicz, F. Bucholtz, (2010)
    "A simple algorithm for generating spectrally colored, non-Gaussian signals
    Probabilistic Engineering Mechanics", Vol 25, 315-322
    Schreiber, T. and Schmitz, A. (1996) "Improved Surrogate Data for
    Nonlinearity Tests", Physical Review Letters, Vol 77, 635-638.

    Keyword arguments:
    Sxx -- Spectral density (two sided)
    x -- Sequence of observations created by the desirable PDF
    dt -- Desired temporal sampling interval. [Dt = 2pi / (N * Dw)]

    """
    N = len(x)
    Sxx[N/2+1] = 0  # zero out the DC component (remove mean)
    Xf = np.sqrt(2*np.pi * N * Sxx / dt)  # Convert PSD to Fourier amplitudes
    Xf = np.fft.ifftshift(Xf)  # Put in Matlab FT format
    # The following lines were commented out because they outscale the data
    # modifying thus its PDF. However, according to Nichols et al. they
    # guarantee that the new data match the signal variance
    #vs = (2 * np.pi / N / dt) * sum(Sxx) * (N / (N-1))  # Get signal variance (as determined by PSD)
    #out = x * np.sqrt(vs / np.var(x))
    out = x
    mx = np.mean(out)
    out = out - mx  # subtract the mean
    indx = np.argsort(out)
    xo = out[indx].copy()  # store sorted signal xo with correct p(x)

    k = 1
    indxp = np.zeros(N)  # initialize counter
    while(k):
        Rk = np.fft.fft(x)  # Compute FT
        Rp = np.angle(Rk)  # ==> np.arctan2(np.imag(Rk), np.real(Rk))  # Get phases
        out = np.real(np.fft.ifft(np.exp(1j * Rp) * np.abs(Xf)))  # Give signal correct PSD
        indx = np.argsort(out)  # Get rank of signal with correct PSD
        out[indx] = xo  # rank reorder (simulate nonlinear transform)
        k = k + 1  # increment counter
        if np.array_equal(indx, indxp):
            print('Converged after %i iterations') % k
            k = 0  # if we converged, stop
        indxp = indx  # re-set ordering for next iter
    out = out + mx  # Put back in the mean
    return makeTimeseries(out)


#  Other Functions ####


def addNoise(load, mode, st,min=0):
    """ Add noise based on specific distribution
    LOAD_ADDNOISE Add noise to a 1x3 array [El Th Co].
       Mode 1 = Normal Dist
       Mode 2 = Uniform Dist
       Mode 3 = Gauss Markov
       st = Noise parameter
    """
    def GaussMarkov(mu, st, r):
        """A.M. Breipohl, F.N. Lee, D. Zhai, R. Adapa
        A Gauss–Markov load model for the application in risk evaluation
        and production simulation
        Transactions on Power Systems, 7 (4) (1992), pp. 1493–1499
        """
        loadlength = len(mu)
        rndN = np.random.randn(loadlength)
        noisevector = st * np.ones(loadlength)
        y = np.zeros(loadlength)
        # noisevector[noisevector==0] = eps;
        y[0] = mu[0] + noisevector[0] * rndN[0]

        for i in xrange(loadlength):
            y[i] = mu[i] + r * noisevector[i] / noisevector[i-1] * \
                (y[i-1] - mu[i-1]) + noisevector[i] * np.sqrt(1 - r**2) * rndN[i]
        y[y < min] = min # remove negative elements
        return y
    if isinstance(load,np.ndarray):
        print 'Input of function is not a Pandas Series. Trying to transform it'
        load = makeTimeseries(load)
    if st == 0:
        print('No noise to add')
        return load
    loadlength = len(load)  # 8760
    noisevector = st * np.random.randn(loadlength)
    if mode == 1:  # Normal
        out = load * (1 + noisevector)
    elif mode == 2:  # Uniform #TODO
        out = load * ((1-.2) + 2 * .2 * np.random.rand(loadlength))
    elif mode == 3:  # Gauss-Markov
        r = 0.9
        out = pd.Series(GaussMarkov(load, st, r), index=load.index)
    return out


def LDC_load(load, bins=999,min=0):
    """Generates the Load Duration Curve based on a given load
    load : energy timeseries
    Returns an array [x, y] ready for plotting (e.g. plt(*LDC_load(load)))
    """
    n, xbins = np.histogram(load, bins=bins, density=True)
    # xbins = xbins[:-1] #remove last element to make equal size
    cum_values = np.zeros(xbins.shape)
    cum_values[1:] = np.cumsum(n*np.diff(xbins))
    out = np.array([1-cum_values, xbins])
    # out = np._r[[1 0], out] # Add extra point
    out[out < min] = min # Trunc non zero elements
    return out


def LDC_empirical(U, TotalHours=8760, bins=1000):
    """Generates the Load Duration Curve based on empirical parameters.
       U: parameter vector [Peak load, capacity factor%, base load%, hours]
       Returns an array [x, y] ready for plotting (e.g. plt(*LDC_empirical(U)))
    """
    P = U[0]  # peak load
    CF = U[1]  # capacity factor
    BF = U[2]  # base load
    h = U[3]  # hours

    x = np.linspace(0, P, bins)

    ff = h * ((P - x)/(P - BF * P))**((CF - 1)/(BF - CF))
    ff[x < (BF*P)] = h
    ff[x > P] = 0
    out = np.array([ff/TotalHours, x])
    return out


def getLoadCharacteristics(load):
    """PROF_CHECK Find load profile characteristics pars and M.
    To be used for validation of simulated loads.

    load: timeseries of load to be examined
    Returns parameter vector, correlation factor
    """
    Tload = load
    # hours = len(Tload)
    hourson = sum(Tload > 0)
    Load_trunc = Tload[Tload > 0]  # TRUNC load to non zero elements
    P = np.percentile(Load_trunc, 99.99)
    CF = np.mean(Load_trunc) / P
    BF = np.percentile(Load_trunc, 1.5) / P
    Tpars = np.r_[P, CF, BF, hourson]
    # M = np.corrcoef(Load)

    # TODO: Stochastic Information ? AR MA ? parcorr
    # TODO: Descriptive version of function
    return Tpars


#  Plotting functions ####


def plotHeatmap(Load, bins=8, **pltargs):
    """ Returns a 2D heatmap that shows the load: time of day vs day of year
    Keyword arguments:
    Load -- 1D pandas with timed index
    bins -- Number of bins for colormap
    **pltargs -- Exposes matplotlib.plot arguments
    """
    if isinstance(Load,np.ndarray):
        print 'Input of function is not a Pandas Series. Trying to transform it'
        Load = makeTimeseries(Load)
    if len(Load) == 8760:
        N = 24
    elif len(Load) == 35040:
        N = 96
    else:
        sys.error('Length should be 8760 or 35040')
    days_and_time = np.zeros([N, 365])
    for day in np.arange(0, 365, 1):  # TODO: Optimize
        for time in np.arange(0, N, 1):
            days_and_time[time][day] = Load.ix[
                                        (Load.index.dayofyear == day + 1) &
                                        (Load.index.hour == time)] #.mean() ?
    fig, ax = plt.subplots(figsize=(16, 6))
    cMap = cm.get_cmap('Oranges', bins)
    heatmap = ax.pcolor(days_and_time, cmap=cMap, edgecolors='w', **pltargs)
    plt.colorbar(heatmap)
    plt.xlim(xmax=365);
    plt.ylim(ymax=N);


def plotBoxplot(Load, **pltargs):
    """Return boxplot plot for each day of the week
    Keyword arguments:
    Load : 1D pandas with timed index
    **pltargs -- Exposes matplotlib.plot arguments
    """
    if isinstance(Load,np.ndarray):
        print 'Input of plotBoxplot is not a Pandas Series. Trying to transform it'
        Load = makeTimeseries(Load)
    a = []
    for day, value in Load.groupby(Load.index.weekday):
        a.append(value)
    days = "Mon Tue Wed Thu Fri Sat Sun".split()
    plt.boxplot(a, labels=days, **pltargs);
    # TODO : Generalize to return monthly, hourly etc.

def plotBoxplot_h(Load, **pltargs):
    """Return boxplot plot for each day of the week
    Keyword arguments:
    Load : 1D pandas with timed index
    **pltargs -- Exposes matplotlib.plot arguments
    """
    if isinstance(Load,np.ndarray):
        print 'Input of plotBoxplot is not a Pandas Series. Trying to transform it'
        Load = makeTimeseries(Load)
    a = []
    for day, value in Load.groupby(Load.index.hour):
        a.append(value)
    #days = "Mon Tue Wed Thu Fri Sat Sun".split()
    plt.boxplot(a, **pltargs);
    # TODO : Generalize to return monthly, hourly etc.


if __name__ == "__main__":
    pass
