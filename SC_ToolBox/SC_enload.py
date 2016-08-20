# -*- coding: utf-8 -*-
"""
Python library with helpful functions to analyze and simulate energy loads.
Copyright (c) 2015-2016 Konstantinos Kavvadias

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


def makeTimeseries(data, year=2015, freq=None):
    """Convert numpy array to a pandas series with a timed index.
    1D annual data only. #TODO: generalize to 2D and pd.Series
    Parameters:
        data: numpy array
        year: year of timeseries
        freq: 15min, H
    """
    startdate = pd.datetime(year, 1, 1, 0, 0, 0)
    enddate = pd.datetime(year, 12, 31, 23, 59, 59)
    if freq is None:
        if len(data) == 8760:
            freq = 'H'
        elif len(data) == 35040:
            freq = '15min'
        else:
            raise AssertionError('Input vector length must be 8760 or 35040')
    date_list = pd.DatetimeIndex(start=startdate, end=enddate, freq=freq)
    out = pd.Series(data, index=date_list)  # FIXME does not work with pd.Series or pd.Dataframe
    return out


def reshape_timeseries(Load, x='dayofyear', y='hour', aggfunc='sum'):
    """Returns a reshaped pandas DataFrame that shows the aggregated load for selected 
    timeslices. e.g. time of day vs day of year

    Parameters:
        Load (Series): timeseries
        x (str): x axis aggregator. Has to be an accessor of pd.DatetimeIndex
         (year, dayoftime, week etc.)
        y (str): similar to above for y axis
    Returns:
        reshaped pandas dataframe according to x,y
    """
    if isinstance(Load, pd.Series):
        a = Load.to_frame(name=0)
    elif isinstance(Load, pd.DataFrame):
        a = Load.copy()
        a.name = 0
    elif isinstance(Load, np.ndarray):
        print ('Input of function is not a Pandas Series. Trying to transform')
        a = makeTimeseries(Load)
    else:
        raise AssertionError('Input has to be either ndarray, series or dataframe')
    if not isinstance(Load.index, pd.DatetimeIndex):
        raise AssertionError('Pandas Index has to be a DatetimeIndex')
    a[x] = getattr(a.index, x)
    a[y] = getattr(a.index, y)
    a = a.reset_index(drop=True)
    return a.pivot_table(index=x, columns=y,
                         values=0, aggfunc=aggfunc).T



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
    out = makeTimeseries(np.nan, year, freq='H')  # Create empty pandas with datetime index
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
        print ('Error')
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
        print ('Error')
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
        N (optional): Length of produced timeseries (if Y is not provided)

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


def addNoise(Load, mode, st, Lmin=0):
    """ Add noise based on specific distribution
    LOAD_ADDNOISE Add noise to a 1x3 array [El Th Co].
       Mode 1 = Normal Dist
       Mode 2 = Uniform Dist
       Mode 3 = Gauss Markov
       st = Noise parameter
    """
    def GaussMarkov(mu, st, r):
        """A.M. Breipohl, F.N. Lee, D. Zhai, R. Adapa
        A Gauss-Markov load model for the application in risk evaluation
        and production simulation
        Transactions on Power Systems, 7 (4) (1992), pp. 1493-1499
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
        y[y < Lmin] = Lmin  # remove negative elements
        return y
    if not isinstance(Load, pd.Series):
        print ('Input of function is not a Pandas Series. Trying to transform it')
        Load = makeTimeseries(Load)
    if st == 0:
        print('No noise to add')
        return Load
    loadlength = len(Load)  # 8760
    noisevector = st * np.random.randn(loadlength)
    if mode == 1:  # Normal
        out = Load * (1 + noisevector)
    elif mode == 2:  # Uniform #TODO
        out = Load * ((1-.2) + 2 * .2 * np.random.rand(loadlength))
    elif mode == 3:  # Gauss-Markov
        r = 0.9
        out = pd.Series(GaussMarkov(Load, st, r), index=Load.index)
    return out


def LDC_load(load, bins=999, Lmin=0):
    """Generates the Load Duration Curve based on a given load
    load : energy timeseries
    Returns an array [x, y] ready for plotting (e.g. plt(*LDC_load(load)))
    """
    #remove nan because histogram does not work
    load_masked = load[~np.isnan(load)]
    n, xbins = np.histogram(load_masked, bins=bins, density=True)
    # xbins = xbins[:-1] #remove last element to make equal size
    cum_values = np.zeros(xbins.shape)
    cum_values[1:] = np.cumsum(n*np.diff(xbins))
    out = np.array([1-cum_values, xbins])
    # out = np._r[[1 0], out] # Add extra point
    out[out < Lmin] = Lmin  # Trunc non zero elements
    return out


def climacogram(Load):
    """ Climacogram: standard deviation versus timescale. See D. Koutsoyiannis (2002,2014)
    pd.Series.resample() from Pandas >= 0.18 
    Parameters:
        Load: energy timeseries
    Returns:
        Climacogram x, y ready to be plotted e.g. (plt(*get_climacogram(load))
    """
    # TODO : Parametrize: from offset rules to time
    offset_rules = ['d', '2d', 'w', '2w', '4w', '8w', '16w', '52w', '200w']
    x = [1, 2, 7, 7*2, 7*4, 7*8, 7*16, 7*52, 7*200]
    y_stds = []

    for i in offset_rules:
        a_temp = Load.resample(i).sum()  # TODO sum gives resampled ts, mean gives SMA
        y_stds.append(a_temp.std())
        #plt.plot(a_temp,lw=.4,alpha=.7) # Plot all ts. TODO How to return plots?
    return x, y_stds


def LDC_empirical(U, duration=8760, bins=1000):
    """Generates the Load Duration Curve based on empirical parameters.
    Parameters:
        U: parameter vector [Peak load, capacity factor%, base load%, hours]
    Returns an array [x, y] ready for plotting (e.g. plt(*LDC_empirical(U)))
    """
    P = U['peak']  # peak load
    CF = U['LF']  # load factor
    BF = U['base']  # base load
    h = U['hourson']  # hours

    x = np.linspace(0, P, bins)

    ff = h * ((P - x)/(P - BF * P))**((CF - 1)/(BF - CF))
    ff[x < (BF*P)] = h
    ff[x > P] = 0
    return ff/duration, x


def getLoadCharacteristics(Load):
    """Find load profile characteristics
    To be used for validation of simulated loads.
    Parameters:        
        load: timeseries of load to be examined
    Returns parameter dict {peak, load factor, base load factor, operating hours}
    """
    # hours = len(Tload) #TODO 2D
    hourson = sum(Load > 0)
    Load_trunc = Load[Load > 0]  # TRUNC load to non zero elements
    P = np.percentile(Load_trunc, 99.99)
    LF = np.mean(Load_trunc) / P
    BF = np.percentile(Load_trunc, 1) / P
    # M = np.corrcoef(Load) 
    # TODO: Stochastic Information ? AR MA ? parcorr
    return {'peak': P,
            'base': BF,
            'LF': LF,
            'hourson': hourson}


#  Plotting functions ####

def plotHeatmap(Load, x='dayofyear', y='hour', aggfunc='sum', bins=8,
                palette='Oranges', colorbar=True, **pltargs):
    """ Returns a 2D heatmap of the reshaped timeseries based on x, y
    Parameters:
        Load: 1D pandas with timed index
        x: Parameter for reshape_timeseries()
        y: Parameter for reshape_timeseries()
        bins: Number of bins for colormap
        palette: palette name (from colorbrewer, matplotlib etc.)
        **pltargs: Exposes matplotlib.plot arguments
    Returns
        2d heatmap
    """
    x_y = reshape_timeseries(Load, x=x, y=y, aggfunc=aggfunc)

    fig, ax = plt.subplots(figsize=(16, 6))
    cmap_obj = cm.get_cmap(palette, bins)
    heatmap = ax.pcolor(x_y, cmap=cmap_obj, edgecolors='w', **pltargs)
    if colorbar:
        fig.colorbar(heatmap)
    ax.set_xlim(xmax=len(x_y.columns))
    ax.set_ylim(ymax=len(x_y.index))
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def plot3d(Load, x='dayofyear', y='hour', aggfunc='sum', bins=15,
           palette='Oranges', colorbar=True, **pltargs):
    """ Returns a 30 plot of the reshaped timeseries based on x, y
    Parameters:
        Load: 1D pandas with timed index
        x: Parameter for reshape_timeseries()
        y: Parameter for reshape_timeseries()
        bins: Number of bins for colormap
        palette: palette name (from colorbrewer, matplotlib etc.)
        **pltargs: Exposes matplotlib.plot arguments
    Returns
        2d heatmap
    """
    from mpl_toolkits.mplot3d import Axes3D

    x_y = reshape_timeseries(Load, x=x, y=y, aggfunc=aggfunc)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')
    cmap_obj = cm.get_cmap(palette, bins)
    X, Y = np.meshgrid(range(len(x_y.columns)), range(len(x_y.index)))
    surf = ax.plot_surface(X, Y, x_y, cmap=cmap_obj, rstride=1, cstride=1,
                           shade=False, antialiased=True, lw=0)
    if colorbar:
        fig.colorbar(surf)
    # Set viewpoint.
    # ax.azim = -130
    ax.elev = 45
    ax.auto_scale_xyz([0, len(x_y.columns)],
                      [0, len(x_y.index)],
                      [0, x_y.max().max()])
    ax.set_xlabel(x)
    ax.set_ylabel(y)



def plotBoxplot(Load, by='day', **pltargs):
    """Return boxplot plot for each day of the week
    Keyword arguments:
        Load : 1D pandas Series with timed index
        by: make plot by day or hour
        **pltargs: Exposes matplotlib.plot arguments
    """
    if not isinstance(Load, pd.Series):
        print 'Input of plotBoxplot is not a Pandas Series. Trying to transform it'
        Load = makeTimeseries(Load)

    if by == 'day':
        iter = Load.groupby(Load.index.weekday)
        labels = "Mon Tue Wed Thu Fri Sat Sun".split()
    elif by == 'hour':
        iter = Load.groupby(Load.index.hour)
        labels = np.arange(0, 24)
    else:
        raise NotImplementedError('Only "day" and "hour" are implemented')
    a = []
    for timestep, value in iter:
        a.append(value)
    plt.boxplot(a, labels=labels, **pltargs)
    # TODO : Generalize to return monthly, hourly etc.


if __name__ == "__main__":
    pass
