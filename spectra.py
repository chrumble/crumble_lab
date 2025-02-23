###############################################################################
# Chris Rumble
# 1/20/22
#
# A series of functions for analyzing spectra. Mostly ports of what I had been 
# doing in MATLAB.
###############################################################################
import numpy as np
import re

###################################
# find a fwhm from the global max #
###################################
def fwhm(x, sig):
    # normalize and find peak
    sig = sig/sig.max()
    ind = sig.argmax()
    
    # find the 0.5 points on either side of max
    left  = closest(0.5, sig[0:ind])
    right = closest(0.5, sig[ind:])
    
    fwhm = x[right] - x[left]
    
    print(x[right])
    print(x[left])
    
    return fwhm

##########################
# thinning in wavelength #
##########################
def thin(si, wi, n_thin):
    n_wi   = len(wi)
    n_ti   = len(si[0,:])
    n_wf   = int((n_wi - np.remainder(n_wi, n_thin))/n_thin)
    wf     = np.zeros(n_wf)
    ind_0  = 0
    ind_1  = n_thin
    s_thin = np.zeros([n_wf, n_ti])
    for i in range(n_wf):
        wf[i]       = np.mean(wi[ind_0:ind_1])
        s_thin[i,:] = np.sum(si[ind_0:ind_1, :], axis=0)
        
        ind_0 = ind_0 + n_thin
        ind_1 = ind_1 + n_thin
    return s_thin, wf
    
##############################
# smooth with moving average #
##############################
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'same') / w

#########################################
# find the closest match to a selection #
#########################################
def closest(values, x):
    values_shape = np.shape(values)
    ind          = np.zeros(values_shape, dtype='int64')
    
    # if values is just a scaler
    if np.size(values_shape) == 0:
        ind = np.argmin(np.abs(values-x))
    # if values are a vector
    elif np.size(values_shape) == 1:
        for i in range(values_shape[0]):
            ind[i] = np.argmin(np.abs(values[i]-x))
    # if values are a 2D array
    elif np.size(values_shape) == 2:
        for i in range(values_shape[0]):
            for j in range(values_shape[1]):
                ind[i,j] = np.argmin(np.abs(values[i,j]-x))
    # if not
    else:
        print('Values array must be a scaler, vector, or 2D array.')
        return
    return ind

############################################
# convert absorption spectra to wavenumber #
############################################
def abs_to_freq(wave, spec, interp=True):
    spec, n = check_dim(spec)
    
    freq = 1e4/wave
    
    if interp:
        freq_new = np.linspace(np.min(freq), np.max(freq), len(freq))
    
        for i in range(n):
            spec[:,i] = np.interp(x=freq_new,
                                  xp=freq, 
                                  fp=spec[:,i])
        freq = freq_new   
        
    if n == 1:
        spec = np.squeeze(spec)
    
    return freq, spec

##########################################
# convert emission spectra to wavenumber #
##########################################
def em_to_freq(wave, spec, interp=True):
    # prepare some things
    spec, n = check_dim(spec)
    
    # convert wavelengths into frequencies in m^-1
    freq = 1/(wave*1e-9)
    
    # do the conversion
    for i in range(n):
        spec[:,i] = 1e14*spec[:,i]/freq**2
        
    # resample to evenly spaced frequencies unless told not to
    if interp:
        freq_new = np.linspace(np.max(freq), np.min(freq), len(freq))
    
        for i in range(n):
            dum = np.interp(x=np.flipud(freq_new),
                            xp=np.flipud(freq), 
                            fp=np.flipud(spec[:,i]))
            spec[:,i] = np.flipud(dum)
        freq = freq_new

    # put frequencies in kcm^-1
    freq = freq*1e-5;
    
    if n == 1:
        spec = np.squeeze(spec)
    
    return freq, spec

##############################################
# apply emission correction function to data #
##############################################
def em_corr(wave, spec):
    # prepare some things
    corr = np.loadtxt('/home/crumble/Documents/Altoona/research/py_modules'+
                      '/em_corr_221206.txt')
    spec, n = check_dim(spec)

    for i in range(n):
        spec[:,i] = spec[:,i]*np.interp(x=wave,
                                        xp=corr[:,0],
                                        fp=corr[:,1])
        
    if n == 1:
        spec = np.squeeze(spec)
    return spec

#####################################
# baseline correct a set of spectra #
#####################################
def baseline(x, y, ends):
    # prepare some things
    y, n = check_dim(y)
    x, n = check_dim(x)

    # find the range to search
    dum = np.zeros(2, dtype=int)
    for i in range(2):
        dum[i] = np.argmin(np.abs(x-ends[i]))
    dum.sort()
    
    # do the correction
    for i in range(n):
        y[:,i] = y[:,i] - y[dum[0]:dum[1], i].mean()
    
    # if only one spectrum, reshape back to being 1D
    if np.shape(y)[1] == 1:
        y = np.reshape(y, len(y))
        
    return y

###################################
# area normalize a set of spectra #
###################################
def nrm_area(y, x=None):
    # prepare some things
    y, n = check_dim(y)
    num_pts = len(y[:,0])
    nrm     = np.zeros(n)
    if n == 1:
        y = np.reshape(y, [num_pts, 1])


    for i in range(n)    :
        nrm[i] = np.trapz(x=x, y=y[:,i])
        y[:,i] = y[:,i]/nrm[i]
        
    if n == 1:
        y = np.squeeze(y)
    return y

###################################
# peak normalize a set of spectra #
###################################
def nrm_peak(y, x=None, ends=False, together=False, width=4, nrm_return=False,
             ind_return=False):
    # prepare some things
    y, n = check_dim(y)
    num_pts = len(y[:,0])
    nrm     = np.zeros(n)
    ind     = np.zeros(n)
    
    # find the range to search if requested
    if isinstance(ends,list):
        dum = np.zeros(2, dtype=int)
        for i in range(2):
            dum[i] = np.argmin(np.abs(x-ends[i]))
        dum.sort()
    else:
        if np.ndim(y) == 1:
            dum = np.asarray([0, len(y)], dtype=int)
        else:
            dum = np.asarray([0, len(y[:,0])], dtype=int)
            
    
    # determine the maximum by fitting the maximum region with a quadratic
    for i in range(n):
        # find the index of the maximum in each spectrum
        ind[i] = np.argmax(y[dum[0]:dum[1],i])
        ind[i] = ind[i] + dum[0]
        
        # determine the ranage of the polyfit and check for boundary issues
        if (ind[i]-width) >= 1 and (ind[i]+width) <= num_pts:
            poly_range = np.arange(start=ind[i]-width,
                                   stop =ind[i]+width,
                                   step =1, dtype=int)
        elif ind[i]-width < 0:
            poly_range = np.arange(start=0,
                                   stop =ind[i]+width,
                                   step =1, dtype=int)
        elif (ind[i]+width) > num_pts:
            poly_range = np.arange(start=ind[i]-width,
                                   stop =num_pts-1,
                                   step =1, dtype=int)
        
        # do the polynomial fit and get normalization factor
        p   = np.polyfit(np.linspace(0, 1, len(poly_range)), 
                         y[poly_range,i], 2)
        fit = np.polyval(p, np.linspace(0, 1, len(poly_range)))
        nrm[i] = fit.max()

    # normalize individually (together=False) or to the max nrm factor 
    if together:
        y = y/nrm.max()
    else:
        for i in range(n):
            y[:,i] = y[:,i]/nrm[i]
            
    # if only one spectrum, reshape back to being 1D
    if np.shape(y)[1] == 1:
        y = np.reshape(y, len(y))
        
    if ind_return:
        return y, ind
    if nrm_return:
        return y, nrm
    else:
        return y
    
############################################################
# reads in an file from the Agilent UV-vis and fluorometer #
############################################################
def read_ss(filename):
    data = np.loadtxt(filename, 
                      skiprows=2, 
                      delimiter=',',
                      usecols=(0,1))
    
    return data
    
##########################################################
# save a set of two-dimensional data in ideal.dat format #
##########################################################
def write_dat(freq, t, data, output_name):
    out = np.zeros([len(freq)+1, len(t)+1])
    out[1:, 1:] = data
    out[1:,0]   = freq
    out[0,1:]   = t
    
    np.savetxt(output_name, out)

##########################################################
# read a set of two-dimensional data in ideal.dat format #
##########################################################
def read_dat(filename):
    data = np.loadtxt(filename)
    t    = data[0,1:]
    freq = data[1:,0]
    spec = data[1:, 1:]
    
    return t, freq, spec

############################################
# handle single dimensional spectra nicely #
############################################
def check_dim(spec):
    if spec.ndim == 1:
        n = 1
        spec = np.reshape(spec, [len(spec),1])
    elif spec.ndim ==2:
        n = np.shape(spec)[1]
        
    return spec, n

###################################################
# read data from the BH TCSPC system w/ time axis #
###################################################
def read_tcspc(filename):
    data = list()
    with open(filename, 'r') as f:
        for line in f:
            dum = re.split(' +', line)
            if len(dum) == 2 and len(dum[0]) > 0:
                data.append(dum)
    data = np.asarray(data, dtype=float)
    
    return data

########################################
# read data from Heitz's streak camera #
########################################
def read_dac(filename):
    # read in the raw file data
    raw = list()
    with open(filename, 'r') as f:
        for line in f:
            dum = re.split('\t', line)
            raw.append(dum)
    
    # parse the file into t, wln, and data
    n    = len(raw)
    wln  = np.asarray(raw[0][1:], dtype='float64')
    t    = np.zeros([n-1])
    data = np.zeros([len(wln), len(t)])
    
    for i in range(n-1):
        t[i]      = np.asarray(raw[i+1][0], dtype='float64')
        data[:,i] = np.asarray(raw[i+1][1:], dtype='float64')
    
    if raw[0][0].split('|')[0] == 'ps':
        t = t/1000    
    out = list()
    out.append(t)
    out.append(wln)
    out.append(data)
    
    return out




























