###############################################################################
# Chris Rumble
# 1/5/22
#
# A series of functions for fitting data. Mostly a port of what I had been 
# doing in MATLAB, but fancier, like, with objects and modules and stuff.
###############################################################################
import numpy as np
from scipy.optimize import least_squares
from scipy.special  import erf
from scipy.special  import gamma
from dataclasses    import dataclass

####################################
# class for holding fit input data #
####################################
@dataclass
class FitData:
    guess:   float = 0
    lb:      bool  = False
    ub:      float = 0
    fix:     int   = 0
    x:       float = 0
    data:    float = 0
    f:       int   = 0
    fitpar:  float = 0
    fit:     float = 0
    resid:   float = 0
    co_det:  float = 0
    tau_int: float = 0
    n_fit:   int   = 0
    weight:  bool  = False

##############
# linear fit #
##############
def fit_polarizer(res):
    # assign the linear model
    res.f = polarizer
    
    # perform the fit
    res = fit_base(res)
    
    return res

def polarizer(par, x):
    polar = par[0]*np.sin(par[1]*x + par[2]) + par[3]
    return polar

##############
# linear fit #
##############
def fit_line(res):
    # assign the linear model
    res.f = line
    
    # perform the fit
    res = fit_base(res)
    
    return res

def line(par, x):
    line = par[0]*x + par[1]
    return line

######################
# exponential fitter #
######################
# simple multi-exponential model
def fit_exp(res: FitData):
    # assign the exponential model
    res.f = multiexp
    
    # perform the fit
    res = fit_base(res)

    # calculate the integral time
    for i in range(res.n_fit):
        norm           =  res.fitpar[i,[2,4,6]].sum()
        res.tau_int[i] = (res.fitpar[i,2]*res.fitpar[i,3] +
                          res.fitpar[i,4]*res.fitpar[i,5] +
                          res.fitpar[i,6]*res.fitpar[i,7])/norm
    
    # print the result
    varnames = [r'h', r'bkg', r'a1', r'tau1', r'a2',
                r'tau2', r'a3', r'tau3', r'<tau>']
    print('---------------------------------------------------------------------------------')
    print('Multi-Exponential Fit Results')
    print('---------------------------------------------------------------------------------')
    print('%9s%9s%9s%9s%9s%9s%9s%9s%9s' % tuple(varnames))
    for i in range(res.n_fit):
        print('%9.3e%9.3e%9.3e%9.3e%9.3e%9.2e%9.3e%9.2e%9.3e' % 
              (*res.fitpar[i,:], res.tau_int[i]))
    print('---------------------------------------------------------------------------------')
    print('Fix:%5d%9d%9d%9d%9d %9d %9d %9d'% tuple(res.fix))
    print('---------------------------------------------------------------------------------')
    return res

# multi-exponential model
def multiexp(par, x):
    exp = par[0]*(par[2]*np.exp(-x/par[3]) + 
                  par[4]*np.exp(-x/par[5]) +
                  par[6]*np.exp(-x/par[7])) + par[1]
    return exp

#########################
# stretched exponential #
#########################
def fit_strexp(res: FitData):
    # assign the exponential model
    res.f = strexp
    
    # perform the fit
    res = fit_base(res)

    # calculate the integral times
    tau_str = np.zeros((res.n_fit, 2))
    res.tau_int = np.zeros(res.n_fit)
    for i in range(res.n_fit):
        tau_str[i,0] = (res.fitpar[i,3]/res.fitpar[i,4]*
                        gamma(1/res.fitpar[i,4]))
        tau_str[i,1] = (res.fitpar[i,5]/res.fitpar[i,6]*
                        gamma(1/res.fitpar[i,6]))
        res.tau_int[i] = (res.fitpar[i,2]*tau_str[i,0] + 
                          (1-res.fitpar[i,2])*tau_str[i,1])
    
    # print the result
    varnames = [r'h', r'bkg', r'a1', r'tau1', r'beta1', r'tau2', r'beta2',
                r'<str1>', r'<str2>', r'<tau>']
    print('---------------------------------------------------------------------------------')
    print('Stretched-Exponential Fit Results')
    print('---------------------------------------------------------------------------------')
    print('%8s%8s%8s%8s%8s%8s%8s%8s%8s%8s' % tuple(varnames))
    for i in range(res.n_fit):
        print('%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f%8.3f %8.3f %8.3f %8.3f' % 
              (*res.fitpar[i,:], tau_str[i, 0], tau_str[i, 1], res.tau_int[i]))
    print('---------------------------------------------------------------------------------')
    print('Fix:%4d%8d%8d%8d%8d%8d%8d'% tuple(res.fix))
    print('---------------------------------------------------------------------------------')
    return res

# multi-exponential model
def strexp(par, x):
    h     = par[0]
    bkg   = par[1]
    a1    = par[2]
    t1    = par[3]
    beta1 = par[4]
    t2    = par[5]
    beta2 = par[6]
    
    # build the decay
    decay = np.zeros(len(x))
    decay = decay + a1*np.exp(-(x/t1)**beta1)
    decay = decay + (1-a1)*np.exp(-(x/t2)**beta2)    
    
    # apply height and background
    decay = h*decay + bkg
    
    return decay

####################
# power law fitter #
####################
def fit_power(res):
    # assign the reorg model
    res.f = power
    
    # perform the fit
    res = fit_base(res)
    
    return res

def power(par, x):
    power = par[0]*x**par[1]
    return power

#########################################
# Vogel-Fulcher-Tamman viscosity fitter #
#########################################
def fit_VFT(res):
    # assign the reorg model
    res.f = VFT
    
    # perform the fit
    res = fit_base(res)
    
    # print the result
    varnames = [r'A', r'B', r'T0']
    print('--------------------------------------------------------------------------------')
    print('VFT Fit Results')
    print('--------------------------------------------------------------------------------')
    print('%7s %7s %7s' % tuple(varnames))
    for i in range(res.n_fit):
        print('%7.3f %7.3f %7.3f' % tuple(np.squeeze(res.fitpar)))
    print('--------------------------------------------------------------------------------')
    print('Fix:%4d%8d%8d'% tuple(res.fix))
    print('--------------------------------------------------------------------------------')
    
    return res

def VFT(par, x):
    par = np.squeeze(par)
    VFT = par[0]*np.exp(par[1]/(x-par[2]))
    return VFT

##########################
# lognormal decay fitter #
##########################
# fit a trace to the sum of a log-normal decay and stretched exponential
def fit_lognrm_decay(res):
    # assign the exponential model
    res.f = lognrm_decay
    
    # perform the fit
    res = fit_base(res)
    
    ###############################
    # calculate the integral time #
    ###############################
    res.tau_int = np.zeros(res.n_fit)
    tau_ln  = np.zeros(res.n_fit)
    tau_str = np.zeros(res.n_fit)
    
    # do the lognorm part
    a    = np.zeros(res.n_fit)
    b    = np.zeros(res.n_fit)
    par1 = np.zeros(res.n_fit)
    par2 = np.zeros(res.n_fit)
    par3 = np.zeros(res.n_fit)
    
    for i in range(res.n_fit):
        a[i] = np.log(2)/(res.fitpar[i,4]**2);
        b[i] = 2*res.fitpar[i,4]/res.fitpar[i,3]
        
        par1[i] = np.sqrt(np.pi/(4*a[i]))
        par2[i] = np.exp(1/(4*a[i]))/b[i]
        par3[i] = 1 + erf(1/(2*np.sqrt(a[i])))
        
        tau_ln[i] = par1[i]*par2[i]*par3[i]
        
    # do the str exp part
    for i in range(res.n_fit):
        tau_str[i] = (res.fitpar[i,5]/res.fitpar[i,6]*
                      gamma(1/res.fitpar[i,6]))
        
    # do the integral time
    for i in range(res.n_fit):
        res.tau_int[i] = (res.fitpar[i,2]*tau_ln[i] + 
                          (1 - res.fitpar[i,2])*tau_str[i])
        
    # print the result
    varnames = [r'h', r'bkg', r'a1', r'sigma', r'gamma',
                r'tau2', r'beta', r'tau_ln', r'tau_str', r'<tau>']
    print('--------------------------------------------------------------------------------')
    print('Lognorm + Str Fit Results')
    print('--------------------------------------------------------------------------------')
    print('%7s %7s %7s %7s %7s %7s %7s %7s %7s %7s' % tuple(varnames))
    for i in range(res.n_fit):
        print('%7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f' % 
              (*res.fitpar[i,:], tau_ln[i], tau_str[i], res.tau_int[i]))
    print('--------------------------------------------------------------------------------')
    print('Fix:%3d%8d%8d%8d%8d%8d%8d'% tuple(res.fix))
    print('--------------------------------------------------------------------------------')
    return res

def lognrm_decay(par, x):
    h     = par[0]
    bkg   = par[1]
    a1    = par[2]
    sigma = par[3]
    gamma = par[4]
    t2    = par[5]
    beta  = par[6]
    
    # build the lognorm
    decay = a1*np.exp(-np.log(2)/gamma**2*(np.log(1 + 2*x*gamma/sigma))**2)
    
    # build the stretched exp
    decay = decay + (1-a1)*np.exp(-(x/t2)**beta)
    
    # apply height and background
    decay = h*decay + bkg
    
    return decay

################################
# reorganization energy fitter #
################################
# fits reorganization energy vs. r
def fit_reorg(res):
    # assign the reorg model
    res.f = reorg_model
    
    # perform the fit
    res = fit_base(res)
    
    return res

def reorg_model(par, x):
    return par[0]*(2 - par[1]/x)

###################################
# fit an area-normalized gaussian #
###################################
# fits a guassian with unit area
def fit_gauss_nrm(res):
    # assign the model
    res.f = gauss_a_nrm
    
    # perform the fit
    res = fit_base(res)
    
    return res

def gauss_a_nrm(par, x):
    nrm = (par[0]*np.sqrt(2*np.pi))**-1
    gauss = nrm*np.exp(-(x-par[1])**2/(2*par[0]**2))
    return gauss

####################################
# fit two gaussians to a lineshape #
####################################
# fits a guassian with unit area
def fit_two_gauss(res):
    # assign the model
    res.f = two_gauss
    
    # perform the fit
    res = fit_base(res)
    
    return res

def gauss_h_nrm(par, x):
    gauss = np.exp(-(x-par[2])**2/(2*par[1]**2))
    gauss = par[0]*gauss/np.max(gauss)
    
    return gauss

def two_gauss(par, x):    
    fit = gauss_h_nrm(par[0:3], x) + gauss_h_nrm(par[3:], x)
    return fit

########################
# base fitting routine #
########################
def fit_base(res):
    # set up some stuff
    dum = np.shape(res.data)
    if len(dum) == 1:
        res.n_fit  = 1
        res.data   = np.reshape(res.data, [len(res.data), 1])
        if type(res.weight) != bool:
            res.weight = np.reshape(res.weight, [len(res.weight), 1])
    elif len(dum) == 2:
        res.n_fit   = dum[1]
        
    res.fitpar  = np.zeros([res.n_fit, len(res.fix)])
    res.fit     = np.zeros([len(res.x), res.n_fit])
    res.tau_int = np.zeros(res.n_fit)
    
    # encode parameters
    par_enc, lb_enc, ub_enc = encode_par(res)
    
    # do the least squares
    for i in range(res.n_fit):
        if type(res.weight) == bool:
            if type(lb_enc) == bool:
                result = least_squares(residuals, 
                                       par_enc,
                                       args=(res.f, 
                                             res.x, 
                                             res.data[:,i], 
                                             res.guess, 
                                             res.fix))
            else:
                result = least_squares(residuals, 
                                       par_enc,
                                       bounds=(lb_enc, 
                                               ub_enc),
                                       args=(res.f, 
                                             res.x, 
                                             res.data[:,i], 
                                             res.guess, 
                                             res.fix))
        else:
            result = least_squares(residuals, 
                                   par_enc,
                                   bounds=(lb_enc, 
                                           ub_enc),
                                   args=(res.f, 
                                         res.x, 
                                         res.data[:,i], 
                                         res.guess, 
                                         res.fix,
                                         res.weight[:,i]))
        
        # decode parameters
        res.fitpar[i,:] = decode_par(result.x, res.guess, res.fix)
        
        # calculate the fit result
        res.fit[:,i] = res.f(res.fitpar[i,:], res.x)
        
    # calculate the residuals
    res.resid = res.data - res.fit
    
    # hang the fit result
    res.result = result
    
    # if len(res.fitpar) == 1:
    #     res.fitpar = np.squeeze(res.fitpar)
    return res

#############
# residuals #
#############
def residuals(par_enc, f, t, data, guess, fix, weight=False):
    par = decode_par(par_enc, guess, fix)
    fit = f(par, t)
    if type(weight) == bool:
        return (data - fit)
    else:
        return (data - fit)*weight

###########################
# encode/decode functions #
###########################
def encode_par(res):
    par = res.guess[res.fix != 0]
    if type(res.lb) == bool:
        lb = False
        ub = False
    else:
        lb  = res.lb[res.fix != 0]
        ub  = res.ub[res.fix != 0]    
    return par, lb, ub

def decode_par(par_enc, guess, fix):
    j = 0
    par_dec = np.zeros(len(fix))
    for i in range(len(fix)):
        if fix[i]:
            par_dec[i] = par_enc[j]
            j = j + 1
        else:
            par_dec[i] = guess[i]
    return par_dec

###################
# Global analysis #
###################
def glob_anal(file, n_exp):
    import matplotlib.pyplot as plt
    import spectra as spc
    from matplotlib import cm
    from scipy.optimize import least_squares
    
    # load data
    t, wave, spec = spc.read_dat(file)
    n_t    = len(t)
     
    # set guess
    guess = np.logspace(np.log10(1e0), np.log10(10**(n_exp-1)), n_exp)
    
    # define the model for the fit
    def model(par, t):
        # calculate the time evolution matrix
        t_mat = np.zeros((n_t, n_exp))
        for i in range(len(par)):
            t_mat[:,i] = np.exp(-t/par[i])
        
        # do the left-hand matrix divison to get the amplitude matrix
        a_mat = np.linalg.lstsq(t_mat, spec.T, rcond=None)[0]
        
        # calculate the fits
        fit   = np.matmul(t_mat, a_mat).T
        return fit, a_mat, t_mat
    
    # define a residual generator to send to the fitter
    def resid_fun(par, f, t, data):
        fit = f(par, t)[0]
        resid = data - fit
        return np.reshape(resid, np.size(resid))
    
    # perform the fit
    result  = least_squares(resid_fun, guess, args=(model, t, spec))
    
    # extract results
    fit_par = result.x
    fit, a_mat, t_mat = model(fit_par, t)
    resid   = fit - spec
    chi_sq  = np.power(resid, 2).sum()*(1/(np.size(resid) - n_exp))
    
    ##################
    # calculate EADS #
    ##################
    # determine the B matrix
    B      = np.zeros((n_exp, n_exp));
    B[:,0] = 1             
    for i in np.arange(2, n_exp+1, 1, dtype=int):
        for j in np.arange(i, n_exp+1, 1, dtype=int):
            B[j-1,i-1] = 1
            for k in np.arange(1, j, 1, dtype=int):
                B[j-1, i-1] = ((1/fit_par[k-1] - 1/fit_par[j-1])/
                               (1/fit_par[k-1]))*B[j-1, i-1] 
    
    # apply the B matrix to the DADS
    EADS = np.zeros(np.shape(a_mat))
    for i in range(n_exp):
        for j in range(n_exp):
            EADS[i,:] = EADS[i,:] + a_mat[j,:]*B[j,i]
    
    ##################
    # spit some text #
    ##################
    print(r'##################################################################')
    print(r'Global Analysis Fitting Results of %s to %d exp:' % (file, n_exp))
    for i in range(n_exp):
        print(r'tau_%1d  = %3.1f ps' % (i, fit_par[i]))
    print(r'chi_sq = %3.2f' % chi_sq)
    print(r'##################################################################')
    
    ################
    # plot results #
    ################
    fig, ax = plt.subplots(nrows=2, ncols=3)
    ax = np.reshape(ax, np.size(ax))
    fig.set_size_inches([15.5,  8.5 ])
    fig.subplots_adjust(top=0.955,
                        bottom=0.08,
                        left=0.07,
                        right=0.965,
                        hspace=0.375,
                        wspace=0.31)
    
    # get the color scales for the spectra and fits
    if spec.max() > np.abs(spec.min()):
        vmin = -spec.max()
        vmax = spec.max()
    else:
        vmin = -np.abs(spec.min())
        vmax = np.abs(spec.min())
    
    # get the color scales for the residuals
    if resid.max() > np.abs(resid.min()):
        vmin_resid = -spec.max()
        vmax_resid = spec.max()
    else:
        vmin_resid = -np.abs(resid.min())
        vmax_resid = np.abs(resid.min())
    
    # plot the spectral surface
    ax[0].contourf(wave, t, spec.T,  levels=16, 
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # plot the fit surface
    ax[1].contourf(wave, t, fit.T,   levels=16,
                   cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    # plot the residual surface
    ax[2].contourf(wave, t, resid.T, levels=16, 
                   cmap='RdBu_r', vmin=vmin_resid/4, vmax=vmax_resid/4)
    
    # plot the DADS and EADS
    for i in range(n_exp):
        ax[3].plot(wave, a_mat[i,:], 
                   label=r'$\tau_%i = %.1f$ ps' % (i+1, fit_par[i]))
        ax[4].plot(wave, EADS[i,:],
                   label=r'$\tau_%i = %.1f$ ps' % (i+1, fit_par[i]))
        
    ax[3].legend(fontsize=10, loc='lower right')
    ax[4].legend(fontsize=10, loc='lower right')
    
    # plot some sample spectra and fits
    sel = np.linspace(0, n_t-10, 6, dtype=int)
    offset = np.flip(np.arange(0, len(sel), 1, dtype=int))*3
    spec_col = cm.get_cmap('rainbow', len(sel))
    for i in range(len(sel)):
        ax[5].plot(wave, spec[:,sel[i]] + offset[i],
                   'o', markerfacecolor='none', markeredgecolor=spec_col(i),
                   label= r'$t = %3.1f$ ps' % t[sel[i]])
        ax[5].plot(wave, fit[:,sel[i]] + offset[i], 'k')
    ax[5].legend(fontsize=10, ncol=2, loc='lower right')
    
    # settings for the surface plots
    for i in [0, 1, 2]:
        ax[i].set_yscale('log')
        ax[i].set_ylim([t[1], t[-1]])
        ax[i].set_xlabel(r'Wavelength / nm')
        ax[i].set_ylabel(r'Time / ps')
        ax[i].tick_params(direction='out', which='both')
    
    # axes titles
    ax[0].set_title(r'\textbf{Measurement}')
    ax[1].set_title(r'\textbf{Fit}')
    ax[2].set_title(r'\textbf{Residuals}')
    ax[3].set_title(r'\textbf{DADS}')
    ax[4].set_title(r'\textbf{EADS}')
    ax[5].set_title(r'\textbf{Sample Spectra}')
    
    # settings for DADS, EADS, and sample spectra
    for i in [3, 4, 5]:
        ax[i].set_xlabel('Wavelength / nm')
        ax[i].set_ylabel(r'$\Delta A$ / $10^{-3}$')






















































