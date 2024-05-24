###############################################################################
# Chris Rumble
# 4/18/23
#
# Subroutines for performing the convolute-and-compare fit using the GUI.
###############################################################################
import spectra as spc
import fitting as ft
import numpy as np
from scipy.optimize import least_squares
from tkinter import messagebox

#############################################################
# residual calculator, will call specific fitting functions #
#############################################################
def residuals(par_enc, guess, fix, irf, data, fit_function, tstep):
    # unpack model parameters
    par = ft.decode_par(par_enc, guess, fix)
    
    # send to the fit generator    
    fit, resid = conv_comp(par, irf, data, fit_function, tstep)
    
    return resid

###################################
# convolute-and-compare algorithm #
###################################
def conv_comp(par, irf, x, data, fit_function):
    # set x to start at 0
    x = x - x[0]

    # build the fit
    if   fit_function == '1-exp':
        h    = par[0]
        bkg  = par[1]
        tsft = par[2]
        tau1 = par[3]
        
        fit = np.exp(-x/tau1)
        
    elif fit_function == '2-exp':
        h    = par[0]
        bkg  = par[1]
        tsft = par[2]
        a1   = par[3]
        tau1 = par[4]
        tau2 = par[5]

        fit = a1*np.exp(-x/tau1) + (1-a1)*np.exp(-x/tau2)

    elif fit_function == '3-exp':
        h    = par[0]
        bkg  = par[1]
        tsft = par[2]
        a1   = par[3]
        tau1 = par[4]
        a2   = par[5]
        tau2 = par[6]
        a3   = par[7]
        tau3 = par[8]

        norm = a1 + a2 + a3
        a1 = a1/norm
        a2 = a2/norm
        a3 = a3/norm
        par[3] = a1
        par[5] = a2
        par[7] = a3

        fit = (a1*np.exp(-x/tau1) +
               a2*np.exp(-x/tau2) +
               a3*np.exp(-x/tau3))

    elif fit_function == '1-str':
        h     = par[0]
        bkg   = par[1]
        tsft  = par[2]
        tau1  = par[3]
        beta1 = par[4]

        fit = np.exp(-(x/tau1)**beta1)
    
    elif fit_function == '2-str':
        h     = par[0]
        bkg   = par[1]
        tsft  = par[2]
        a1    = par[3]
        tau1  = par[4]
        beta1 = par[5]
        tau2  = par[6]
        beta2 = par[7]

        fit = np.exp(-(x/tau1)**beta1) + np.exp(-(x/tau2)**beta2)

    # convolute with the irf
    fit = np.convolve(irf, fit, mode='full')
    fit = fit[:len(data)]
    
    # set height and bkg
    fit = fit/fit.max()*h + bkg
    
    # apply the time-shift
    fit = np.interp(x - tsft, x, fit) 

    # calculate residuals
    resid = fit - data  
    data[data == 0] = 1
    resid = resid/np.sqrt(np.abs(data))
    
    return fit, resid


#########################
# main fitting function #
#########################
def fit_routine(res):
    # load in the data files
    try:
        res.irf_full  = spc.read_tcspc(res.irf_file)
    except:
        messagebox.showinfo('Input Error', 
                            'IRF file does not exist.')
        return
    try:    
        res.data_full = spc.read_tcspc(res.fl_file)
    except:
        messagebox.showinfo('Input Error', 
                            'Fluorescence file does not exist.')
        return

    # ensure the IRF and fluorescence have the same time axes
    if np.array_equal(res.irf_full[:,0], res.data_full[:,0]):
        res.x_full    = res.data_full[:,0]
        res.data_full = res.data_full[:,1]
        res.irf_full  = res.irf_full[:,1]
    else:
        messagebox.showinfo('Input Error', 
                            'Fluorescence and IRF time axes do not match.')
        return

    # set the range to use for the fitting
    res.x_irf = res.x_full[np.arange(res.irf_range[0], 
                                     res.irf_range[1], 
                                     1,
                                     dtype=int)]
    res.irf   = res.irf_full[np.arange(res.irf_range[0], 
                                       res.irf_range[1],
                                       1,
                                       dtype=int)]
    res.x     = res.x_full[np.arange(res.fl_range[0], 
                                     res.fl_range[1], 
                                     1,
                                     dtype=int)]
    res.data  = res.data_full[np.arange(res.fl_range[0], 
                                     res.fl_range[1], 
                                     1,
                                     dtype=int)]
    
    # calculate the timestep
    res.tstep = res.x[1]-res.x[0]
    
    ######################################
    # send the information to the fitter #
    ######################################
    par_enc, lb_enc, ub_enc = ft.encode_par(res)
    
    result = least_squares(residuals, 
                            par_enc, 
                            bounds=(lb_enc, ub_enc), 
                            args=(res.guess,
                                  res.fix,
                                  res.irf,
                                  res.x,
                                  res.data,
                                  res.fit_function))

    
    # decode parameters
    res.fitpar = ft.decode_par(result.x, res.guess, res.fix)
            
    # calculate the fit result
    res.fit, res.resid = conv_comp(res.fitpar, res.irf, res.x, res.data, res.fit_function)

    # calculate chi_sq
    res.chi_sq = np.sum(np.power(res.resid, 2))/(len(res.x)-len(res.fix[res.fix==1]))
    
    return res



