#!/usr/bin/env python3
###############################################################################
# Chris Rumble
# 7/18/23
#
# A program for performing a convolute-and-compare fit of TCSPC anisotropy 
# data.
###############################################################################
import fitting as ft
import spectra as spc
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths

#############
# Load data #
#############
irf_range_t = [1, 1.75]
flr_range_t = [1, 24]

t   = spc.read_tcspc('./acn/prodan_acn_460nm_mag.asc')[:,0]
par = spc.read_tcspc('./acn/prodan_acn_460nm_par.asc')[:,1]
per = spc.read_tcspc('./acn/prodan_acn_460nm_per.asc')[:,1]
irf = spc.read_tcspc('./acn/irf.asc')[:,1]

# t   = spc.read_tcspc('./tol/prodan_tol_420nm_mag.asc')[:,0]
# par = spc.read_tcspc('./tol/prodan_tol_420nm_par.asc')[:,1]
# per = spc.read_tcspc('./tol/prodan_tol_420nm_per.asc')[:,1]
# irf = spc.read_tcspc('./tol/irf.asc')[:,1]

# t   = spc.read_tcspc('./mecl/prodan_mecl_440nm_mag.asc')[:,0]
# par = spc.read_tcspc('./mecl/prodan_mecl_440nm_par.asc')[:,1]
# per = spc.read_tcspc('./mecl/prodan_mecl_440nm_per.asc')[:,1]
# irf = spc.read_tcspc('./mecl/irf.asc')[:,1]

irf_range_t = spc.closest(irf_range_t, t)
irf         = irf[irf_range_t[0]:irf_range_t[1]]
t_irf       = t[irf_range_t[0]:irf_range_t[1]]

flr_range_t = spc.closest(flr_range_t, t)
par         = par[flr_range_t[0]:flr_range_t[1]]
per         = per[flr_range_t[0]:flr_range_t[1]]
t           = t[flr_range_t[0]:flr_range_t[1]]


###################
# prepare the fit #
###################
res      = ft.FitData()
res.irf  = irf
res.data = np.vstack([par, per]).T
res.x    = t
res.lb    = np.zeros(17)
res.guess = np.zeros(17)
res.ub    = np.zeros(17)
res.fix   = np.zeros(17)

res.lb[0]  =    0; res.guess[0]  = 1;   res.ub[0]  = 1e2; res.fix[0]  = 1 # bkg
res.lb[1]  = -1e2; res.guess[1]  = 0;   res.ub[1]  = 1e2; res.fix[1]  = 1 # ts_par
res.lb[2]  = -1e2; res.guess[2]  = 0;   res.ub[2]  = 1e2; res.fix[2]  = 1 # ts_per
res.lb[3]  =    0; res.guess[3]  = 1;   res.ub[3]  = 2;   res.fix[3]  = 1 # G
res.lb[4]  = -1e3; res.guess[4]  = 5e3; res.ub[4]  = 1e7; res.fix[4]  = 1 # a1
res.lb[5]  = 1e-2; res.guess[5]  = 0.02;res.ub[5]  = 1e7; res.fix[5]  = 1 # t1
res.lb[6]  = -1e7; res.guess[6]  = 3e3; res.ub[6]  = 1e7; res.fix[6]  = 1 # a2
res.lb[7]  = 1e-2; res.guess[7]  = 1;   res.ub[7]  = 1e7; res.fix[7]  = 1 # t2
res.lb[8]  = -1e7; res.guess[8]  = 1e4; res.ub[8]  = 1e7; res.fix[8]  = 1 # a3
res.lb[9]  = 1e-2; res.guess[9]  = 2;   res.ub[9] = 1e7;  res.fix[9]  = 1 # t3
res.lb[10] =    0; res.guess[10] = 0.1; res.ub[10] = 0.4; res.fix[10] = 1 # ra1
res.lb[11] = 1e-2; res.guess[11] = 0.01;res.ub[11] = 1e7; res.fix[11] = 1 # rt1
res.lb[12] =    0; res.guess[12] = 0;   res.ub[12] = 0.4; res.fix[12] = 0 # ra2
res.lb[13] = 1e-2; res.guess[13] = 0.1; res.ub[13] = 1e0; res.fix[13] = 0 # rt2
res.lb[14] =   -1; res.guess[14] = 0;   res.ub[14] = 0.4; res.fix[14] = 0 # ra3
res.lb[15] = 1e-2; res.guess[15] = 0.2; res.ub[15] = 1e7; res.fix[15] = 0 # rt3

#################################
# make the fitter and residuals #
#################################
def conv_comp_anis_2fit(par, x, irf, data):
    bkg  = par[0]
    tsft = [par[1], par[2]]
    G    = par[3]
    a1   = par[4]
    t1   = par[5]
    a2   = par[6]
    t2   = par[7]
    a3   = par[8]
    t3   = par[9]
    ra1  = par[10]
    rt1  = par[11]
    ra2  = par[12]
    rt2  = par[13]
    ra3  = par[14]
    rt3  = par[15]
    
    # set x to start at t=0
    x = x - x[0]
    
    # make the anisotropy
    r = ra1*np.exp(-x/rt1) + ra2*np.exp(-x/rt2) + ra3*np.exp(-x/rt3)
    
    # create the decays (MA, VV, VH)
    fit      = np.zeros([len(x), 2])
    decay    = a1*np.exp(-x/t1) + a2*np.exp(-x/t2) + a3*np.exp(-x/t3)
    fit[:,0] = decay*(1 + 2*r)
    fit[:,1] = G*decay*(1 - r)
    
    # area normalize the IRF
    irf = irf/np.trapz(irf)
    
    # do convolution and time-shift for each decay
    for i in range(2):
        fit[:,i] = np.convolve(irf, fit[:,i], mode='full')[:len(x)]
        fit[:,i] = np.interp(x - tsft[i], x, fit[:,i]) 
        fit[:,i] = fit[:,i] + bkg
            
    # calculate residuals
    resid = fit - data
    data[data == 0] = 1
    resid = resid/np.sqrt(np.abs(data))
   
    return fit, resid

def residuals(par_enc, guess, fix, x, irf, data):
    par        = ft.decode_par(par_enc, guess, fix)
    fit, resid = conv_comp_anis_2fit(par, x, irf, data)
    return np.concatenate(resid)

##############
# do the fit #
##############
par_enc, lb_enc, ub_enc = ft.encode_par(res)
result = least_squares(residuals, 
                        par_enc, 
                        bounds=(lb_enc, ub_enc), 
                        args=(res.guess,
                              res.fix,
                              res.x,
                              res.irf,
                              res.data))

# decode parameters
res.fitpar = ft.decode_par(result.x, res.guess, res.fix)
        
# calculate the fit result
res.fit, res.resid = conv_comp_anis_2fit(res.fitpar, res.x, res.irf, res.data)

# calculate chi_sq
res.chi_sq = np.zeros(2)
for i in range(2):
    res.chi_sq[i] = np.sum(np.power(res.resid[:,i], 2))/(len(res.x)-len(res.fix[res.fix==1]))

# calculate some anisotropy stuff
r0    = res.fitpar[10] + res.fitpar[12] + res.fitpar[14]
r_avg = (res.fitpar[10]*res.fitpar[11] +
          res.fitpar[12]*res.fitpar[13] + 
          res.fitpar[14]*res.fitpar[15])
r_avg = r_avg/(res.fitpar[10] + res.fitpar[12] + res.fitpar[14])

###############
# Plot result #
###############
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True,
                        gridspec_kw={'height_ratios':[1,6],
                                    'hspace': 0.05,
                                    'top':    0.975,
                                    'right':  0.95,
                                    'bottom': 0.1})
fig.set_size_inches([6, 5.5])

# residuals
colors = ['b', 'g']
for i in range(2):
    ax[0].plot(res.x, res.resid[:,i], '.', markersize=3, color=colors[i])
ax[0].plot([res.x[0], res.x[-1]], [0,0], 'k', alpha=0.5)
ax[0].set_ylabel('Residuals')

if np.abs(res.resid).max() > np.abs(res.resid).min():
    ax[0].set_ylim([-1.1*np.abs(res.resid).max(),
                      1.1*np.abs(res.resid).max()])
else:
    ax[0].set_ylim([-1.1*np.abs(res.resid).min(),
                      1.1*np.abs(res.resid).min()])

# data and fit
labels = ['VV', 'VH', 'VV Fit', 'VH Fit']
ax[1].plot(t_irf, irf, color='gray', label='IRF')
for i in range(2):
    ax[1].plot(res.x, res.data[:,i], '.', color=colors[i], markersize=3, label=labels[i])
    ax[1].plot(res.x, res.fit[:,i], color='r', label=labels[i+2], linewidth=1)
ax[1].set_yscale('log')
ax[1].set_xlim([res.x[0], res.x[-1]])
ax[1].set_xlabel('Time / ns')
ax[1].set_ylabel('Counts')
ax[1].legend(fontsize=10, ncol=2, loc='lower right')

##################
# spit some text #
##################
names = ['bkg', 'ts_par', 'ts_per', 'G',
          'a1',   'tau1',     'a2',   'tau2',  'a3', 'tau3',
        'r_a1', 'r_tau1',   'r_a2', 'r_tau2', 'r_a3', 'r_tau3']
tech  = [0, 1, 2, 3]
pop   = [4, 5, 6, 7, 8, 9]
anis  = [10, 11, 12, 13, 14, 15]

print('##########################')
print('#     Anisfit result     #')
print('##########################')
print('Technical:\t\t\t Population:\t   Anisotropy:')
for i in range(len(tech)):
    print('{:>8} = {:6.3f} {:>8} = {:8.3f} {:>8} = {:6.3f}'.format(names[tech[i]], res.fitpar[tech[i]],
                                                                    names[pop[i]], res.fitpar[pop[i]],
                                                                    names[anis[i]], res.fitpar[anis[i]]))
print('\t\t\t\t  {:>8} = {:8.3f} {:>8} = {:6.3f}'.format(names[pop[4]], res.fitpar[pop[4]],
                                                     names[anis[4]], res.fitpar[anis[4]]))
print('\t\t\t\t  {:>8} = {:8.3f} {:>8} = {:6.3f}'.format(names[pop[5]], res.fitpar[pop[5]],
                                                     names[anis[5]], res.fitpar[anis[5]]))
print()
print('{:>8} = {:6.3f} {:>8} = {:8.3f}'.format('chi2_par', res.chi_sq[0], 'r0', r0))
print('{:>8} = {:6.3f} {:>8} = {:8.3f}'.format('chi2_per', res.chi_sq[1], '<r>', r_avg))
