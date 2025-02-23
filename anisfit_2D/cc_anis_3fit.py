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
# mag_raw = spc.read_dac('./x00/500ps/c153meoh_500psjc_magic.dac')
# par_raw = spc.read_dac('./x00/500ps/c153meoh_500psjc_para.dac')
# per_raw = spc.read_dac('./x00/500ps/c153meoh_500psjc_perp.dac')
# irf_raw = spc.read_dac('./x00/500ps/ccwater_500psjc_magic2.dac')
fl_range    = [510, 500]
irf_range   = [430, 420]
irf_range_t = [0, 0.3]


mv fmag_raw = spc.read_dac('./x00/2ns/c153_x00glymeoh_2nsjc_magic.dac')
par_raw = spc.read_dac('./x00/2ns/c153_x00glymeoh_2nsjc_para.dac')
per_raw = spc.read_dac('./x00/2ns/c153_x00glymeoh_2nsjc_perp.dac')
irf_raw = spc.read_dac('./x00/2ns/ccwater_2nsjc_magic.dac')
t       = mag_raw[0]
wln     = mag_raw[1]

fl_range = spc.closest(fl_range, wln)

mag = np.sum(mag_raw[2][fl_range[0]:fl_range[1], :], axis=0)
par = np.sum(par_raw[2][fl_range[0]:fl_range[1], :], axis=0)
per = np.sum(per_raw[2][fl_range[0]:fl_range[1], :], axis=0)

irf_range   = spc.closest(irf_range, irf_raw[1])
irf_range_t = spc.closest(irf_range_t, irf_raw[0])
irf         = np.sum(irf_raw[2][irf_range[0]:irf_range[1], 
                                irf_range_t[0]:irf_range_t[1]], axis=0)
t_irf       = irf_raw[0][irf_range_t[0]:irf_range_t[1]]

#############
# Load data #
#############
res      = ft.FitData()
res.irf  = irf
res.data = np.vstack([mag, par, per]).T
res.x    = t


###################
# prepare the fit #
###################
res.lb    = np.zeros(18)
res.guess = np.zeros(18)
res.ub    = np.zeros(18)
res.fix   = np.zeros(18)

res.lb[0]  =    0; res.guess[0]  = 1;   res.ub[0]  = 1e2; res.fix[0]  = 1 # bkg
res.lb[1]  = -1e2; res.guess[1]  = 0;   res.ub[1]  = 1e2; res.fix[1]  = 1 # ts_mag
res.lb[2]  = -1e2; res.guess[2]  = 0;   res.ub[2]  = 1e2; res.fix[2]  = 1 # ts_par
res.lb[3]  = -1e2; res.guess[3]  = 0;   res.ub[3]  = 1e2; res.fix[3]  = 1 # ts_per
res.lb[4]  =    0; res.guess[4]  = 1;   res.ub[4]  = 2;   res.fix[4]  = 1 # G
res.lb[5]  =    0; res.guess[5]  = 1;   res.ub[5]  = 2;   res.fix[5]  = 1 # A
res.lb[6]  = -1e3; res.guess[6]  = 5e1; res.ub[6]  = 1e7; res.fix[6]  = 1 # a1
res.lb[7]  = 1e-2; res.guess[7]  = 0.2; res.ub[7]  = 1e7; res.fix[7]  = 1 # t1
res.lb[8]  = -1e7; res.guess[8]  = 5e2; res.ub[8]  = 1e7; res.fix[8]  = 1 # a2
res.lb[9]  = 1e-2; res.guess[9]  = 5;   res.ub[9]  = 1e7; res.fix[9]  = 1 # t2
res.lb[10] = -1e7; res.guess[10] = 0;   res.ub[10] = 1e7; res.fix[10] = 0 # a3
res.lb[11] = 1e-2; res.guess[11] = 10;  res.ub[11] = 1e7; res.fix[11] = 0 # t3
res.lb[12] =    0; res.guess[12] = 0.2; res.ub[12] = 0.4; res.fix[12] = 1 # ra1
res.lb[13] = 1e-2; res.guess[13] = 0.1; res.ub[13] = 1e7; res.fix[13] = 1 # rt1
res.lb[14] =    0; res.guess[14] = 0  ; res.ub[14] = 0.4; res.fix[14] = 0 # ra2
res.lb[15] = 1e-2; res.guess[15] = 0.2; res.ub[15] = 1e7; res.fix[15] = 0 # rt2
res.lb[16] =   -1; res.guess[16] = 0;   res.ub[16] = 0.4; res.fix[16] = 0 # ra3
res.lb[17] = 1e-2; res.guess[17] = 1;   res.ub[17] = 1e7; res.fix[17] = 0 # rt3

#################################
# make the fitter and residuals #
#################################
def conv_comp_anis_3fit(par, x, irf, data):
    bkg  = par[0]
    tsft = [par[1], par[2], par[3]]
    G    = par[4]
    A    = par[5]
    a1   = par[6]
    t1   = par[7]
    a2   = par[8]
    t2   = par[9]
    a3   = par[10]
    t3   = par[11]
    ra1  = par[12]
    rt1  = par[13]
    ra2  = par[14]
    rt2  = par[15]
    ra3  = par[16]
    rt3  = par[17]
    
    # set x to start at t=0
    x = x - x[0]
    
    # make the anisotropy
    r = ra1*np.exp(-x/rt1) + ra2*np.exp(-x/rt2) + ra3*np.exp(-x/rt3)
    
    # create the decays (MA, VV, VH)
    fit      = np.zeros([len(x), 3])
    fit[:,0] = a1*np.exp(-x/t1) + a2*np.exp(-x/t2) + a3*np.exp(-x/t3)
    fit[:,1] = A*fit[:,0]*(1 + 2*r)
    fit[:,2] = A*G*fit[:,0]*(1 - r)
    
    # area normalize the IRF
    irf = irf/np.trapz(irf)
    
    # do convolution and time-shift for each decay
    for i in range(3):
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
    fit, resid = conv_comp_anis_3fit(par, x, irf, data)
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
res.fit, res.resid = conv_comp_anis_3fit(res.fitpar, res.x, res.irf, res.data)

# calculate chi_sq
res.chi_sq = np.zeros(3)
for i in range(3):
    res.chi_sq[i] = np.sum(np.power(res.resid[:,i], 2))/(len(res.x)-len(res.fix[res.fix==1]))

# calculate some anisotropy stuff
r0    = res.fitpar[12] + res.fitpar[14] + res.fitpar[16]
r_avg = (res.fitpar[12]*res.fitpar[13] +
         res.fitpar[14]*res.fitpar[15] + 
         res.fitpar[16]*res.fitpar[17])
r_avg = r_avg/(res.fitpar[12] + res.fitpar[14] + res.fitpar[16])

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
ax[0].plot(res.x, res.resid, '.', markersize=3)
ax[0].plot([res.x[0], res.x[-1]], [0,0], 'k', alpha=0.5)
ax[0].set_ylabel('Residuals')

if np.abs(res.resid).max() > np.abs(res.resid).min():
    ax[0].set_ylim([-1.1*np.abs(res.resid).max(),
                      1.1*np.abs(res.resid).max()])
else:
    ax[0].set_ylim([-1.1*np.abs(res.resid).min(),
                      1.1*np.abs(res.resid).min()])

# data and fit
colors = ['b', 'r', 'g']
labels = ['MA', 'VV', 'VH', 'MA Fit', 'VV Fit', 'VH Fit']
ax[1].plot(t_irf, irf, color='gray', label='IRF')
for i in range(3):
    ax[1].plot(res.x, res.data[:,i], '.', color=colors[i], markersize=3, label=labels[i])
    ax[1].plot(res.x, res.fit[:,i], color='m', label=labels[i+3])
ax[1].set_yscale('log')
ax[1].set_xlim([res.x[0], res.x[-1]])
ax[1].set_xlabel('Time')
ax[1].set_ylabel('Counts')
ax[1].legend(fontsize=10, ncol=2, loc='lower right')

##################
# spit some text #
##################
names = ['bkg', 'ts_mag', 'ts_par', 'ts_per',   'G',    'A',
          'a1',   'tau1',     'a2',   'tau2',  'a3', 'tau3',
         'r_a1',  'r_tau1',    'r_a2',  'r_tau2', 'r_a3', 'r_tau3']
tech  = [0, 1, 2, 3, 4, 5]
pop   = [6, 7, 8, 9, 10, 11]
anis  = [12, 13, 14, 15, 16, 17]

print('##########################')
print('#     Anisfit result     #')
print('##########################')
print('Technical:\t\t\t Population:\t   Anisotropy:')
for i in range(len(pop)):
    print('{:>8} = {:6.3f} {:>8} = {:8.3f} {:>8} = {:6.3f}'.format(names[tech[i]], res.fitpar[tech[i]],
                                                                   names[pop[i]], res.fitpar[pop[i]],
                                                                   names[anis[i]], res.fitpar[anis[i]]))
print()
print('{:>8} = {:6.3f} {:>8} = {:8.3f}'.format('chi2_mag', res.chi_sq[0], 'r0', r0))
print('{:>8} = {:6.3f} {:>8} = {:8.3f}'.format('chi2_par', res.chi_sq[1], '<r>', r_avg))
print('{:>8} = {:6.3f}'.format('chi2_per', res.chi_sq[2]))
