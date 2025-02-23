#!/usr/bin/env python3
###############################################################################
# Chris Rumble
# 6/14/23
#
# A GUI interface for the convolute-and-compare fitter that uses a 2D data file
# as input.
###############################################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog
from scipy.optimize import least_squares
import numpy as np
import fitting as ft
import spectra as spc
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)

###############################################################################
#                           Window Preparation                                #
###############################################################################
# create the main window
root = tk.Tk()
root.title('2D Anisotropy Fitter')
root.geometry('500x500')

# frames go inside windows
frm_files   = ttk.Frame(root, padding=5)
frm_ranges  = ttk.Frame(root, padding=5)
frm_pars    = ttk.Frame(root, padding=5)

# set up the grid
frm_files.grid()
frm_ranges.grid()
frm_pars.grid()

# these will hold the parameter widgets
lb    = list()
guess = list()
ub    = list()
fix   = list()

# we need some names for the parameters
exp1_names = ['bkg:',  'ts_mag:', 'ts_par:', 'ts_perp:',  'a1:',  'tau1:']
exp2_names = ['bkg:',  'ts_mag:', 'ts_par:', 'ts_perp:',  'a1:',  'tau1:', 'a2:', 'tau2:']
exp3_names = ['bkg:',  'ts_mag:', 'ts_par:', 'ts_perp:',  'a1:',  'tau1:', 'a2:', 'tau2:', 'a3:', 'tau3:']
str1_names = ['bkg:',  'ts_mag:', 'ts_par:', 'ts_perp:',  'a1:',  'tau1:', 'beta1:']
str2_names = ['bkg:',  'ts_mag:', 'ts_par:', 'ts_perp:',  'a1:',  'tau1:', 'beta1:', 'a1', 'tau2:', 'beta2:']
aexp_names = ['ra1:',   'rtau1:',    'ra2:',   'rtau2:', 'ra3:', 'rtau3:']

###############################################################################
#                       GUI and Fitting Functions                             #
###############################################################################
############################################
# this will generate the parameter widgets #
############################################
def change_model(fit_function):
    # first clear the parameter frame widgets
    for widget in frm_pars.winfo_children():
        widget.destroy()

    # fit parameter headers
    ttk.Label(frm_pars, text='lb').grid(column=1, row=1)
    ttk.Label(frm_pars, text='guess').grid(column=2, row=1)
    ttk.Label(frm_pars, text='ub').grid(column=3, row=1)
    ttk.Label(frm_pars, text='fix').grid(column=4, row=1)

    # identify the model parameters and their defaults
    if   fit_function == '1-exp':
        par_names = exp1_names
        ini_ub    = np.asarray([ 1e2,  1, 1e4,  1e5])
        ini_guess = np.asarray([   0,  0, 5e3,    1])
        ini_lb    = np.asarray([-1e2, -1, 1e0, 1e-3])
    elif fit_function == '2-exp':
        par_names = exp2_names 
        ini_ub    = np.asarray([ 1e2,  1e0,  1e4,  1e2,  1e4,  1e2])
        ini_guess = np.asarray([   0,    0,  5e2,  0.5,  1e2,    4])
        ini_lb    = np.asarray([-1e2, -1e0, -1e4, 1e-3, -1e4, 1e-3])
    elif fit_function == '3-exp':
        par_names = exp3_names
        ini_ub    = np.asarray([ 1e2,  1e0,  1e4,  1e2,   1e4,  1e2,  1e4,  1e2])
        ini_guess = np.asarray([   0,    0,  5e2,  0.1,   1e2,  0.5,  1e2,    4])
        ini_lb    = np.asarray([-1e2, -1e0, -1e4, 1e-3,  -1e4, 1e-3, 1e-4, 1e-3])
    elif fit_function == '1-str':
        par_names = str1_names
        ini_ub    = np.asarray([ 1e2,  1, 1e5,  1e5,   1])
        ini_guess = np.asarray([   0,  0, 5e3,    1, 0.7])
        ini_lb    = np.asarray([-1e2, -1,   1, 1e-4,   0])
    elif fit_function == '2-str':
        par_names = str2_names
                               # bkg, tsft,   a1, tau1, beta1, a2, tau2, beta2
        ini_ub    = np.asarray([ 1e2,    1,  1e5,  1e2,    1,  1e5,   1e2,   1])
        ini_guess = np.asarray([   0,    0,  1e3,  0.1,    1,  5e2,     2, 0.7])
        ini_lb    = np.asarray([-1e2,   -1, -1e5, 1e-3, 1e-4, -1e5,  1e-3,   0])
    n_par = len(par_names)

    # build the parameter widgets
    lb.clear()
    guess.clear()
    ub.clear()
    fix.clear()
    for i in range(n_par):
        # create the widget
        ttk.Label(frm_pars, text=par_names[i]).grid(column=0, row=2+i, pady=2)
        lb.append(ttk.Entry(frm_pars))
        guess.append(ttk.Entry(frm_pars))
        ub.append(ttk.Entry(frm_pars))
        fix.append(ttk.Checkbutton(frm_pars))

        # initialize
        lb[-1].insert(tk.END, ini_lb[i])
        guess[-1].insert(tk.END, ini_guess[i])
        ub[-1].insert(tk.END, ini_ub[i])

        # place them
        lb[i].grid(column=1, row=2+i, pady=2, padx=5)
        guess[i].grid(column=2, row=2+i, pady=2, padx=5)
        ub[i].grid(column=3, row=2+i, pady=2, padx=5)
        fix[i].grid(column=4, row=2+i, pady=2, padx=5)

#############################################################
# residual calculator, will call specific fitting functions #
#############################################################
def residuals(par_enc, guess, fix, x, x_irf, irf, data, fit_function):
    # unpack model parameters
    par = ft.decode_par(par_enc, guess, fix)

    # send to the fit generator    
    fit, resid = conv_comp(par, x, x_irf, irf, data, fit_function)

    return resid

###################################
# convolute-and-compare algorithm #
###################################
def conv_comp(par, x, x_irf, irf, data, fit_function):
    # redefine x to start at 0
    x = x - x[0]

    # build the fit
    if   fit_function == '1-exp':
        bkg  = par[0]
        tsft = par[1]
        a1   = par[2]
        tau1 = par[3]

        fit = a1*np.exp(-x/tau1)

    elif fit_function == '2-exp':
        bkg  = par[0]
        tsft = par[1]
        a1   = par[2]
        tau1 = par[3]
        a2   = par[4]
        tau2 = par[5]

        fit = a1*np.exp(-x/tau1) + a2*np.exp(-x/tau2)

    elif fit_function == '3-exp':
        bkg  = par[0]
        tsft = par[1]
        a1   = par[2]
        tau1 = par[3]
        a2   = par[4]
        tau2 = par[5]
        a3   = par[6]
        tau3 = par[7]

        fit = a1*np.exp(-x/tau1) + a2*np.exp(-x/tau2) + a3*np.exp(-x/tau3)

    elif fit_function == '1-str':
        bkg   = par[0]
        tsft  = par[1]
        a1    = par[2]
        tau1  = par[3]
        beta1 = par[4]

        fit = np.exp(-(x/tau1)**beta1)

    elif fit_function == '2-str':
        bkg   = par[0]
        tsft  = par[1]
        a1    = par[2]
        tau1  = par[3]
        beta1 = par[4]
        a2    = par[5]
        tau2  = par[6]
        beta2 = par[7]

        fit = a1*np.exp(-(x/tau1)**beta1) + (1-a1)*np.exp(-(x/tau2)**beta2)

    # convolute with the irf
    irf = irf/np.trapz(irf)
    fit = np.convolve(irf, fit, mode='full')
    fit = fit[:len(data)]

    # set height and bkg
    fit = fit + bkg

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
    # send the information to the fitter
    par_enc, lb_enc, ub_enc = ft.encode_par(res)

    result = least_squares(residuals,
                            par_enc,
                            bounds=(lb_enc, ub_enc),
                            args=(res.guess,
                                  res.fix,
                                  res.x,
                                  res.x_irf,
                                  res.irf,
                                  res.data,
                                  res.fit_function))

    # decode parameters
    res.fitpar = ft.decode_par(result.x, res.guess, res.fix)

    # calculate the fit result
    res.fit, res.resid = conv_comp(res.fitpar, res.x, res.x_irf, res.irf, res.data, res.fit_function)

    print(result)

    # calculate chi_sq
    res.chi_sq = np.sum(np.power(res.resid, 2))/(len(res.x)-len(res.fix[res.fix==1]))

    return res

###############################################################################
#                           Main Data Class                                   #
###############################################################################
class raw_spectra:
    def __init__(self):
        # file names
        self.fl_filename  = None
        self.irf_filename = None

        # raw data
        self.t_fl_raw    = None
        self.wln_fl_raw  = None
        self.fl_raw      = None
        self.t_irf_raw   = None
        self.wln_irf_raw = None
        self.irf_raw     = None

        # single trace data
        self.fl_range  = np.zeros([2,2])
        self.irf_range = np.zeros([2,2]) 
        self.t_fl      = None
        self.fl        = None
        self.t_irf     = None
        self.irf       = None

    def load_files(self):
        # check for strings in both insert boxes

        # get the file names
        self.fl_filename  = file_widgets[0].get()
        self.irf_filename = file_widgets[1].get()

        # load the files into memory
        try:
            if self.fl_filename.split('.')[-1] == 'dac':
                self.t_fl_raw, self.wln_fl_raw, self.fl_raw = spc.read_dac(self.fl_filename)
            elif self.fl_filename.split('.')[-1] == 'dat':
                self.t_fl_raw, self.wln_fl_raw, self.fl_raw = spc.read_dat(self.fl_filename)
        except:
            messagebox.showinfo('Error', 'Could not load fluorescence file.')
            return
        try:
            if self.irf_filename.split('.')[-1] == 'dac':
                self.t_irf_raw, self.wln_irf_raw, self.irf_raw = spc.read_dac(self.irf_filename)
            elif self.irf_filename.split('.')[-1] == 'dat':
                self.t_irf_raw, self.wln_irf_raw, self.irf_raw = spc.read_dat(self.irf_filename)
        except:
            messagebox.showinfo('Error', 'Could not load IRF file.')
            return

        messagebox.showinfo('Success', 'Data files successfully loaded.')

    def get_mag_filename(self):
        filename = filedialog.askopenfilename()
        file_widgets[0].delete(0, tk.END)
        file_widgets[0].insert(tk.END, filename)

    def get_irf_filename(self):
        filename = filedialog.askopenfilename()
        file_widgets[1].delete(0, tk.END)
        file_widgets[1].insert(tk.END, filename)

    def plot_2D(self):
        # take the two 2D spectra and plot them nicely
        data_window = tk.Toplevel()
        data_window.title(r'Data and IRF')
        data_window.geometry('600x750')

        fig = Figure()
        ax  = fig.subplots(nrows=2, ncols=1)
        fig.subplots_adjust(top=0.975, bottom=0.075, right=0.95, hspace=0.225)
        ax[0].pcolor(self.t_fl_raw, self.wln_fl_raw, self.fl_raw, label='Fluorescence')
        ax[1].pcolor(self.t_irf_raw, self.wln_irf_raw, self.irf_raw, label='IRF')

        for i in range(2):
            ax[i].set_xlabel('Time / ns')
            ax[i].set_ylabel('Wavelength / nm')
            ax[i].legend(fontsize=10, loc='upper right')
        
        # hocus pocus required to draw the figure with toolbar
        canvas = FigureCanvasTkAgg(fig, master=data_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, data_window).update()
        canvas.get_tk_widget().pack()

    def trim(self):
        # single trace data
        self.fl_range  = np.zeros([2,2])
        self.irf_range = np.zeros([2,2])
        self.t_fl      = None
        self.fl        = None
        self.t_irf     = None
        self.irf       = None

        # extract ranges from GUI
        try:
            self.fl_range[0,:] = np.asarray([float(range_widgets[0].get()),
                                             float(range_widgets[1].get())])
        except:
            messagebox.showinfo('Input Error', 'Could not process Fl. Wave. Range')
            return

        try:
            self.fl_range[1,:] = np.asarray([float(range_widgets[2].get()),
                                             float(range_widgets[3].get())])
        except:
            messagebox.showinfo('Input Error', 'Could not process Fl. Time Range')
            return

        try:
            self.irf_range[0,:] = np.asarray([float(range_widgets[4].get()),
                                              float(range_widgets[5].get())])
        except:
            messagebox.showinfo('Input Error', 'Could not process IRF Wave. Range')
            return

        try:
            self.irf_range[1,:] = np.asarray([float(range_widgets[6].get()),
                                              float(range_widgets[7].get())])
        except:
            messagebox.showinfo('Input Error', 'Could not process IRF Time Range')
            return

        # convert wavelengths/times to indicies
        self.fl_range[0,:]  = np.sort(spc.closest(self.fl_range[0,:], self.wln_fl_raw))
        self.fl_range[1,:]  = np.sort(spc.closest(self.fl_range[1,:], self.t_fl_raw))
        self.irf_range[0,:] = np.sort(spc.closest(self.irf_range[0,:], self.wln_irf_raw))
        self.irf_range[1,:] = np.sort(spc.closest(self.irf_range[1,:], self.t_irf_raw))

        self.fl_range  = np.asarray(self.fl_range,  dtype=int)
        self.irf_range = np.asarray(self.irf_range, dtype=int)

        # generate the fl and irf traces and time axes 
        self.wln_fl = self.wln_fl_raw[self.fl_range[0,0]:self.fl_range[0,1]]
        self.t_fl   = self.t_fl_raw[self.fl_range[1,0]:self.fl_range[1,1]]
        self.fl     = self.fl_raw[self.fl_range[0,0]:self.fl_range[0,1],
                                  self.fl_range[1,0]:self.fl_range[1,1]].sum(axis=0)

        self.wln_irf = self.wln_irf_raw[self.irf_range[0,0]:self.irf_range[0,1]]
        self.t_irf   = self.t_irf_raw[self.irf_range[1,0]:self.irf_range[1,1]]
        self.irf     = self.irf_raw[self.irf_range[0,0]:self.irf_range[0,1],
                                    self.irf_range[1,0]:self.irf_range[1,1]].sum(axis=0)

    def check_range(self):
        # trim the data first
        self.trim()

        # plot the traces
        trace_window = tk.Toplevel()
        trace_window.title(r'Averaged Data and IRF')
        trace_window.geometry('600x525')

        fig = Figure()
        ax  = fig.subplots(nrows=1, ncols=1)
        fig.subplots_adjust(top=0.975, bottom=0.125, right=0.95)
        ax.plot(self.t_fl, self.fl, label='Fluorescence')
        ax.plot(self.t_irf, self.irf, color='gray', label='IRF')
        ax.set_xlabel('Time / ns')
        ax.set_ylabel('Averaged Counts')
        ax.set_yscale('log')
        ax.legend(fontsize=12, loc='upper right')

        # hocus pocus required to draw the figure with toolbar
        canvas = FigureCanvasTkAgg(fig, master=trace_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, trace_window).update()
        canvas.get_tk_widget().pack()

    def perform_fit(self):
        # update the trim
        self.trim()

        # make a fit data object
        res = ft.FitData()

        # get the model function
        res.fit_function = selected_model.get()

        # hang the data we need for the fit
        res.x     = self.t_fl
        res.data  = self.fl
        res.irf   = self.irf
        res.x_irf = self.t_irf
        res.x_full = self.t_fl_raw

        # identify the fit function
        if   res.fit_function == '1-exp':
            res.par_names = exp1_names
        elif res.fit_function == '2-exp':
            res.par_names = exp2_names
        elif res.fit_function == '3-exp':
            res.par_names = exp3_names
        elif res.fit_function == '1-str':
            res.par_names = str1_names
        elif res.fit_function == '2-str':
            res.par_names = str2_names
        n_par = len(res.par_names)

        # gather fit parameters
        res.lb    = np.zeros(n_par)
        res.guess = np.zeros(n_par)
        res.ub    = np.zeros(n_par)
        res.fix   = np.zeros(n_par)
        for i in range(n_par):
            try:
                res.lb[i] = float(lb[i].get())
            except:
                messagebox.showinfo('Input Error', 'An lb is not a number.')
                return

            try:
                res.guess[i] = float(guess[i].get())
            except:
                messagebox.showinfo('Input Error', 'A guess is not a number.')
                return

            try:
                res.ub[i] = float(ub[i].get())
            except:
                messagebox.showinfo('Input Error', 'A ub is not a number.')
                return

            if fix[i].state() == ():
                res.fix[i] = 1
            elif fix[i].state()[0] == 'selected':
                res.fix[i] = 0
            else:
                res.fix[i] = 1

        # send res to the fitter
        res = fit_routine(res)

        # plot the result
        result_window = tk.Toplevel()
        result_window.title(r'%s Fit Result (chi_sq = %.3f)' % (res.fit_function,
                                                                res.chi_sq))
        result_window.geometry('600x750')

        fig = Figure()
        ax  = fig.subplots(nrows=2, ncols=1, sharex=True, gridspec_kw={'height_ratios' : [1,5]})

        fig.subplots_adjust(left=0.135,  right=0.975,
                            top=0.975,  bottom=0.12,
                            wspace=0.1,  hspace=0.05)
        ax[0].plot(res.x, res.resid, '.m', markersize=3)
        ax[0].plot([-100, 500], [0,0], 'k', alpha=0.5)
        ax[0].set_ylabel('Residuals')

        if np.abs(res.resid).max() > np.abs(res.resid).min():
            ax[0].set_ylim([-1.1*np.abs(res.resid).max(),
                             1.1*np.abs(res.resid).max()])
        else:
            ax[0].set_ylim([-1.1*np.abs(res.resid).min(),
                             1.1*np.abs(res.resid).min()])

        ax[1].plot(res.x, res.data, '.b', markersize=3, label='Fluorescence')
        ax[1].plot(self.t_irf, self.irf, color='gray', label='IRF')
        ax[1].plot(res.x, res.fit, 'r', label='Fit', linewidth=2)

        ax[1].set_xlabel('Time / ns')
        ax[1].set_ylabel('Counts')
        ax[1].set_yscale('log')
        ax[1].set_xlim([self.t_fl_raw[0], self.t_fl_raw[-1]])
        ax[1].legend(fontsize=12, loc='upper right')

        fig.align_ylabels(ax)

        # hocus pocus required to draw the figure with toolbar
        canvas = FigureCanvasTkAgg(fig, master=result_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, result_window).update()
        canvas.get_tk_widget().pack()

        # spit out some text
        text = tk.Text(result_window)
        text.pack(fill=tk.BOTH, expand=True)

        # output the fit parameters
        text.insert(tk.END, '%s fit results:\n\n' % res.fit_function)
        if res.fit_function == '1-exp':
            text.insert(tk.END, 'I(t) = a1*exp(-t/tau1) + bkg (and tsft)\n\n')
        elif res.fit_function == '2-exp':
            text.insert(tk.END, 'I(t) = a1*exp(-t/tau1) + a2*exp(-t/tau2) + bkg (and tsft)\n\n')
        elif res.fit_function == '3-exp':
            text.insert(tk.END, 'I(t) = a1*exp(-t/tau1) + a2*exp(-t/tau2) + a3*exp(-t/tau3)] + bkg (and tsft)\n\n')
        elif res.fit_function == '1-str':
            text.insert(tk.END, 'I(t) = a1*exp(-[t/tau1]**beta1) + bkg (and tsft)\n\n')
        elif res.fit_function == '2-str':
            text.insert(tk.END, 'I(t) = a1*exp(-[t/tau1]**beta1) + a2*exp(-[t/tau2]**beta2) + bkg (and tsft)\n\n')

        for i in range(n_par):
            text.insert(tk.END, '{:>10} {:10.3f}\n'.format(res.par_names[i], res.fitpar[i]))

        # add chi_sq
        text.insert(tk.END, '\n{:>10} {:10.3f}\n\n'.format('chi_sq:', res.chi_sq))
        text.insert(tk.END, 'amplitudes, and bkg in counts, tsft and tau in ns\n\n')

        # write out some easy to copy text
        for i in range(n_par):
            text.insert(tk.END, '{:<10.3f}'.format(res.fitpar[i]))
        text.insert(tk.END, '{:<10.3f}'.format(res.chi_sq))

        # make the box un-editable
        text['state'] = 'disabled'

        # update GUI guesses with result
        for i in range(n_par):
            guess[i].delete(0, 'end')
            guess[i].insert(tk.END, '%.3f' % res.fitpar[i])

        return

# generate an instance of the class
data = raw_spectra()

###############################################################################
#                           Widget Placement                                  #
###############################################################################
####################
# File input frame #
####################
file_widgets = list()

# main buttons (row=0)
ttk.Button(frm_files, text='Load Data', command=data.load_files).grid(row=0, column=0, padx=5, pady=5)
ttk.Button(frm_files, text='Plot 2D', command=data.plot_2D).grid(row=0, column=1, padx=5)
ttk.Button(frm_files, text='Check Range', command=data.check_range).grid(row=0, column=2, padx=5)
ttk.Button(frm_files, text='Fit', command=data.perform_fit).grid(row=0, column=3, padx=5)
ttk.Button(frm_files, text='Quit', command=root.destroy).grid(row=0, column=4, padx=5)

# mag file (row=1, file_widgets[0])
ttk.Label(frm_files, text='Mag. File:').grid(row=1, column=0)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '/home/crumble/Documents/Altoona/research/crumble_lab/anisfit_2D/c153_x00glymeoh_2nsjc_magic.dac')
file_widgets[-1].grid(row=1, column=1, columnspan=3, padx=5, sticky='EW')
ttk.Button(frm_files, text='Select', command=data.get_fl_filename).grid(row=1, column=4, pady=5)

# par file (row=2, file_widgets[1])
ttk.Label(frm_files, text='Par. File:').grid(row=2, column=0)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '/home/crumble/Documents/Altoona/research/crumble_lab/anisfit_2D/cc153_x00glymeoh_2nsjc_para.dac')
file_widgets[-1].grid(row=1, column=1, columnspan=3, padx=5, sticky='EW')
ttk.Button(frm_files, text='Select', command=data.get_fl_filename).grid(row=1, column=4, pady=5)

# per file (row=2, file_widgets[2])
ttk.Label(frm_files, text='Per. File:').grid(row=3, column=0)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '/home/crumble/Documents/Altoona/research/crumble_lab/anisfit_2D/c153_x00glymeoh_2nsjc_perp.dac')
file_widgets[-1].grid(row=1, column=1, columnspan=3, padx=5, sticky='EW')
ttk.Button(frm_files, text='Select', command=data.get_fl_filename).grid(row=1, column=4, pady=5)

# irf file (row=2, file_widgets[3])
ttk.Label(frm_files, text='IRF File:').grid(row=4, column=0)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '/home/crumble/Documents/Altoona/research/crumble_lab/anisfit_2D/ccwater_2nsjc_magic.dac')
file_widgets[-1].grid(row=2, column=1, columnspan=3, padx=5, sticky='EW')
ttk.Button(frm_files, text='Select', command=data.get_irf_filename).grid(row=2, column=4, pady=5)

#########################
# Range selection frame #
#########################
range_widgets = list()

# lower/upper labels (row=0)
ttk.Label(frm_ranges, text='lower').grid(row=0, column=1, pady=5)
ttk.Label(frm_ranges, text='upper').grid(row=0, column=2, pady=5)

# fl range wavelength (row=1, range_widgets[0,1])
ttk.Label(frm_ranges, text='Fl. Wave. Range (nm):').grid(row=1, column=0, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '478')
range_widgets[-1].grid(row=1, column=1, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '484')
range_widgets[-1].grid(row=1, column=2, padx=5)

# fl range time (row=2, range_widgets[2,3])
ttk.Label(frm_ranges, text='Fl. Time Range (ns):').grid(row=2, column=0, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '0')
range_widgets[-1].grid(row=2, column=1, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '20')
range_widgets[-1].grid(row=2, column=2, padx=5)

# irf range wavelength (row=3, range_widgets[4,5])
ttk.Label(frm_ranges, text='IRF Wave. Range (nm):').grid(row=3, column=0, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '420')
range_widgets[-1].grid(row=3, column=1, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '430')
range_widgets[-1].grid(row=3, column=2, padx=5)

# fl range time (row=4, range_widgets[6,7])
ttk.Label(frm_ranges, text='IRF Time Range (ns):').grid(row=4, column=0, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '0')
range_widgets[-1].grid(row=4, column=1, padx=5)
range_widgets.append(ttk.Entry(frm_ranges))
range_widgets[-1].insert(tk.END, '0.23')
range_widgets[-1].grid(row=4, column=2, padx=5)

# fit function selector (row=5)
model_functions = ['1-exp', '2-exp', '3-exp', '1-str', '2-str']
selected_model = tk.StringVar()
selected_model.set(model_functions[0])
ttk.Label(frm_ranges, text='Fitting function:').grid(row=5, column=0, padx=5)
model = ttk.OptionMenu(frm_ranges, selected_model, model_functions[0],
                       *model_functions, command=change_model)
model.grid(row=5, column=1, padx=5, pady=5)

##################################
# initialize the parameter frame #
##################################
change_model('1-exp')

###########################
# Start the infinite loop #
###########################
root.mainloop()
















































