#!/usr/bin/env python3
###############################################################################
# Chris Rumble
# 4/15/23
#
# A GUI interface for the convolute-and-compare fitter
###############################################################################
import tkinter as tk
from tkinter import ttk
import numpy as np
import conv_comp_fit
from tkinter import messagebox
from tkinter import filedialog
import fitting as ft
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

# create the main window
root = tk.Tk()
root.title('Convolute-and-Compare Fitter')
root.geometry('800x600')

# frames go inside windows
frm_files   = ttk.Frame(root, padding=10)
frm_pars    = ttk.Frame(root, padding=10)
frm_buttons = ttk.Frame(root, padding=10)

# use a grid layout
frm_files.grid()
frm_buttons.grid()
frm_pars.grid()

# these will hold the parameter widgets
lb    = list()
guess = list()
ub    = list()
fix   = list()

# we need some names for the parameters
exp1_names = ['h:', 'bkg:', 'tsft:', 'tau1:']
exp2_names = ['h:', 'bkg:', 'tsft:', 'a1:', 'tau1:', 'tau2:']
exp3_names = ['h:', 'bkg:', 'tsft:', 'a1:', 'tau1:', 
                                     'a2:', 'tau2:', 
                                     'a3:', 'tau3:']
str1_names = ['h:', 'bkg:', 'tsft:', 'tau1:', 'beta1:']
str2_names = ['h:', 'bkg:', 'tsft:', 'a1:', 'tau1:', 'beta1:', 
                                     'a2:', 'tau2:', 'beta2:']

##################################
# generate model parameter frame #
##################################
def change_model(fit_function):
    # first clear the parameter frame widgets
    for widget in frm_pars.winfo_children():
        widget.destroy()
    
    # fit parameter headers
    ttk.Label(frm_pars, text='lb').grid(column=1, row=1, pady=5)
    ttk.Label(frm_pars, text='guess').grid(column=2, row=1, pady=5)
    ttk.Label(frm_pars, text='ub').grid(column=3, row=1, pady=5)
    ttk.Label(frm_pars, text='fix').grid(column=4, row=1, pady=5)

    # identify the model parameters and their defaults
    if   fit_function == '1-exp':
        par_names = exp1_names
        ini_lb    = np.asarray([1,   -1e2, -1, 1e-4])
        ini_guess = np.asarray([5e3,    0,  0, 1])
        ini_ub    = np.asarray([1e5,  1e2,  1, 1e5])
    elif fit_function == '2-exp':
        par_names = exp2_names
        ini_lb    = np.asarray([1,   -1e2, -1,  -1, 1e-4, 1e-4])
        ini_guess = np.asarray([5e3,    0,  0, 0.5, 1,    5])
        ini_ub    = np.asarray([1e5,  1e2,  1,   1, 1e5,  1e5])
    elif fit_function == '3-exp':
        par_names = exp3_names
        ini_lb    = np.asarray([1,   -1e2, -1,  -1, 1e-4, -1, 1e-4, -1,   1e-4])
        ini_guess = np.asarray([5e3,    0,  0, 0.5, 1,   0.5, 10,    0.5, 50])
        ini_ub    = np.asarray([1e5,  1e2,  1,   1, 1e5,   1, 1e5,   1,   1e5])
    elif fit_function == '1-str':
        par_names = str1_names
        ini_lb    = np.asarray([1,   -1e2, -1, 1e-4, 0])
        ini_guess = np.asarray([5e3,    0,  0, 1,    0.7])
        ini_ub    = np.asarray([1e5,  1e2,  1, 1e5,  1])
    elif fit_function == '2-str':
        par_names = str2_names
        ini_lb    = np.asarray([1,   -1e2, -1,  -1, 1e-4, 0,    -1, 1e-4, 0])
        ini_guess = np.asarray([5e3,    0,  0, 0.5, 1,    0.7, 0.5, 1,    0.7])
        ini_ub    = np.asarray([1e5,  1e2,  1,   1, 1e5,  1,     1, 1e5,  1])
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
        
###################
# perform the fit #
###################
def perform_fit():
    # make a fit data object
    res = ft.FitData()
    
    # get the file data
    res.fl_file = file_widgets[0].get()
    res.irf_file = file_widgets[3].get()
    
    # get the fluorescence range
    try:
        res.fl_range = np.asarray([int(file_widgets[1].get()),
                                   int(file_widgets[2].get())])
    except:
        messagebox.showinfo('Input Error', 'Fluorescence range is not an integer')
        return

    # get the irf range
    res.irf_file = file_widgets[3].get()
    try:
        res.irf_range = np.asarray([int(file_widgets[4].get()),
                                   int(file_widgets[5].get())])
    except:
        messagebox.showinfo('Input Error', 'IRF range is not an integer')
        return

    # get the model function
    res.fit_function = selected_model.get()
    
    # identify the fit function
    if   res.fit_function == '1-exp':
        n_par = 4
        res.par_names = exp1_names
    elif res.fit_function == '2-exp':
        n_par = 6
        res.par_names = exp2_names
    elif res.fit_function == '3-exp':
        n_par = 9
        res.par_names = exp3_names
    elif res.fit_function == '1-str':
        n_par = 5
        res.par_names = str1_names
    elif res.fit_function == '2-str':
        n_par = 9
        res.par_names = str2_names

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
        
        if fix[i].state()[0] == 'selected':
            res.fix[i] = 0
        else:
            res.fix[i] = 1

    # send res to the fitter
    res = conv_comp_fit.fit_routine(res)
    
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
    ax[1].plot(res.x_irf, res.irf, color='gray', label='IRF')
    ax[1].plot(res.x, res.fit, 'r', label='Fit', linewidth=2)

    ax[1].set_xlabel('Time / ns')
    ax[1].set_ylabel('Counts')
    ax[1].set_yscale('log')
    ax[1].set_xlim([res.x_full[0], res.x_full[-1]])
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
        text.insert(tk.END, 'I(t) = h*exp(-t/tau1) + bkg (and tsft)\n\n')
    elif res.fit_function == '2-exp':
        text.insert(tk.END, 'I(t) = h[a1*exp(-t/tau1) + (1-a1)*exp(-t/tau2)] + bkg (and tsft)\n\n')
    elif res.fit_function == '3-exp':
        text.insert(tk.END, 'I(t) = h[a1*exp(-t/tau1) + a2*exp(-t/tau2) + a3*exp(-t/tau3)] + bkg (and tsft)\n')
        text.insert(tk.END, 'a1 + a2 + a3 = 1\n\n')
    elif res.fit_function == '1-str':
        text.insert(tk.END, 'I(t) = h*exp(-[t/tau1]**beta1) + bkg (and tsft)\n\n')
    elif res.fit_function == '2-str':
        text.insert(tk.END, 'I(t) = h*[a1*exp(-[t/tau1]**beta1) + (1-a1)*exp(-[t/tau2]**beta2)] + bkg (and tsft)\n\n')

    for i in range(n_par):
        text.insert(tk.END, '{:>10} {:10.3f}\n'.format(res.par_names[i], res.fitpar[i]))

    # add chi_sq
    text.insert(tk.END, '\n{:>10} {:10.3f}\n\n'.format('chi_sq:', res.chi_sq))

    text.insert(tk.END, 'h and bkg in counts, tsft and tau in ns')

    # make the box un-editable
    text['state'] = 'disabled'
    
    # update GUI guesses with result
    for i in range(n_par):
        guess[i].delete(0, 'end')
        guess[i].insert(tk.END, '%.3f' % res.fitpar[i])

    return

##########################################
# implement the file selection dialogues #
##########################################
def get_fl_filename():
    filename = filedialog.askopenfilename()
    file_widgets[0].delete(0, tk.END)
    file_widgets[0].insert(tk.END, filename)
    return

def get_irf_filename():
    filename = filedialog.askopenfilename()
    file_widgets[3].delete(0, tk.END)
    file_widgets[3].insert(tk.END, filename)
    return

####################
# File input frame #
####################
# fit and quit buttons
ttk.Button(frm_files, text='Fit', command=perform_fit).grid(
           column=0, row=0, padx=10, pady=20)

ttk.Button(frm_files, text='Quit', command=root.destroy).grid(
           column=2, row=0, padx=10, pady=20)

file_widgets = list()

# fluorescence file
ttk.Label(frm_files, text='Fl. File:').grid(column=0, row=1, pady=2)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '/home/crumble/Documents/Altoona/research/tcspc/conv_comp/C153_MeOH_400ex_520em.asc')
file_widgets[-1].grid(column=1, row=1, padx=5, columnspan=4, sticky='ew')
ttk.Button(frm_files, text='Select', command=get_fl_filename).grid(column=5, row=1)

# fluorescence range
ttk.Label(frm_files, text='Fl. Range:').grid(column=0, row=3, pady=2)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '100')
file_widgets[-1].grid(column=1, row=3, pady=2)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '3750')
file_widgets[-1].grid(column=2, row=3, pady=2)

# irf file
ttk.Label(frm_files, text='IRF File:').grid(column=0, row=4, pady=2)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '/home/crumble/Documents/Altoona/research/tcspc/conv_comp/irf_400ex_400em_6-11ps.asc')
file_widgets[-1].grid(column=1, row=4, padx=5, columnspan=4, sticky='ew')
ttk.Button(frm_files, text='Select', command=get_irf_filename).grid(column=5, row=4)

# irf range
ttk.Label(frm_files, text='IRF Range:').grid(column=0, row=5, pady=2)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '100')
file_widgets[-1].grid(column=1, row=5, pady=2)
file_widgets.append(ttk.Entry(frm_files))
file_widgets[-1].insert(tk.END, '400')
file_widgets[-1].grid(column=2, row=5, pady=2)

# fit function selector
model_functions = ['1-exp', '2-exp', '3-exp', '1-str', '2-str']
selected_model = tk.StringVar()
selected_model.set(model_functions[0])
ttk.Label(frm_files, text='Fitting function:').grid(column=0, row=6)
model = ttk.OptionMenu(frm_files, selected_model, model_functions[0],
                       *model_functions, command=change_model)
model.grid(column=1, row=6, padx=10, pady=2)

##################################
# initialize the parameter frame #
##################################
change_model('1-exp')

###########################
# Start the infinite loop #
###########################
root.mainloop()
