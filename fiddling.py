# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:15:09 2021

@author: conta
"""
      #%%
import sn_functions as sn
import sn_plotting as sp
import numpy as np
import data_utils as du


(datapath, savepath,t,i,e, label, sector, discdate, gal_mags, 
 info, binT, binI, binE, absT, absI, absE, standI, standE) = sn.load_2020efe()


#%% fULL light curve

(samples, best, 
 upper, lower) = stepped_powerlaw_basic(savepath, label, binT, 
                                        standI, standE, sector,
                                        discdate, plot = True, n1=40000, 
                                        n2=60000)

sp.print_table_formatting(best,upper,lower)




#%% DOUBLE POWER LAW PLOT
nrows = 2
ncols = 1
x = binT
y = standI
yerr = standE
t0 = 1916.3513
A = 0.0889
beta = 1.1509
B = 0.0539
t1 = x[0:41] - t0

t02 = 1919.9
A2 =  0.2443
beta2 = 0.5019
B2 = 0.3423
t2 = x[41:] - t02

model1 = (np.heaviside((t1), 1) * 
                      A *np.nan_to_num((t1**beta), copy=False) 
                      + B)
model2 = (np.heaviside((t2), 1) * 
                      A2 *np.nan_to_num((t2**beta2), copy=False) 
                      + B2)

fig, ax = plt.subplots(nrows, ncols, sharex=True,
                               figsize=(8*ncols * 2, 3*nrows * 2))

ax[0].plot(x[:41], model1, label="best fit model", color = 'red')
ax[0].plot(x[41:], model2, color = 'red')
ax[0].scatter(x, y, label = "FFI data", s = 5, color = 'black')
for n in range(nrows):
    ax[n].axvline(t0, color = 'blue', label="t1")
    ax[n].axvline(discdate, color = 'green', label="discovery time")
    ax[n].set_ylabel("Rel. Flux", fontsize=16)
    
#main
ax[0].set_title(label + " - Double Power Law Fit", fontsize=20)
ax[0].legend(fontsize=12, loc="upper left")
ax[nrows-1].set_xlabel("BJD-2457000", fontsize=16)

#residuals
ax[1].set_title("Residual (y-model)", fontsize=16)
residuals = y[:41] - model1
residual2 = y[41:] - model2
ax[1].scatter(x[:41],residuals, s=5, color = 'black', label='residual')
ax[1].scatter(x[41:], residual2, s = 5, color = 'black')
ax[1].axhline(0,color='purple', label="zero")
ax[1].legend()
plt.tight_layout()
plt.savefig(datapath + "double_powerlaw.png")


