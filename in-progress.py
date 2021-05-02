
"""
Created on Mon May 18 17:01:26 2020

@author: Lindsey Gordon 

Updated: May 31 2020
"""

import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

from pylab import rcParams
rcParams['figure.figsize'] = 10,10

rcParams["lines.markersize"] = 2
from scipy.signal import argrelextrema


import astropy
import astropy.units as u
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions
from astroquery import exceptions
from astroquery.exceptions import RemoteServiceError
#from astropy.utils.exceptions import AstropyWarning, RemoteServiceError

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import sklearn
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import fnmatch

from sklearn.metrics import confusion_matrix
from sklearn.neighbors import LocalOutlierFactor

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations



                   
    #%% HERE
"""To do: 
    - rerun fitting to them
    - do data analysis on the fittings
    - pick a bunch and deep dive on them!! literally that's all that's left to do bitch!!
    
    
    
    """
 
import sn_functions as sn
import sn_plotting as sp
import data_utils as dt

#%% individual load
#datafolder = "C:/Users/conta/UROP/lightcurves/batch1BEST/"
datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2019yft/'
savepath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2019yft/'
all_t, all_i, all_e, all_labels, sector_list, discovery_dictionary, t_starts, gal_mags, info = sn.mcmc_load_lygos(datapath, savepath, runproduce = False)


best_params_file = savepath + "best_params.csv"
ID_file = savepath + "ids.csv"
upper_errors_file = savepath + "uppererr.csv"
lower_errors_file = savepath + "lowererr.csv"

#bestParams, upperError, lowerError, sn_names = sn.load_params(savepath)

#badIndexes for 2020efe: 0:100, 628:641

#%% determine bad indexes
plt.scatter(all_t[0],all_i[0], color = 'red', s=5)

plt.axvline(all_t[0][415])
plt.axvline(all_t[0][440])
#plt.axvline(all_t[0][435])



#badIndexes = np.arange(590,655)
badIndexes = None #2019yft
#badIndexes = np.arange(415,440) #2018koy
#%%
fig, ax = plt.subplots()
ax.scatter(absT, absI)
ax.axvline(absT[58])
ax.invert_yaxis()
#%%
maggie = absI[58]
import astropy.units as u
L_0 = 3.0128 * 1e28 * u.W
lum = L_0 * 10**(-0.4 * maggie)

radnickel = 3.0 * 10**16 * u.erg/u.g

amtnickel = (lum/radnickel).to(u.g/u.s)
#%% plotting rel flux and abs mag stuff

#key = all_labels[0]
#apparent_galmag = gal_mags[key[:-4]]
t = all_t[0]
i = all_i[0]
e = all_e[0]
label = all_labels[0]
sector = 21
#yft: 0.069, 17.1
apparent_galmag = 20.4
gal_extinction = 0.069
z = info["Z"][0]

import sn_functions as sn



(binT, binI, binE, absT, absI, absE) = plot_SN_LCs(savepath, t,i,e,label,sector,
                                       apparent_galmag,gal_extinction,
                                       z, badIndexes)


        #%%
import sn_plotting as sp
sp.print_table_formatting(best,upper,lower)



#%%
import sn_functions as sn
#binT40, binI40, binE40 = sn.crop_to_percent(binT, standardBinnedI,df, 0.4)
savepath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/40/'
(best, upper, 
 lower) = stepped_powerlaw_basic(savepath, all_labels[0],
                                binT, standardBinnedI,df, 
                                 sector_list[0],
                     discovery_dictionary, t_starts, best_params_file,
                     ID_file, upper_errors_file, lower_errors_file, plot = True, 
                     n1=20000,n2=20000)
#%%
def stepped_powerlaw_basic(path, targetlabel, t, intensity, error, sector,
                     discovery_times, t_starts, best_params_file,
                     ID_file, upper_errors_file, lower_errors_file,
                     plot = True, n1=20000, n2=40000):
    """ Runs MCMC fitting for stepped power law fit
    This is the fitting that matches: Firth 2015 (fireball), Olling 2015, Fausnaugh 2019
    fireball power law with A, beta, B, and t0 floated
    t1 = t - t0
    F = A(t1)**beta + B + corrections
    Runs two separate chains to hopefully hit convergence
    Params:
            - path to save into
            - targetlabel for file names
            - time axis
            - intensities
            - errors
            - sector number (list)
            - discovery times dictionary
            - time start dictionary
            - file to save best params into
            - file to save name of sn into
            - file to save upper errors on all params into
            - file to save lower errors on all params into
            - plot (true/false)
            - path to folder containing quaternions
            - path to folder containing CBVs
    
    """

    def log_likelihood(theta, x, y, yerr):
        """ calculates the log likelihood function. 
        constrain beta between 0.5 and 4.0
        A is positive
        only fit up to 40% of the flux"""
        t0, A, beta, B = theta 
        t1 = x - t0
        model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)) + B)
        
        yerr2 = yerr**2.0
        returnval = -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        return returnval
    
    def log_prior(theta, disctime):
        """ calculates the log prior value """
        t0, A, beta, B= theta 
        if ((disctime - 5) < t0 < (disctime + 18) and 0.5 < beta < 4.0 
            and -0.1 < A < 2.0 and -5 < B < 5):
            return 0.0
        return -np.inf
        
        #log probability
    def log_probability(theta, x, y, yerr, disctime):
        """ calculates log probabilty"""
        lp = log_prior(theta,disctime)
            
        if not np.isfinite(lp) or np.isnan(lp): #if lp is not 0.0
            return -np.inf
        
        return lp + log_likelihood(theta, x, y, yerr)
    
    import matplotlib.pyplot as plt
    import emcee
    rcParams['figure.figsize'] = 16,6
     
    x = t
    y = intensity
    yerr = error
    disctime = discovery_times[targetlabel]
    
    #running MCMC
    np.random.seed(42)   
    nwalkers = 32
    ndim = 4
    labels = ["t0", "A", "beta",  "B"]
    
    p0 = np.zeros((nwalkers, ndim)) 
    for n in range(len(p0)):
        p0[n] = np.array((disctime, 0.1, 1.8, 1)) #mean values from before

    p0 += (np.array((0.1,0.1,0.1, 0.1)) * np.random.rand(nwalkers,ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr, disctime))
    
   # run ONCE
    sampler.run_mcmc(p0, n1, progress=True)
    sampler.get_chain()
    if plot:
        samples = sampler.get_chain()
        plot_chain(path, targetlabel, "-burn-in-plot-intermediate.png", samples, labels, ndim)
    
    
    flat_samples = sampler.get_chain(discard=6000, thin=15, flat=True)
    
    #get intermediate best
    best_mcmc_inter = np.zeros((1,ndim))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        best_mcmc_inter[0][i] = mcmc[1]
        

    #reset p0 and run again
    np.random.seed(50)
    p0 = np.zeros((nwalkers, ndim))
    for i in range(nwalkers):
        p0[i] = best_mcmc_inter[0] + 0.1 * np.random.rand(1, ndim)
       
    #sampler.reset()
    sampler.run_mcmc(p0,n2, progress = True)
    if plot:
        samples = sampler.get_chain()
        plot_chain(path, targetlabel, "-burn-in-plot-final.png", samples[n1:], labels, ndim)
    
    flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
    #print(len(flat_samples), "samples post second run")

    #print out the best fit params based on 16th, 50th, 84th percentiles
    best_mcmc = np.zeros((1,ndim))
    upper_error = np.zeros((1,ndim))
    lower_error = np.zeros((1,ndim))
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        print(labels[i], mcmc[1], -1 * q[0], q[1] )
        best_mcmc[0][i] = mcmc[1]
        upper_error[0][i] = q[1]
        lower_error[0][i] = q[0]
 
    
    if plot:
        #corner plot the samples
        import corner
        fig = corner.corner(
            flat_samples, labels=labels,
            quantiles = [0.16, 0.5, 0.84],
                           show_titles=True,title_fmt = ".4f", title_kwargs={"fontsize": 12}
        );
        fig.savefig(path + targetlabel + 'corner-plot-params.png')
        plt.show()
        plt.close()
        
         
        #best fit model
        t1 = x - best_mcmc[0][0]
        A = best_mcmc[0][1]
        beta = best_mcmc[0][2]
        B = best_mcmc[0][3]
        
        best_fit_model = (np.heaviside((t1), 1) * 
                          A *np.nan_to_num((t1**beta), copy=False) 
                          + B)
        
        nrows = 2
        ncols = 1
        fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                       figsize=(8*ncols * 2, 3*nrows * 2))
        
        ax[0].plot(x, best_fit_model, label="best fit model", color = 'red')
        ax[0].scatter(x, y, label = "FFI data", s = 5, color = 'black')
        for n in range(nrows):
            ax[n].axvline(best_mcmc[0][0], color = 'blue', label="t0")
            ax[n].axvline(disctime, color = 'green', label="discovery time")
            ax[n].set_ylabel("Rel. Flux")
            
        #main
        ax[0].set_title(targetlabel)
        ax[0].legend(fontsize=8, loc="upper left")
        ax[nrows-1].set_xlabel("BJD-" + str(t_starts[targetlabel]))
        
        #residuals
        ax[1].set_title("Residual (y-model)")
        residuals = y - best_fit_model
        ax[1].scatter(x,residuals, s=5, color = 'black', label='residual')
        ax[1].axhline(0,color='purple', label="zero")
        ax[1].legend()
        
        plt.savefig(path + targetlabel + "-MCMCmodel-stepped-powerlaw.png")
        #plt.close()
        
                
    with open(best_params_file, 'a') as f:
        for i in range(ndim):
            f.write(str(best_mcmc[0][i]) + ",")
        f.write("\n")
    with open(ID_file, 'a') as f:
        f.write(targetlabel)
        f.write("\n")
    with open(upper_errors_file, 'a') as f:
        for i in range(ndim):
            f.write(str(upper_error[0][i]) + ",")
        f.write("\n")
    with open(lower_errors_file, 'a') as f:
        for i in range(ndim):
            f.write(str(lower_error[0][i]) + ",")
        f.write("\n")
    
    beep()
    return best_mcmc, upper_error, lower_error

#%% converting the calculated power law into the 

t1 = binT - best[0][0]
A = best[0][1]
beta = best[0][2]
B = best[0][3]

best_fit_model = (np.heaviside((t1), 1) * 
                  A *np.nan_to_num((t1**beta), copy=False) 
                  + B)

model_noH = A * np.nan_to_num((binT**beta), copy=False) + B

plt.scatter(binT, best_fit_model)
plt.scatter(binT, model_noH)
#%%
#retesting quaternions
import sn_functions as sn
yftQUATS = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/hlsp_tess-spoc_tess_phot_2-4-s0006_tess_v1_cbv.fits"
from astropy.io import fits

opened = fits.open(yftQUATS, memmap = False)

times = []
cbv1 = []
cbv2 = []
cbv3 = []

#rcParams['figure.figsize'] = 8,3
for n in range(len(opened[1].data)):
    times.append(opened[1].data[n][0])
    cbv1.append(opened[1].data[n][5])
    cbv2.append(opened[1].data[n][6])
    cbv3.append(opened[1].data[n][7])
    
opened.close()

times1, cbv1 = sn.bin_8_hours_CBV(times, cbv1)


times2, cbv2 = sn.bin_8_hours_CBV(times, cbv2)


times3, cbv3 = sn.bin_8_hours_CBV(times, cbv3)

nrows = 3
ncols = 1
fig, ax = plt.subplots(nrows, ncols, sharex=True, 
                       figsize=(8*ncols, 3*nrows))
ax[0].plot(times1, cbv1)
ax[0].set_title("Sector 6 Camera 2 CCD 4 CBV1")
ax[1].plot(times2, cbv2 )
ax[1].set_title("Sector 6 Camera 2 CCD 4 CBV2")
ax[2].plot(times3, cbv3 )
ax[2].set_title("Sector 6 Camera 2 CCD 4 CBV3")
ax[nrows-1].set_xlabel("BJD-2457000")
plt.savefig("C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/cbvs.png")



#%%

#fixing background on 2018koy

datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/'
savepath = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/"
(t,i,e, label, sector, 
 discdate, gal_mags, info) = sn.mcmc_load_one(datapath, savepath, 
                                              runproduce = False)
#%%

savepath_lc = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/"
RA = info["RA"][0]
DEC = info["DEC"][0]
data = du.eleanor_lc(savepath_lc, RA, DEC, '2018koy',6)
q = data.quality==0

#%%
#plt.scatter(data.time[q], data.raw_flux[q])
plt.plot(data.time, data.flux_bkg, 'k', label='1D postcard', linewidth=3)
plt.plot(data.time, data.tpf_flux_bkg, 'r--', label='1D TPF', linewidth=2)
#%%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,4))
ax1.imshow(data.tpf[0])
ax1.set_title('Target Pixel File')
ax2.imshow(data.bkg_tpf[0])
ax2.set_title('2D interpolated background')
#%%
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(30,12))
ax1.imshow(data.tpf[0])
ax1.set_title('Target Pixel File')
ax2.imshow(data.aperture)
ax2.set_title('Aperture');


#%%
import eleanor
vis = eleanor.Visualize(data)


#%%
eleanor.TargetData.custom_aperture(data, shape='circle', 
                                   r=1, pos=[7,4], method='exact')
eleanor.TargetData.get_lightcurve(data)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15,4), 
                               gridspec_kw={'width_ratios':[1,3]})
ax1.imshow(data.tpf[0])
ax1.imshow(data.aperture, cmap='Greys', alpha=0.5)
ax1.set_title('Aperture over TPF')

#plt.imshow(data.aperture)

ax2.plot(data.time[q], data.raw_flux[q]/np.nanmedian(data.raw_flux[q]), 'k', label='Raw')
ax2.plot(data.time[q], data.corr_flux[q]/np.nanmedian(data.corr_flux[q]) - 0.015, 'r', label='Corrected')
ax2.legend()
ax2.set_xlabel('Time [BJD - 2457000]')
ax2.set_ylabel('Normalized Flux');

#%%
file = "C:/Users/conta/Downloads/MAST_2021-05-01T1836/HLSP/hlsp_eleanor_tess_ffi_postcard-s0006-2-4-cal-1196-1240_tess_v2_pc/hlsp_eleanor_tess_ffi_postcard-s0006-2-4-cal-1196-1240_tess_v2_pc.fits"

filey = fits.open(file, memmap=False)


#%%
from astropy.wcs import WCS
wicks = WCS(filey[1].header)

#%%
plt.imshow(filey[2].data[0])

#%%
from astropy.coordinates import SkyCoord
ra = info["RA"][0]
dec = info["DEC"][0]
pixCoords=SkyCoord(ra, dec, frame='fk5', 
                   unit=u.deg).to_pixel(wcs=wicks, mode='all')
fig = plt.figure(figsize=(10,10))                        #plotting only quasars that lie in image
fig.add_subplot(111, projection=wicks)
plt.xlim(pixCoords[0]-10, pixCoords[0]+10)
plt.ylim(pixCoords[1]-10, pixCoords[1]+10)
plt.imshow(filey[2].data[0], origin='lower', cmap='gray',clim=(0,.99))
plt.xlabel('RA')
plt.ylabel('Dec')
plt.grid(color='white', ls='solid')
#plt.contour(hdu.data, levels=[.1,.5,1,4,7,10,20], cmap='cool', alpha=0.5)

#pixCoords=np.concatenate(pixCoords)            #cropping image to only quasar






