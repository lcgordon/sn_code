
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
 
import sn_functions as sn
import sn_plotting as sp
import data_utils as dt


#%%




#%%
#retesting quaternions
import sn_functions as sn
yftQUATS = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2020chi/hlsp_tess-spoc_tess_phot_1-4-s0021_tess_v1_cbv.fits"
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
ax[0].set_title("Sector 21 Camera 1 CCD 4 CBV1")
ax[1].plot(times2, cbv2 )
ax[1].set_title("Sector 21 Camera 1 CCD 4 CBV2")
ax[2].plot(times3, cbv3 )
ax[2].set_title("Sector 21 Camera 1 CCD 4 CBV3")
ax[nrows-1].set_xlabel("BJD-2457000")
plt.savefig("C:/Users/conta/UROP/plot_output/IndividualFollowUp/2020chi/2020chi-cbvs.png")





#%%
t0 = 1
xfake = np.arange(0,8,0.5)

betas = [0.5, 1,1.5,2,2.5]
A = 0.01

plt.axvline(1, label = "t_0", c='black')
for n in range(len(betas)):
    t1 = xfake - t0
    model = np.heaviside((t1), 1) * A * t1**(betas[n]) 
    plt.plot(xfake, model, label=str(betas[n]))
    

plt.xlabel("Time", fontsize=12)
plt.ylabel("Amplitude", fontsize=12)
plt.title("Demonstration of different " + r'$\beta$' + 
          " terms in " + r'$H(t_1)At_1^{\beta}$', fontsize = 12)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("C:/Users/conta/UROP/plot_output/IndividualFollowUp/beta-terms.png")

