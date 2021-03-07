# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 12:38:52 2021

@author: Lindsey Gordon

run_supernovae
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

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import fnmatch

import astroquery
from astroquery.simbad import Simbad
from astroquery.mast import Catalogs
from astroquery.mast import Observations

#import data_functions as df
#import plotting_functions as pf
#import ffi_hyperleda as fh

#%% run the TNS cross check on a single sector's date range
import sn_functions as sn
        #%%
info, observed = sn.cross_check_TNS("C:/Users/conta/Downloads/", 1, "2018-07-25", "2018-08-22")

#%% cross-check ALL sectors between TNS and TESSCUT

import sn_functions as sn
#run through entire catalog of cross-checks
folder = "C:/Users/conta/UROP/all_crosschecks/"
tessdates_csv = "C:/Users/conta/Downloads/orbit_times_20201013_1338.csv"
sn.run_all_TNS_TESSCUT(folder, tessdates_csv, sector=1)
#%% concatenate all individual sectors together, saving sector information.

savefilename = "all_tns_tesscut_results"       
all_tns_tesscut = sn.compile_csvs(folder, "-crossmatched.csv",
                          savefilename = savefilename, sector = True)
#%% Convert to degrees for WTV upload

tns_file = "C:/Users/conta/UROP/all_crosschecks/all_tns_tesscut_results.csv"
output_file = "C:/Users/conta/UROP/all_crosschecks/WTV_tns_tesscut_upload.csv"

sn.prep_WTV_file(tns_file, output_file)

#%% Then you have to upload that file into the WTV webpage, 
#https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py
#and save the resulting CSV file. 
#%% Run output WTV file through all the correct sectors you're looking for data in
wtv_check_output_file = folder + "WTV_TESSCUT_double_confirmed.csv"
WTV_values = pd.read_csv("C:/Users/conta/UROP/all_crosschecks/wtv-WTV_tns_tesscut_upload.csv", skiprows = 61)  
sn.process_WTV_results(all_tns_tesscut, WTV_values, wtv_check_output_file)

#%% worthwhile to also run on ENTIRE TNS list pre-TESSCUT run as well:

savefilename = "all_tns_results"  
folder = "C:/Users/conta/UROP/all_crosschecks/"
all_tns = sn.compile_csvs(folder, "-0.csv",
                          savefilename = savefilename, sector = True)
#%%
tns_file = "C:/Users/conta/UROP/all_crosschecks/all_tns_results.csv"
output_file = "C:/Users/conta/UROP/all_crosschecks/WTV_tns_upload.csv"

sn.prep_WTV_file(tns_file, output_file)

#%% Then you have to upload that file into the WTV webpage, 
#https://heasarc.gsfc.nasa.gov/cgi-bin/tess/webtess/wtv.py
#and save the resulting CSV file. 
# Run output WTV file through all the correct sectors you're looking for data in
wtv_check_output_file = "C:/Users/conta/UROP/all_crosschecks/WTV_only_crossmatch.csv"
WTV_values = pd.read_csv("C:/Users/conta/UROP/all_crosschecks/wtv-WTV_tns_upload.csv", skiprows = 61)  
sn_list = sn.process_WTV_results(all_tns, WTV_values, wtv_check_output_file)

#%% clean to only have Ia < 20 mags
output = "C:/Users/conta/UROP/all_crosschecks/WTV_only_cleaned.csv"
cleaned_targets = sn.only_Ia_20th_mag(sn_list, output)