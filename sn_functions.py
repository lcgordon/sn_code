# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 23:07:49 2020

@author: conta
"""

#Supernovae fitting functions
import numpy as np
import numpy.ma as ma 
import pandas as pd 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)

import scipy.signal as signal
from scipy.stats import moment
from scipy import stats
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10
rcParams["lines.markersize"] = 2
# rcParams['lines.color'] = 'k'
from scipy.signal import argrelextrema

import fnmatch

from datetime import datetime
import os
import shutil
from scipy.stats import moment, sigmaclip

import astropy
from astropy.io import fits
import scipy.signal as signal
from astropy.stats import SigmaClip
from astropy.utils import exceptions

import data_utils as du

  
####### CROSS MATCHING ######
    
def run_all_TNS_TESSCUT(savefolder, tessdates_csv, sector=1):
    """ Runs cross_check_TNS_with_TESSCUT() for all sectors using the dates per
    sector provided by TESS. Saves each file into the given save folder
    Parameters:
        - savefolder: where to save the output files
        - tessdates_csv: path to the csv file from TESS of dates
        - sector: sector to start at, defaults to sector 1
        
    **your file is "C:/Users/conta/Downloads/orbit_times_20201013_1338.csv" """
    tessdates = pd.read_csv(tessdates_csv, skiprows = 5)
    n = sector * 2 - 2
    while n < len(tessdates):
        start = tessdates["Start UTC"][n].split(" ")[0]
        end = tessdates["End UTC"][n+1].split(" ")[0]
        print(sector)
        print(start, end)
        info, obs = cross_check_TNS_with_TESSCUT(savefolder, sector, start, end)
        sector += 1
        n += 2
    return

def cross_check_TNS_with_TESSCUT(savepath, sector, sector_start, sector_end):
    """" For a given sector's date range, retrieves the TNS results and then 
    cross-matches all positions against TESSCUT. Saves only the ones that were 
    observed during their discovery date. 
    Parameters:
        - savepath for output files
        - sector being observed (int)
        - sector_start: date to start search, format like "2020-11-10"
        -sector_end: date to end search, same as above format
    Returns: nothing. saved output csv's end in -crossmatched.csv """
    import tns_py as tns
    url = tns.CSV_URL(date_start = sector_start, date_end = sector_end,
                      classified_sne = True)
    filelabel = "Sector-" + str(sector)
    tns.TNS_get_CSV(savepath, filelabel, url)
    
    #open and read file
    import pandas as pd
    info = pd.DataFrame()
    for i in range(2):
        file = savepath + filelabel + "-" + str(i) + ".csv"
        pand = pd.read_csv(file)
        if info.empty:
            #putintothing
            info = pand
            
        if pand.empty:
            #dont concatenate
            continue
        else:
            #concatenate
            info = pd.concat((info, pand))
    
    #run each set of coordinates into tesscut
    observed_targets = pd.DataFrame(columns = info.columns)
    from astroquery.mast import Tesscut
    from astropy.coordinates import SkyCoord
    import warnings
    import astropy.units as u

    with warnings.catch_warnings():
        for n in range(len(info)):#for each entry
            coord = SkyCoord(info["RA"][n], info["DEC"][n], unit=(u.hourangle, u.deg))
            sector_table = Tesscut.get_sectors(coordinates=coord)
            #print(sector_table)
            
            #check each table item
            for i in range(len(sector_table)):
                if sector_table["sector"][i] == sector or sector_table["sector"][i] == sector - 1:
                    #if in this sector or the previous one, add to list of ones to save   
                    #if observed_targets.empty:
                     #   observed_targets = info[n]
                    #else:
                    observed_targets = observed_targets.append(info.iloc[[n]])
    
    savefile = savepath + filelabel + "-crossmatched.csv"
    observed_targets.to_csv(savefile)
            
    return info, observed_targets



def compile_csvs(folder, suffix, savefilename = None, sector = False):
    """Compile all TNS CSV's, including sector information as one of the columns in the output 
    Parameters:
        - folder: containing all tns csv files
        - suffix: ending to search for on end of all files (probably -0.csv)
        - savefilename: if not none, will save output of this into this file
        - sector: if not false, save sector information into output list.
        
    Returns: concatenated pandas data frame containing all info from the csv files"""
    all_info = pd.DataFrame()
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith((suffix)):
                filepath = root  + f
                p = pd.read_csv(filepath)
                if sector:
                    s = f.split("-")[1]
                    p.rename(columns = {"Unnamed: 0": "Sector"}, inplace = True)
                    p.rename(columns = {"index" : "Sector"}, inplace = True)
                    p = p.assign(Sector = int(s))
                
                #print(p)
                if all_info.empty:
                    all_info = p
                    #print("first loaded")
                
                else:
                    all_info = pd.concat((all_info, p))
                    #print("concatenated")
                    
    if savefilename is not None:
        all_info.to_csv(folder + savefilename + ".csv", index=False)
        
    return all_info.reset_index()

def prep_WTV_file(tnsfile, outputfile):
    """This converts all RA and DEC in an input pandas-readable CSV file
    from whatever format it currently is in into degrees and saves it into 
    its own new csv file."""
    #converting to decimal degrees
    import pandas as pd
    from astropy.coordinates import Angle
    import astropy.units as u
    df = pd.read_csv(tnsfile)
    print (df)
    
    for n in range(len(df)):
        a = Angle(df['RA'][n], u.hourangle)
        a = a.degree
        df['RA'][n] = a
        b = Angle(df['DEC'][n], u.degree)
        b = b.degree
        df['DEC'][n] = b
    
    new = df[['RA', 'DEC']].copy()
    print(new)
    new.to_csv(outputfile, index = False)
    return

def process_WTV_results(all_tns, WTV_values, output_file):
    """Returns a file only containing the WTV results where the sector observed
    matches the sector of discovery OR the prior sector"""
    just_sectors = all_tns["Sector"]
    #counter = 0
    WTV_confirmed =  pd.DataFrame(columns = all_tns.columns)
    for n in range(len(WTV_values) - 1):
        correct_sector = just_sectors[n]
        columnname = "S" + str(correct_sector)
        prior_sector = correct_sector - 1
        priorname = "S" + str(prior_sector)
        next_sector = correct_sector + 1
        nextname = "S" + str(next_sector)
        if correct_sector != 1:
            if WTV_values[columnname][n] != 0.0 or WTV_values[nextname][n] != 0.0 or WTV_values[priorname][n] != 0.0: 
                #THE MINUS ONE MAKES A CORRECTION FOR THE WEIRD FIRST EMPTY LINE IN WTV FILES
                WTV_confirmed = WTV_confirmed.append(all_tns.iloc[[(n-1)]])
        else:
            if WTV_values[columnname][n] != 0.0 or WTV_values[nextname][n] != 0.0: 
                #THE MINUS ONE MAKES A CORRECTION FOR THE WEIRD FIRST EMPTY LINE IN WTV FILES
                WTV_confirmed = WTV_confirmed.append(all_tns.iloc[[(n-1)]])
            
    WTV_confirmed.reset_index(inplace = True, drop=True)   
    #del WTV_confirmed['index']     
    WTV_confirmed.to_csv(output_file, index=False)
    return WTV_confirmed

def only_Ia_20_mag(sn_list, output):
    """Clears out all sn from a list that are not Ia and brighter than 22nd magnitude
    at time of detection"""
    cleaned_targets = pd.DataFrame(columns = sn_list.columns)
    for n in range(len(sn_list) -1):
        if sn_list["Obj. Type"][n] == "SN Ia" and sn_list["Discovery Mag/Flux"][n] <= 20:
            cleaned_targets = cleaned_targets.append(sn_list.iloc[[n]])
            
    cleaned_targets.reset_index(inplace = True, drop=True)   
    #del cleaned_targets['index']     
    cleaned_targets.to_csv(output, index=False)
    return cleaned_targets

def produce_lygos_list(sn_list, output):
    """Saves just target names + RA/DEC values """
    correct_cols = sn_list[["Name","RA", "DEC"]].copy()
    correct_cols.to_csv(output, index=False)
    return correct_cols

def clear_list(listtoclean, listtoremove, output):
    """Clears out all sn from a list that are not Ia and brighter than 22nd magnitude
    at time of detection"""
    badrows = []
    for n in range(len(listtoclean) -1):
        for i in range(len(listtoremove)):
            if listtoclean["Name"][n].endswith(listtoremove["SNE"][i]):
                badrows.append(n)
                break
    newlist = listtoclean.copy()
    newlist.drop(badrows, inplace=True)
    newlist.to_csv(output, index=False)
    return newlist


def retrieve_all_TNS_and_NED(savepath, SN_list):
    """"For a given list of SN, retrieves the TNS information
    and the magnitude of the most likely host galaxy (nearest)
    If no Gal in NED, sets to 19 as background."""
    import tns_py as tns
    from astroquery.ned import Ned
    import astropy.units as u
    from astropy import coordinates
    import time
    
    file = savepath + "TNS_information.csv"
    with open(file, 'a') as f:
        f.write("ID,RA,DEC,TYPE,DISCDATE,DISCMAG,Z,GALMAG,GALFILTER\n")
    
    for n in range(len(SN_list)):
        name = SN_list[n]
        #print(name)
        if name.startswith("SN") or name.startswith("AT"):
            name = name[2:]
        if (name.endswith('1') or name.endswith('2') 
            or name.endswith('3') or name.endswith('4')):
            name = name[:-4]
        
        prevname = SN_list[n-1] 
        if prevname.startswith("SN") or prevname.startswith("AT"):
            prevname = prevname[2:]
        if (prevname.endswith('1') or prevname.endswith('2') 
            or prevname.endswith('3') or prevname.endswith('4')):
            prevname = prevname[:-4]
        
        #print(name)
        #print(prevname)
        
        if name == prevname:
            print("skipping duplicates")
            continue
        print(name)
        
            #print(name)
        time.sleep(3)
        RA_DEC_hr, RA_DEC_decimal, type_sn, redshift, discodate, discomag = tns.SN_page(name)
        print(RA_DEC_decimal, type_sn)
        RA, DEC = RA_DEC_decimal.split(" ")
        
        co = coordinates.SkyCoord(ra=RA, dec=DEC,
                                   unit=(u.deg, u.deg), frame='fk4')
        #constrain it to within a four pixel square
        result_table = Ned.query_region(co, radius=0.01 * u.deg) #equiox defaults to J2000
        #print(result_table)
        
        gal_mag = 19
        gal_filter = "x"
        for n in range(len(result_table)):
            if result_table[n]["Type"] == "G":
                print("Found most likely host galaxy")
                if result_table[n]["Magnitude and Filter"] != "":
                    gal_mag = float(result_table[n]["Magnitude and Filter"][:-1])
                    gal_filter = result_table[n]["Magnitude and Filter"][-1]
                    break
                else:
                    print("No recorded information about gal.mag. in NED")
                    
        print("Galaxy magnitude: ", gal_mag)
        
        with open(file, 'a') as f:
            f.write("{},{},{},{},{},{},{},{},{}\n".format(name, RA, DEC, type_sn, 
                                                          discodate, discomag, redshift,
                                                          gal_mag, gal_filter))


def limit_sector(datafolder, destinationfolder, sectorcutoff):
    """Move all files from the wrong sector (> sectorcuttof) into badfolder """
    import shutil
    #badpath = datapath + "/outofbounds/"
    for root, dirs, files in os.walk(datafolder):
            for name in files:
                if name.startswith(("rflx")):
                    label = name.split("_")
                    filepath = root + "/" + name
                    if int(label[3][0:2]) > sectorcutoff:
                        shutil.move(filepath, destinationfolder)
    return
def clear_not_on_list(pandasfile, datapath, badpath):
    """Move all lygos files that aren't on the list of tragets into
    a separate folder"""
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if name.startswith(("rflx")):
                label = name.split("_")
                filepath = root + "/" + name
                s = pandasfile["Name"]
                if not ("SN " + label[1]) in s.values:
                    shutil.move(filepath, badpath)
    return
def isolate_fit_plots(mainfolder, subfolder):
    """ scoot all fit plots into a subdir"""
    import shutil
    #badpath = datapath + "/outofbounds/"
    for root, dirs, files in os.walk(mainfolder):
            for name in files:
                if name.endswith(("powerlaw.png")):
                    
                    filepath = root + "/" + name
                    shutil.move(filepath, subfolder)
    return

def load_params(folder):
    bestparams = pd.read_csv(folder + "best_params.csv",index_col=False)
    uppererror = pd.read_csv(folder + "uppererr.csv",index_col=False)
    lowererror = pd.read_csv(folder + "lowererr.csv",index_col=False)
    sn_names = pd.read_csv(folder + "ids.csv", index_col = False)
    return bestparams, uppererror, lowererror, sn_names
############ HELPER FUNCTIONS ####################
def conv_to_abs_mag(t, i, e, galmag, z):
    """Convert apparent magnitudeto absolute magnitude using the redshift in 
    a Planck15 cosmology"""
    from astropy.cosmology import Planck15
    import astropy.units as u
    cosmo = Planck15
    d = cosmo.luminosity_distance(z).to(u.pc)
    if galmag > 19.0 or galmag is None:
        galmag = 19.0
     
    #clip all <=0.1 values to clean up array when running into the log10's
    i[i<=0.1] = np.nan
    nan_array = np.argwhere(np.isnan(i))
    i = np.delete(i, nan_array)
    t = np.delete(t, nan_array)
    e = np.delete(e, nan_array)
    #convert
    apparent_mag = -2.5* np.log10(i) + galmag
    apparent_e = -2.5* np.log10(e) + galmag
    M = apparent_mag + 5 - 5*np.log10(d.value)
    E = apparent_e + 5 - 5*np.log10(d.value)
    return t,M, E

def convert_all_to_abs_mag(allt, alli, alle, info, all_labels, gal_mags):
    ret_t = allt
    ret_i = alli
    ret_e = alle
    for n in range(len(alli)):
        key = all_labels[n][:-4]
        #dig z values out
        z_table = info[info['ID'].str.contains(key)]
        z_table.reset_index(inplace=True)
        for i in range(len(z_table)): #in caseyou have like 2020kt and 2020kte
            if z_table['ID'][i] == key:
                z = z_table['Z'][i]
        ret_t[n], ret_i[n], ret_e[n] = conv_to_abs_mag(allt[n],alli[n], alle[n], gal_mags[key], z)
    return ret_t, ret_i, ret_e


def preclean_mcmc(file):
    """ opens the file and cleans the data for use in MCMC modeling"""
    #load data
    t,ints,error = du.load_lygos_csv(file)

    #sigma clip - set outliers to the mean value because idk how to remove them
    #is sigma clipping fucking up quaternions?
    mean = np.mean(ints)
    
    sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
    clipped_inds = np.nonzero(np.ma.getmask(sigclip(ints)))
    ints[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
    
    t_sub = t - t.min() #put the start at 0 for better curve fitting
    
    #ints = df.normalize(ints, axis=0)
    
    return t_sub, ints, error, t.min()


def crop_to_40(t, y, err):
    """ only fit first 40% of brightness of curve"""
    brightness40 = (y.max() - y.min()) * 0.4

    for n in range(len(y)):
        if y[n] > brightness40:
            cutoffindex = n
            break
                
    t_40 = t[0:cutoffindex]
    ints_40 = y[0:cutoffindex]
    err_40 = err[0:cutoffindex]
    return t_40, ints_40, err_40

def lists_by_folder(savepath, listToUse, listToCheck, batchname):
    re_run_list = []

    for n in range(len(listToCheck)):
        key = listToCheck['Name'][n][3:]
        for k in range(len(listToUse)):
            if listToUse[k][:-8] == key:
                re_run_list.append(n)
                break
    print("found", len(re_run_list))
    p = len(listToCheck)
    ally = np.arange(0,p,1)
    ally = np.delete(ally, re_run_list) 
    batch = listToCheck.drop(ally, inplace=False)
    batch.to_csv(savepath + batchname + ".csv")    


def plot_chain(path, targetlabel, plotlabel, samples, labels, ndim):
    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    #samples = sampler.get_chain()
    labels = labels
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)
    
    axes[-1].set_xlabel("step number");
    plt.savefig(path + targetlabel+ plotlabel)
    plt.show()
    return

def bin_8_hours(t, i, e):
    n_points = 16
    binned_t = []
    binned_i = []
    binned_e = []
    n = 0
    m = n_points
        
    while m <= len(t):
        bin_t = t[n + 8] #get the midpoint of this data as the point to plot at
        binned_t.append(bin_t) #put into new array
        bin_i = np.mean(i[n:m]) #bin the stretch
        binned_i.append(bin_i) #put into new array
        bin_e = np.sqrt(np.sum(e[n:m]**2)) / n_points #error propagates as sqrt(sum of squares of error)
        binned_e.append(bin_e)
            
        n+= n_points
        m+= n_points
        
    return np.asarray(binned_t), np.asarray(binned_i), np.asarray(binned_e)

def get_SN_IDs_in_LC_folder(folder):
    """For a given folder of light curves, strip identifiers from the filenames """
    listy = []
    for root, dirs, files in os.walk(folder):
            for name in files:
                #print(name)
                labels = name.split('-')
                listy.append(labels[0])
    return listy

def retrieve_quaternions_bigfiles(savepath, quaternion_folder, sector, x):
        
    for root, dirs, files in os.walk(quaternion_folder):
        for name in files:
            if name.endswith(("sector"+sector+"-quat.fits")):
                filepath = root + "/" + name
                
    tQ, Q1, Q2, Q3, outliers = du.extract_smooth_quaterions(savepath, filepath, None, 
                                           31, x)
    return tQ, Q1, Q2, Q3, outliers 

def produce_discovery_time_dictionary(all_labels, info, t_starts):
    """ returns the discovery time MINUS the start time of the sector"""
    discovery_dictionary = {} #initialize the dictionary
    #print("creating discovery dictionary")
    from astropy.time import Time #import time module
    for n in range(len(all_labels)): #for every label in the list
        key = all_labels[n][:-4] #cuts off last four digits of sector/cam
        sectorstart = t_starts[all_labels[n]] #get start date for the target
        df1 = info[info['ID'].str.contains(key)]
        df1.reset_index(inplace=True)
        for i in range(len(df1)): # there are sometimes multiple thingies w/ the same key
            if df1["ID"][i] == key:
                discotime = Time(df1["DISCDATE"][i], format = 'iso', scale='utc')
                discotime = discotime.jd - sectorstart
                discovery_dictionary[all_labels[n]] = discotime #this is Nth not Ith!!
                
        del df1          
    return discovery_dictionary

def produce_gal_mag_dictionary(info):
    gal_mag_dict = {}
    for n in range(len(info)):
        gal_mag_dict[info["ID"][n]] = info["GALMAG"][n]  
    return gal_mag_dict
        

def generate_clip_quats_cbvs(sector, x, y, yerr, targetlabel, CBV_folder):
    tQ, Q1, Q2, Q3, outliers = du.metafile_load_smooth_quaternions(sector, x)
    Qall = Q1 + Q2 + Q3
    #load CBVs
    camera = targetlabel[-2]
    ccd = targetlabel[-1]
    cbv_file = CBV_folder + "s00{sector}/cbv_components_s00{sector}_000{camera}_000{ccd}.txt".format(sector = sector,
                                                                                          camera = camera,
                                                                                          ccd = ccd)
    cbvs = np.genfromtxt(cbv_file)
    CBV1 = cbvs[:,0]
    CBV2 = cbvs[:,1]
    CBV3 = cbvs[:,2]
    #correct length differences:
    lengths = np.array((len(x), len(tQ), len(CBV1)))
    length_corr = lengths.min()
    x = x[:length_corr]
    y = y[:length_corr]
    yerr = yerr[:length_corr]
    tQ = tQ[:length_corr]
    Qall = Qall[:length_corr]
    CBV1 = CBV1[:length_corr]
    CBV2 = CBV2[:length_corr]
    CBV3 = CBV3[:length_corr]
    return x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3



############### BAYESIAN CURVE FITS ##############
def mcmc_load_lygos(datapath, savepath, runproduce = False):
    """ Opens all Lygos files and loads them in.
    label_use = 1: filenames like: rflxtarg_2018eod_0114_30mn.csv
    label_use = 2: filename like: rflxtarg_SN2018enj_n011_1312_30mn_m001.csv"""
    
    all_t = [] 
    all_i = []
    all_e = []
    all_labels = []
    sector_list = []
    discovery_dictionary = {}
    t_starts = {}
    
        
    
    infofile = datapath + "TNS_information.csv"

    if runproduce:
        sn_names = []
    
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if name.startswith(("rflx")):
                filepath = root + "/" + name
                label = name.split("_")
                
                if len(label) == 4:
                    full_label = label[1]+label[2]
                    sector = label[2][0:2]
                elif len(label) == 6:
                    full_label = label[1][2:]+label[3]
                    sector = label[3][0:2]
                
                
                
                if int(sector) >= 27:
                    print("Sector out of bounds")
                    continue
                    #skip to next one
                else:
                    all_labels.append(full_label)
                    sector_list.append(sector)
                    
                    t,i,e, t_start = preclean_mcmc(filepath)
                    t_starts[full_label] = t_start
                    
                    all_t.append(t)
                    all_i.append(i)
                    all_e.append(e)
                    
                    if runproduce:
                        #print(label[1])
                        sn_names.append(label[1])
    
    if runproduce:
        #print(sn_names[0:10])
        
        retrieve_all_TNS_and_NED(datapath, sn_names) 
    
    info = pd.read_csv(infofile)
    discovery_dictionary = produce_discovery_time_dictionary(all_labels, info, t_starts)
    gal_mags = produce_gal_mag_dictionary(info)              
    return all_t, all_i, all_e, all_labels, sector_list, discovery_dictionary, t_starts, gal_mags, info


def stepped_powerlaw(path, targetlabel, t, intensity, error, sector,
                                  discovery_times, t_starts, best_params_file,
                                  ID_file, upper_errors_file, lower_errors_file,plot = True, 
                                 quaternion_folder = "/users/conta/urop/quaternions/", 
                                 CBV_folder = "C:/Users/conta/.eleanor/metadata/"):
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
        need to add in cQ and CBVs!!
        only fit up to 40% of the flux"""
        t0, A, beta, B, cQ, cbv1, cbv2, cbv3 = theta #, cQ, cbv1, cbv2, cbv3
        #print(A, beta, B)
        t1 = x - t0
        model = (np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta)) + B + 
                 cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3)
        
        yerr2 = yerr**2.0
        returnval = -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        return returnval
    
    def log_prior(theta, disctime):
        """ calculates the log prior value """
        t0, A, beta, B, cQ, cbv1, cbv2, cbv3 = theta #, cQ, cbv1, cbv2, cbv3
        #print(A, beta, B, cQ, cbv1, cbv2, cbv3)
        if ((disctime - 2) < t0 < (disctime +2) and 0.5 < beta < 6.0 
            and 0.0 < A < 5.0 and -10 < B < 10 and (-100 > cQ > -800 ) 
            and -5000 < cbv1 < 5000 and -5000 < cbv2 < 5000 
            and -5000 < cbv3 < 5000):
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
    
    #load quaternions and CBVs
    x,y,yerr, tQ, Qall, CBV1, CBV2, CBV3 = generate_clip_quats_cbvs(sector, x, y, yerr,targetlabel, CBV_folder)
        
    
    #running MCMC
    np.random.seed(42)   
    nwalkers = 32
    ndim = 8
    labels = ["t0", "A", "beta", "B", "cQ", "cbv1", "cbv2", "cbv3"] #, "cQ", "cbv1", "cbv2", "cbv3"
    
    
    p0 = np.zeros((nwalkers, ndim)) 
    for n in range(len(p0)):
        p0[n] = np.array((disctime, 0.1, 1.3, 0.8, 0, 0, 0, 0)) #mean values from before

    k = np.array((0.1,0.1,0.1,0.1, 500,500,500,500)) * np.random.rand(nwalkers,ndim)
    p0 += k
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr, disctime))
    
   # run ONCE
    sampler.run_mcmc(p0, 20000, progress=True)
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
    sampler.run_mcmc(p0,40000, progress = True)
    if plot:
        samples = sampler.get_chain()
        plot_chain(path, targetlabel, "-burn-in-plot-final.png", samples[20000:], labels, ndim)
    
    flat_samples = sampler.get_chain(discard=6000, thin=15, flat=True)

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
        cQ = best_mcmc[0][4]
        cbv1 = best_mcmc[0][5]
        cbv2 = best_mcmc[0][6]
        cbv3 = best_mcmc[0][7]
        
        best_fit_model = np.heaviside((t1), 1) * A *np.nan_to_num((t1**beta), copy=False) + B + cQ * Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
        
        nrows = 3
        ncols = 1
        fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                       figsize=(8*ncols * 2, 3*nrows * 2))
        
        ax[0].scatter(x, best_fit_model, label="best fit model", s = 5, color = 'red')
        ax[0].scatter(x, y, label = "FFI data", s = 2, color = 'gray')
        for n in range(nrows):
            ax[n].axvline(best_mcmc[0][0], color = 'blue', label="t0")
            ax[n].axvline(disctime, color = 'green', label="discovery time")
            ax[n].axvline(disctime - 2, color='pink', label="lower t0 prior")
            ax[n].axvline(disctime + 2, color='pink', ls= 'dashed',label='upper t0 prior')
            ax[n].set_ylabel("Rel. Flux")
            
        #main
        ax[0].set_title(targetlabel)
        ax[0].legend(fontsize=8, loc="upper left")
        ax[nrows-1].set_xlabel("BJD-" + str(t_starts[targetlabel]))
        
        #residuals
        ax[1].set_title("Residual (y-model)")
        residuals = y - best_fit_model
        ax[1].scatter(x,residuals, s=2, color = 'red')
        
        #CBV fits\
        ax[2].set_title("Residuals (y-corrective terms)")
        corrections = cQ*Qall + cbv1 * CBV1 + cbv2 * CBV2 + cbv3 * CBV3
        correction_residuals = y-corrections
        ax[2].scatter(x, correction_residuals)
        
        plt.savefig(path + targetlabel + "-MCMCmodel-stepped-powerlaw.png")
                
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
    return best_mcmc, upper_error, lower_error

def run_all_discdate(path, all_t, all_i, all_e, all_labels, sector_list, 
                        discovery_dictionary, t_starts, best_params_file,
                        ID_file, upper_errors_file, lower_errors_file,
                        quaternion_folder = "/users/conta/urop/quaternions/", 
                        CBV_folder = "C:/Users/conta/.eleanor/metadata/"):
    

    with open(best_params_file, 'a') as f:
        f.write("t0,A,beta,B,cQ,CBV1,CBV2,CBV3\n")
    with open(upper_errors_file, 'a') as f:
        f.write("t0,A,beta,B,cQ,CBV1,CBV2,CBV3\n")
    with open(lower_errors_file, 'a') as f:
        f.write("t0,A,beta,B,cQ,CBV1,CBV2,CBV3\n")
    with open(ID_file, 'a') as p:
        p.write("ID\n")
    
    for n in range(len(all_labels)):
        key = all_labels[n]
        if -1 <= discovery_dictionary[key] <= 30:
            stepped_powerlaw(path, all_labels[n], all_t[n], all_i[n],
                                          all_e[n], sector_list[n], discovery_dictionary, 
                                          t_starts, best_params_file,ID_file, 
                                          upper_errors_file, lower_errors_file,plot = True, 
                                         quaternion_folder = "/users/conta/urop/quaternions/", 
                                         CBV_folder = "C:/Users/conta/.eleanor/metadata/")

    return

def retry_endofsectorlist(path, use_labels, all_t, all_i, all_e, all_labels, sector_list, 
                        discovery_dictionary, t_starts, best_params_file,
                        ID_file, upper_errors_file, lower_errors_file,
                        quaternion_folder = "/users/conta/urop/quaternions/", 
                        CBV_folder = "C:/Users/conta/.eleanor/metadata/"):
    
    no_data = []
    with open(best_params_file, 'a') as f:
        f.write("t0,A,beta,B,cQ,CBV1,CBV2,CBV3\n")
    with open(upper_errors_file, 'a') as f:
        f.write("t0,A,beta,B,cQ,CBV1,CBV2,CBV3\n")
    with open(lower_errors_file, 'a') as f:
        f.write("t0,A,beta,B,cQ,CBV1,CBV2,CBV3\n")
    with open(ID_file, 'a') as p:
        p.write("ID\n")
    
    for i in range(len(use_labels)):
        id_ = use_labels[i] #full string for one to look for
        try:
            n = all_labels.index(id_) #position in big list
            s = int(use_labels[i][-4:-2]) + 1 #get current sector + 1
            #if the next label in the list 
            if (all_labels[n+1].startswith(id_[:-4] + str(s))): #if the next label is the next sector
                n = n+1
                #key = all_labels[n]
                sn.stepped_powerlaw(path, all_labels[n], all_t[n], all_i[n],
                                     all_e[n], sector_list[n], discovery_dictionary, 
                                     t_starts, best_params_file,ID_file, 
                                     upper_errors_file, lower_errors_file,plot = True, 
                                    quaternion_folder = "/users/conta/urop/quaternions/", 
                                    CBV_folder = "C:/Users/conta/.eleanor/metadata/")
            else: #next one is not hte next sector
                no_data.append(id_)
                print('nothing on', id_)
        except ValueError:
            print("not in list")

    return no_data