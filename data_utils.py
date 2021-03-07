# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 18:00:13 2021

@author: Emma Chickles, Lindsey Gordon
data_utils.py
* data load
* data cleaning
* feature loading

To Do List:
    - HYPERLEDA LC load in
    - Loading fxns for CAE features
"""
import numpy as np
from __init__ import *


####### DATA LOADING (FFI) ##########

def load_lygos_csv(file):
    import pandas as pd
    data = pd.read_csv(file, sep = ' ', header = None)
    #print (data)
    t = np.asarray(data[0])
    ints = np.asarray(data[1])
    error = np.asarray(data[2])
    return t, ints, error

def load_all_lygos(datapath):
    
    all_t = [] 
    all_i = []
    all_e = []
    all_labels = []
    
    for root, dirs, files in os.walk(datapath):
        for name in files:
            if name.startswith(("rflx")):
                filepath = root + "/" + name 
                print(name)
                label = name.split("_")
                full_label = label[1] + label[2]
                all_labels.append(full_label)
                
                t,i,e = load_lygos_csv(name)
                mean = np.mean(i)
                sigclip = SigmaClip(sigma=4, maxiters=None, cenfunc='median')
                clipped_inds = np.nonzero(np.ma.getmask(sigclip(i)))
                i[clipped_inds] = mean #reset those values to the mean value (or remove??)
    
                all_t.append(t)
                all_i.append(i)
                all_e.append(e)
                
    return all_t, all_i, all_e, all_labels



######## DATA CLEANING ###########

def normalize(flux, axis=1):
    '''Dividing by median.
    !!Current method blows points out of proportion if the median is too close to 0?'''
    medians = np.median(flux, axis = axis, keepdims=True)
    flux = flux / medians
    return flux

def mean_norm(flux, axis=1): 
    """ normalizes by dividing by mean - necessary for TLS running 
    modified lcg 07192020"""
    means = np.mean(flux, axis = axis, keepdims=True)
    flux = flux / means
    return flux

def rms(x, axis=1):
    rms = np.sqrt(np.nanmean(x**2, axis = axis))
    return rms

def standardize(x, ax=1):
    means = np.nanmean(x, axis = ax, keepdims=True) # >> subtract mean
    x = x - means
    stdevs = np.nanstd(x, axis = ax, keepdims=True) # >> divide by standard dev
    
    # >> avoid dividing by 0.0
    stdevs[ np.nonzero(stdevs == 0.) ] = 1e-8
    
    x = x / stdevs
    return x



######### QUATERNION HANDLING ##########
def convert_to_quat_metafile(file, fileoutput):
    f = fits.open(file, memmap=False)
    
    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    f.close()
    
    big_quat_array = np.asarray((t, Q1, Q2, Q3))
    np.savetxt(fileoutput, big_quat_array)

def metafile_load_smooth_quaternions(sector, maintimeaxis, 
                                     quaternion_folder = "/users/conta/urop/quaternions/"):
    
    def quaternion_binning(quaternion_t, q, maintimeaxis):
        sector_start = maintimeaxis[0]
        bins = 900 #30 min times sixty seconds/2 second cadence
                
        def find_nearest_values_index(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return idx
                
        binning_start = find_nearest_values_index(quaternion_t, sector_start)
        n = binning_start
        m = n + bins
        binned_Q = []
        binned_t = []
                
        while m <= len(quaternion_t):
            bin_t = quaternion_t[n]
            binned_t.append(bin_t)
            bin_q = np.mean(q[n:m])
            binned_Q.append(bin_q)
            n += 900
            m += 900
                
            
        standard_dev = np.std(np.asarray(binned_Q))
        mean_Q = np.mean(binned_Q)
        outlier_indexes = []
                
        for n in range(len(binned_Q)):
            if binned_Q[n] >= mean_Q + 5*standard_dev or binned_Q[n] <= mean_Q - 5*standard_dev:
                outlier_indexes.append(n)
                
                      
        return np.asarray(binned_t), np.asarray(binned_Q), outlier_indexes
        
    from scipy.signal import medfilt
    for root, dirs, files in os.walk(quaternion_folder):
            for name in files:
                if name.endswith(("S"+sector+"-quat.txt")):
                    print(name)
                    filepath = root + "/" + name
                    c = np.genfromtxt(filepath)
                    tQ = c[0]
                    Q1 = c[1]
                    Q2 = c[2]
                    Q3 = c[3]

    q = [Q1, Q2, Q3]

    
    for n in range(3):
        smoothed = medfilt(q[n], kernel_size = 31)
        if n == 0:
            Q1 = smoothed
            tQ_, Q1, Q1_outliers = quaternion_binning(tQ, Q1, maintimeaxis)
        elif n == 1:
            Q2 = smoothed
            tQ_, Q2, Q2_outliers = quaternion_binning(tQ, Q2, maintimeaxis)
        elif n == 2:
            Q3 = smoothed
            tQ_, Q3, Q3_outliers = quaternion_binning(tQ, Q3, maintimeaxis)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    return tQ_, Q1, Q2, Q3, outlier_indexes  

def extract_smooth_quaterions(path, file, momentum_dump_csv, kernal, maintimeaxis, plot = False):

    from scipy.signal import medfilt
    f = fits.open(file, memmap=False)

    t = f[1].data['TIME']
    Q1 = f[1].data['C1_Q1']
    Q2 = f[1].data['C1_Q2']
    Q3 = f[1].data['C1_Q3']
    
    f.close()
    
    
    q = [Q1, Q2, Q3]
    
    if plot:
        with open(momentum_dump_csv, 'r') as f:
            lines = f.readlines()
            mom_dumps = [ float(line.split()[3][:-1]) for line in lines[6:] ]
            inds = np.nonzero((mom_dumps >= np.min(t)) * \
                              (mom_dumps <= np.max(t)))
            mom_dumps = np.array(mom_dumps)[inds]
    #q is a list of qs
    for n in range(3):
        
        smoothed = medfilt(q[n], kernel_size = kernal)
        
        if plot:
            plt.scatter(t, q[n], label = "original")
            plt.scatter(t, smoothed, label = "smoothed")
            
            for k in mom_dumps:
                plt.axvline(k, color='g', linestyle='--', alpha = 0.1)
            plt.legend(loc = "upper left")
            plt.title("Q" + str(n+1))
            plt.savefig(path + str(n + 1) + "-kernal-" + str(kernal) +"-both.png")
            plt.show()
            #plt.scatter(t, q[n], label = "original")
            plt.scatter(t, smoothed, label = "smoothed")
            for k in mom_dumps:
                plt.axvline(k, color='g', linestyle='--', alpha = 0.1)
            plt.legend(loc="upper left")
            plt.title("Q" + str(n+1) + "Smoothed")
            plt.savefig(path + str(n + 1) + "-kernal-" + str(kernal) +"-median-smoothed-only.png")
            plt.show()
            
        def quaternion_binning(quaternion_t, q, maintimeaxis):
            sector_start = maintimeaxis[0]
            bins = 900 #30 min times sixty seconds/2 second cadence
            
            def find_nearest_values_index(array, value):
                array = np.asarray(array)
                idx = (np.abs(array - value)).argmin()
                return idx
            binning_start = find_nearest_values_index(quaternion_t, sector_start)
            n = binning_start
            m = n + bins
            binned_Q = []
            binned_t = []
            
            while m <= len(t):
                bin_t = quaternion_t[n]
                binned_t.append(bin_t)
                bin_q = np.mean(q[n:m])
                binned_Q.append(bin_q)
                n += 900
                m += 900
            plt.scatter(binned_t, binned_Q)
            plt.show()
        
            standard_dev = np.std(np.asarray(binned_Q))
            mean_Q = np.mean(binned_Q)
            outlier_indexes = []
            
            for n in range(len(binned_Q)):
                if binned_Q[n] >= mean_Q + 5*standard_dev or binned_Q[n] <= mean_Q - 5*standard_dev:
                    outlier_indexes.append(n)
            
            print(outlier_indexes)      
            return outlier_indexes
        
        if n == 0:
            Q1 = smoothed
            Q1_outliers = quaternion_binning(t, Q1, maintimeaxis)
        elif n == 1:
            Q2 = smoothed
            Q2_outliers = quaternion_binning(t, Q2, maintimeaxis)
        elif n == 2:
            Q3 = smoothed
            Q3_outliers = quaternion_binning(t, Q3, maintimeaxis)
    
    outlier_indexes = np.unique(np.concatenate((Q1_outliers, Q2_outliers, Q3_outliers)))
    print(outlier_indexes)
    return t, Q1, Q2, Q3, outlier_indexes  



            









