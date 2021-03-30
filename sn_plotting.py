# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:47:01 2021

@author: conta

SN Plotting
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_histogram(data, bins, x_label, filename):
    """ 
    Plot a histogram with one light curve from each bin plotted on top
    * Data is the histogram data
    * Bins is bins for the histogram
    * x_label for the x-axis of the histogram
    * filename is the exact place you want it saved
    """
    fig, ax1 = plt.subplots()
    n_in, bins, patches = ax1.hist(data, bins)
    
    y_range = np.abs(n_in.max() - n_in.min())
    x_range = np.abs(data.max() - data.min())
    ax1.set_ylabel('Number of light curves')
    ax1.set_xlabel(x_label)
    
    plt.savefig(filename)
    plt.close()
    
def plot_beta_redshift(savepath, info, sn_names, bestparams):
    for n in range(len(bestparams)):
        target = sn_names['ID'][n][:-4]
        #get z where the name matches in info??
        beta = bestparams['beta'][n]
        df1 = info[info['ID'].str.contains(target)]
        df1.reset_index(inplace=True)
        for i in range(len(df1)): #AHHHHH so there are sometimes multiple thingies w/ the same key
            if df1["ID"][i] == target:
                redshift = df1['Z'][i]
                
        plt.scatter(redshift, beta)
        
    
    plt.xlabel('redshift')
    plt.ylabel('beta value')
    plt.title("Plotting " +  r'$\beta$' + " versus redshift for Ia SNe")   
    plt.savefig(savepath + "redshift-beta.png") 
    
def plot_absmag(t,i, xlabel='',ylabel='', title='',savepath=None):
    fig, ax =plt.subplots()
    ax.scatter(t,i)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if savepath is not None:
        plt.savefig(savepath)