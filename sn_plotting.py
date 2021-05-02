# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:47:01 2021

@author: conta

SN Plotting
"""
import matplotlib.pyplot as plt
import numpy as np

def plot_mcmc(path, x, y, targetlabel, disctime, best_mcmc, flat_samples,
              labels):
    import corner
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles = [0.16, 0.5, 0.84],
                       show_titles=True,title_fmt = ".4f", 
                       title_kwargs={"fontsize": 12}
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
    ax[nrows-1].set_xlabel("BJD-2457000")
    
    #residuals
    ax[1].set_title("Residual (y-model)")
    residuals = y - best_fit_model
    ax[1].scatter(x,residuals, s=5, color = 'black', label='residual')
    ax[1].axhline(0,color='purple', label="zero")
    ax[1].legend()
    
    plt.savefig(path + targetlabel + "-MCMCmodel-stepped-powerlaw.png")
    return

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
        
def quicklook_plotall(path, all_t, all_i, all_labels, discovery_dictionary):
    """Plot all in the list and save plots into dedicated folder
    allows for quick flip thru them all to get rid of gunk. """
    from pylab import rcParams
    rcParams['figure.figsize'] = 8,3
    for n in range(len(all_labels)):
        key = all_labels[n]
        if -3 <= discovery_dictionary[key] <= 30:
            plt.scatter(all_t[n], all_i[n])
            plt.axvline(discovery_dictionary[key])
            plt.title(all_labels[n])
            plt.savefig(path + all_labels[n] + "-.png")
            plt.close()
            
def print_table_formatting(best,upper,lower):
    for n in range(len(best[0])):
        print("param ${:.4f}".format(best[0][n]), "^{:.4f}".format(upper[0][n]),
              "_{:.4f}$".format(lower[0][n]))
        
def plot_SN_LCs(path, t,i,e,label,sector,galmag,extinction, z, 
                discdate, badIndexes):

    import sn_functions as sn
    
    nrows = 4
    ncols = 1
    fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                  figsize=(8*ncols * 2, 3*nrows * 2))
    
    #row 0: raw lygos light curve
    if badIndexes is None:
        cPlot = 'green'
        ecoll = 'springgreen'
        binT, binI, binE = sn.bin_8_hours_TIE(t,i,e) #bin to 8 hours
    else:
        cPlot = 'yellow'
        ecoll = 'yellow'
        cT, cI, cE = sn.clip_TIE(badIndexes, t,i,e) #clip out designated indices
        ax[0].errorbar(cT, cI, cE, fmt = 'o', label = "Lygos (Clipped)", 
                       color = 'green',
                       ecolor = 'springgreen', zorder=2)
        binT, binI, binE = sn.bin_8_hours_TIE(cT, cI, cE) #bin to 8 hours
        
    ax[0].errorbar(t,i,yerr=e, fmt = 'o', label = "Lygos (Raw)", color = cPlot,
                   ecolor=ecoll, zorder=1)
    
    #ax[0].axhline(1, color='orchid', label='Lygos Background')
    ax[0].set_ylabel("Rel. Flux")
    ax[0].set_title(label)
    
    #row 1: binned and cleaned up flux.
    ax[1].set_title("Binned Flux")
    ax[1].errorbar(binT, binI, yerr=binE, fmt = 'o', label = "Binned and Cleaned",
                   color = 'blue', ecolor = "blue", markersize = 5)
    ax[1].set_ylabel("Rel. Flux")
    
    #row 2: apparent TESS magnitude
    (absT, absI, absE, absGalmag,
     d, apparentM, apparentE) = sn.conv_to_abs_mag(binT, binI, binE , galmag, z,
                                                       extinction = extinction)
    
    
    ax[2].errorbar(absT, apparentM, yerr=apparentE, fmt = 'o', 
                   color = 'darkslateblue', ecolor='slateblue', markersize=5)
    ax[2].set_title("Apparent TESS Magnitude")
    ax[2].set_ylabel("Apparent Mag.")
    ax[2].invert_yaxis()
    
    #row 3: absolute magntiude conversion
    
    ax[3].errorbar(absT, absI, yerr = absE, 
                   fmt = 'o',label="abs mag",  color = 'purple',
                   ecolor='lavender', markersize=5)
    #ax[3].axhline(absGalmag, color = 'orchid',label="background mag." )
    ax[3].invert_yaxis()
    ax[3].set_title("Absolute Magnitude Converted")
    ax[3].set_ylabel("Abs. Magnitude")
    
    for i in range(nrows):
        ax[i].axvline(discdate, color = 'black', 
                      label="discovery time")
        
        ax[i].legend(loc="upper left")
        
        
    ax[nrows-1].set_xlabel("BJD-2457000")
    
    plt.tight_layout()
    plt.savefig(path + label + "flux-plot.png")
    plt.show()
    #plt.close()
    return binT, binI, binE, absT, absI, absE