# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 19:15:09 2021

@author: conta
"""
def stepped_powerlaw_basic(path, targetlabel, t, intensity, error, sector,
                     disctime, plot = True, n1=20000, n2=40000):
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
            - sector number 
            - discovery time
            - plot (true/false)
            - n1 = steps in first chain
            - n2 = steps in second chain
    
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
        if ((disctime - 10) < t0 < (disctime-1) and 0.7 < beta < 4.0 
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
    
    #running MCMC
    np.random.seed(42)   
    nwalkers = 32
    ndim = 4
    labels = ["t0", "A", "beta",  "B"]
    
    p0 = np.zeros((nwalkers, ndim)) 
    for n in range(len(p0)):
        p0[n] = np.array((disctime-3, 0.1, 1.8, 1)) #mean values from before

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
        import sn_plotting as sp
        sp.plot_mcmc(path, x, y, targetlabel, disctime, best_mcmc, flat_samples,
                     labels)
        
    
    sn.beep()
    return flat_samples, best_mcmc, upper_error, lower_error



      #%%
import sn_functions as sn
import sn_plotting as sp
import numpy as np
import data_utils as du


def load_2018koy():
    datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/'
    savepath = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/"
    (t,i,e, label, sector, 
     discdate, gal_mags, info) = sn.mcmc_load_one(datapath, savepath, 
                                              runproduce = False)
    galmag = 17
    extinction = 0.314
    z = info["Z"][0]
    badIndexes = np.arange(415,440) #2018koy
    binT, binI, binE, absT, absI, absE = sp.plot_SN_LCs(savepath, t,i,e,label,
                                                    sector,galmag,extinction,
                                                    z, discdate, badIndexes)

    #standardize to 1
    standI, standE = du.flux_standardized(binI, binE)
    return (datapath, savepath, t,i,e, label, sector, discdate, gal_mags, 
            info, binT, binI, binE, absT, absI, absE, standI, standE)
 
 
def load_2019yft():
    datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2019yft/'
    savepath = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2019yft/"
    (t,i,e, label, sector, 
     discdate, gal_mags, info) = sn.mcmc_load_one(datapath, savepath, 
                                              runproduce = False)
    galmag = 17.1
    extinction = 0.069
    z = info["Z"][0]
    badIndexes = None  #2019yft
    binT, binI, binE, absT, absI, absE = sp.plot_SN_LCs(savepath, t,i,e,label,
                                                    sector,galmag,extinction,
                                                    z, discdate, badIndexes)

    #standardize to 1
    standI, standE = du.flux_standardized(binI, binE)
    return (datapath, savepath, t,i,e, label, sector, discdate, gal_mags, 
            info, binT, binI, binE, absT, absI, absE, standI, standE)   

def load_2020chi():
    datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2020chi/'
    savepath = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2020chi/"
    (t,i,e, label, sector, 
     discdate, gal_mags, info) = sn.mcmc_load_one(datapath, savepath, 
                                              runproduce = False)
    galmag = 20.44
    extinction = 0.051
    z = info["Z"][0]
    badIndexes = np.concatenate((np.arange(906,912), np.arange(645,665)))
    binT, binI, binE, absT, absI, absE = sp.plot_SN_LCs(savepath, t,i,e,label,
                                                    sector,galmag,extinction,
                                                    z, discdate, badIndexes)

    #standardize to 1
    standI, standE = du.flux_standardized(binI, binE)
    return (datapath, savepath,t,i,e, label, sector, discdate, gal_mags, 
            info, binT, binI, binE, absT, absI, absE, standI, standE) 

def load_2020efe():
    datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2020efe/'
    savepath = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2020efe/"
    (t,i,e, label, sector, 
     discdate, gal_mags, info) = sn.mcmc_load_one(datapath, savepath, 
                                              runproduce = False)
    galmag = 20.44
    extinction = 0.034
    z = info["Z"][0]
    badIndexes = np.concatenate((np.arange(0,100), np.arange(628,641)))
    binT, binI, binE, absT, absI, absE = sp.plot_SN_LCs(savepath, t,i,e,label,
                                                    sector,galmag,extinction,
                                                    z, discdate, badIndexes)

    #standardize to 1
    standI, standE = du.flux_standardized(binI, binE)
    return (datapath, savepath,t,i,e, label, sector, discdate, gal_mags, 
            info, binT, binI, binE, absT, absI, absE, standI, standE) 


#%% load whatever you are using
(datapath, savepath,t,i,e, label, sector, discdate, gal_mags, 
 info, binT, binI, binE, absT, absI, absE, standI, standE) = load_2020efe()




#%% fULL light curve

(samples, best, 
 upper, lower) = stepped_powerlaw_basic(savepath, label, binT, 
                                        standI, standE, sector,
                                        discdate, plot = True, n1=40000, 
                                        n2=60000)

sp.print_table_formatting(best,upper,lower)
#%% PARTIAL 2
binT60, binI60, binE60 = sn.crop_to_percent(binT, standI,standE, 0.4)
plt.scatter(binT60, binI60)
#%%
savepath = datapath + "40/"
samps, best60, upper60, lower60 = stepped_powerlaw_basic(savepath, label, 
                                                  binT60, binI60, binE60, 
                                                  sector,discdate, 
                                                  plot = True, n1=40000, n2=60000)

sp.print_table_formatting(best60,upper60,lower60)

#%%

def double_powerlaw(path, targetlabel, t, intensity, error, sector,
                     disctime, plot = True, n1=20000, n2=40000):
    """ Runs MCMC fitting for stepped power law fit
    This is the fitting that matches: Shappee 2019
    fireball power law with A, beta, B, and t0 floated
    h(t-t_1)^a1 + B from t1 <= t <= t2
    h(t - t_1)^a1 + h(t-t_2)^a2 + B from t_2 <= t
    Runs two separate chains to hopefully hit convergence
    Params:
            - path to save into
            - targetlabel for file names
            - time axis
            - intensities
            - errors
            - sector number 
            - discovery time
            - plot (true/false)
            - n1 = steps in first chain
            - n2 = steps in second chain
    
    """
    def func1(x, t1, t2, a1, a2, B1, B2):
        return B1 *(x-t1)**a1
    def func2(x, t1, t2, a1, a2, B1, B2):
        return B1 * (x-t1)**a1 + B2 * (x-t2)**a2
    
    def log_likelihood(theta, x, y, yerr):
        """ calculates the log likelihood function. 
        constrain beta between 0.5 and 4.0
        A is positive
        only fit up to 40% of the flux"""
        #t1, t2,a1, a2, B1, B2, C = theta 
        a1, a2, B1, B2, C = theta 
        t1 = 1917.01641
        t2 = 1919.68304
    
        model = np.piecewise(x, [(t1 <= x)*(x < t2), t2 <= x], 
                             [func1, func2],
                             t1, t2, a1, a2, B1, B2) + C
       
        
        yerr2 = yerr**2.0
        returnval = -0.5 * np.nansum((y - model) ** 2 / yerr2 + np.log(yerr2))
        return returnval
    
    def log_prior(theta, disctime):
        """ calculates the log prior value """
        #t1, t2, a1, a2, B1, B2, C = theta 
        a1, a2, B1, B2, C = theta 
        if ( #(disctime-3)<t1<(disctime+5) and t1 < t2 < (disctime+6) and
            0.5 < a1 < 6.0 and 0.5 < a2 < 6.0
            and 1 > B1 > 0 and 1 > B2 > 0 and -5 < C < 5):
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
    
    #running MCMC
    np.random.seed(42)   
    nwalkers = 32
    #ndim = 7
    #labels = ["t1", "t2", "a1", "a2", "B1", "B2", "C"] # YYY
    ndim = 5
    labels = ["a1", "a2", "B1", "B2", "C"]
    
    
    p0 = np.zeros((nwalkers, ndim)) 
    for n in range(len(p0)):
        #p0[n] = np.array((disctime, disctime, 1, 1, 0.2, 0.2, 0.2)) 
        p0[n] = np.array(( 1, 1, 0.2, 0.2, 0.2)) 

    #p0 += (np.array((0.1,0.1,0.1, 0.1, 0.1, 0.1,0.1)) * np.random.rand(nwalkers,ndim))
    p0 += (np.array((0.1, 0.1, 0.1, 0.1,0.1)) * np.random.rand(nwalkers,ndim))
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                    args=(x, y, yerr, disctime))
    
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
        #t1 = best_mcmc[0][0] 
        t1 = 1917.01641
        #t2 = best_mcmc[0][1]
        t2 = 1919.68304
        a1 = best_mcmc[0][0]
        a2 = best_mcmc[0][1] 
        B1 = best_mcmc[0][2]
        B2 = best_mcmc[0][3]
        C = best_mcmc[0][4]
        
        best_fit_model = model = np.piecewise(x, [(t1 <= x)*(x < t2), t2 <= x], 
                             [func1, func2],
                             t1, t2, a1, a2, B1, B2) + C
        
        nrows = 2
        ncols = 1
        fig, ax = plt.subplots(nrows, ncols, sharex=True,
                                       figsize=(8*ncols * 2, 3*nrows * 2))
        
        ax[0].plot(x, best_fit_model, label="best fit model", color = 'red')
        ax[0].scatter(x, y, label = "FFI data", s = 5, color = 'black')
        for n in range(nrows):
            ax[n].axvline(t1, color = 'blue', label="t1")
            ax[n].axvline(t2, color = "pink", label = "t2")
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
        
        plt.savefig(path + targetlabel + "-MCMCmodel-double-powerlaw.png")
        
    
    sn.beep()
    return flat_samples, best_mcmc, upper_error, lower_error

#%%
savepath = datapath + "doubleLaw-fixed/"
binT60, binI60, binE60 = sn.crop_to_percent(binT, standI,standE, 0.8)
samps, best60, upper60, lower60 = double_powerlaw(savepath, label, 
                                                  binT60, binI60, binE60, 
                                                  sector,discdate, 
                                                  plot = True, 
                                                  n1=50000, n2=70000)
#%% fix the two section's times

plt.scatter(binT60, binI60)
plt.axvline(binT60[33]) #start
plt.axvline(binT[41]) #stop