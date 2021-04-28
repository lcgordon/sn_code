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
        if ((disctime - 10) < t0 < (disctime + 10) and 0.5 < beta < 4.0 
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
        ax[nrows-1].set_xlabel("BJD-2457000")
        
        #residuals
        ax[1].set_title("Residual (y-model)")
        residuals = y - best_fit_model
        ax[1].scatter(x,residuals, s=5, color = 'black', label='residual')
        ax[1].axhline(0,color='purple', label="zero")
        ax[1].legend()
        
        plt.savefig(path + targetlabel + "-MCMCmodel-stepped-powerlaw.png")
        #plt.close()
        
    
    sn.beep()
    return best_mcmc, upper_error, lower_error

#%%


      #%%
import sn_functions as sn
import sn_plotting as sp
import data_utils as du
datapath = 'C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/'
savepath = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/2018koy/full/"
t,i,e, label, sector, discdate, gal_mags, info = sn.mcmc_load_one(datapath, savepath, runproduce = False)

#conversion and plotting
#2019yft is 17.1, 0.069
#2020chi is 20.4, 0.051
#2018koy is 20.44,0.314
galmag =20.44
extinction = 0.314
z = info["Z"][0]
#badIndexes = None #2019yft
#badIndexes = np.concatenate((np.arange(0,100), np.arange(628,641))) #2020efe
badIndexes = np.arange(415,440) #2018koy
#badIndexes = np.concatenate((np.arange(906,912), np.arange(645,665))) #2020chi
binT, binI, binE, absT, absI, absE = sp.plot_SN_LCs(savepath, t,i,e,label,
                                                    sector,galmag,extinction,
                                                    z, discdate, badIndexes)

#standardize to 1
standI, standE = du.flux_standardized(binI, binE)

#%% fULL light curve
best_params_file = savepath + "best_params.csv"
ID_file = savepath + "ids.csv"
upper_errors_file = savepath + "uppererr.csv"
lower_errors_file = savepath + "lowererr.csv"
best, upper, lower = stepped_powerlaw_basic(savepath, label, binT, standI, standE, sector,
                     discdate, best_params_file,
                     ID_file, upper_errors_file, lower_errors_file,
                     plot = True, n1=40000, n2=60000)

sp.print_table_formatting(best,upper,lower)
#%%
# PARTIAL light curve
binT40, binI40, binE40 = sn.crop_to_percent(binT, standI,standE, 0.75)
savepath = datapath + "75/"
best40, upper40, lower40 = stepped_powerlaw_basic(savepath, label, 
                                                  binT40, binI40, binE40, 
                                                  sector,discdate, best_params_file,
                                                  ID_file, upper_errors_file, lower_errors_file,
                                                  plot = True, n1=40000, n2=60000)

sp.print_table_formatting(best40,upper40,lower40)
#%% PARTIAL 2
binT60, binI60, binE60 = sn.crop_to_percent(binT, standI,standE, 0.6)
savepath = datapath + "60/"
best60, upper60, lower60 = stepped_powerlaw_basic(savepath, label, 
                                                  binT60, binI60, binE60, 
                                                  sector,discdate, best_params_file,
                                                  ID_file, upper_errors_file, lower_errors_file,
                                                  plot = True, n1=40000, n2=60000)

sp.print_table_formatting(best60,upper60,lower60)