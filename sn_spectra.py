# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:44:08 2021

@author: conta
"""
import numpy as np
import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt

def photosphere_vel(rest_wavelength, observed_wavelength):
    import astropy.constants as const
    import astropy.units as u
    delta_lambda = ((observed_wavelength - rest_wavelength) * u.Angstrom)
    lambda_rest = rest_wavelength * u.Angstrom

    return (delta_lambda/lambda_rest) * const.c 

def load_txt_spectra(file):
    """Extracts spectra from 2column text file """
    spectra = np.loadtxt(file)
    wave = spectra[:,0]
    flux = spectra[:,1]
    return np.asarray(wave), np.asarray(flux)

def load_ascii_spectra(file):
    """Load data in from a two column ascii filie """
    f = open(file, 'r')
    wave = []
    flux = []
    for line in f:
        line = line.strip()
        columns = line.split()
        wave.append(float(columns[0]))
        flux.append(float(columns[1]))
    return np.asarray(wave), np.asarray(flux)

def blackbody(lam, T):
        """ Blackbody as a function of wavelength (m) and temperature (K)."""
        lam = lam * u.m
        T = T * u.K
        return(( 2*const.h*const.c**2 / (lam**5 * 
                (np.exp(const.h*const.c / (lam*const.k_B*T)) - 1)
                )).to(u.J/(u.s * u.Angstrom**3))).value

def fit_bb(path, wave, flux, label, p0 = 10000):
    """Fit a blackbody curve to the spectra. 
    wave should be in units of Angstrom
    flux should be in units of J/sA^3 """
    from scipy.optimize import curve_fit
    import pylab as plt
    import numpy as np

    

    #wave = wave.to(u.Angstrom) # set into units
    flux = flux.to((u.J/(u.s * u.Angstrom**3)))
    wave2 = wave.to(u.m).value #converting to m for use in bb function
    
    #run curve fit
    coeff, covar = curve_fit(blackbody, wave2, flux.value, p0 = p0)
    print("T:", coeff)
    sigma = np.sqrt(np.diag(covar))
    print("Error:", sigma)
    #plotting
    plt.scatter(wave, flux, label = "raw spectra", color = 'black',
                s = 2,alpha = 0.5)
    
    model = blackbody(wave2, *coeff)
    plt.plot(wave, model, label="best fit model", color = 'red')
    
    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Spectral Flux (J/sA^3)")
    plt.title(label + " Spectra. Blackbody best fit T= %.1f K" % coeff[0])
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(spectra_folder + label + "_bbfit.png")

    return coeff, covar, sigma

#%% velocity calculations
#rest_wavelength = 4000 #2018koy
#observed_wavelength = 4124 #2018koy

#rest_wavelength = 6000 #2019yft
#observed_wavelength = 6450 #2019yft

rest_wavelength = 6000#2020chi
observed_wavelength = 6420#2020chi

vel = photosphere_vel(rest_wavelength, observed_wavelength)
print(vel, vel/1e7)

#%% black body fitting
#loading 2020chi - this runs no problem! 
spectra_folder = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/"
file2020chi = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/tns_2020chi_2020-02-14_12-16-05_P60_SEDM_ZTF.ascii"


wave2020chi, flux2020chi = load_ascii_spectra(file2020chi)
wave2020chi = wave2020chi * u.Angstrom
flux2020chi = flux2020chi * u.J/(u.s * u.Angstrom**3)
fit_bb(spectra_folder, wave2020chi, flux2020chi, "2020chi")

#2020koy - fixed issue
file2018koy = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/tns_2018koy_2018-12-31_05-17-41_ESO-NTT_EFOSC2-NTT_ePESSTO.asci"

wave2018koy, flux2018koy = load_ascii_spectra(file2018koy)
wave2018koy = wave2018koy * u.Angstrom
flux2018koy = flux2018koy * u.J/(u.s * u.Angstrom**3)

fit_bb(spectra_folder, wave2018koy, flux2018koy, "2018koy")

#%% blackbody curve fitting

#load in spectra
file2019yft = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/tns_2019yft_2019-12-30_03-23-38_LT_SPRAT_TCD.txt"

wave2019yft, flux2019yft = load_txt_spectra(file2019yft)
wave2019yft = wave2019yft * u.Angstrom
flux2019yft = flux2019yft * u.J/(u.s * u.Angstrom * u.cm**2)
fit_bb(spectra_folder, wave2019yft, flux2019yft, "2019yft")

#%% converting to tmin

from astropy.time import Time
classDate2019yft = Time("2019-12-30 03:23:38",format = 'iso', scale='utc')

vel2019yft = 2.3 * 10**7 * u.m/u.s #m/s
T2019yft = 8500 * u.K#Kelvin

fig, ax = plt.subplots()
ax.scatter(absT, absI)
ax.axvline(classDate2019yft.jd-2457000, color='green')
plt.axvline(discdate, color = 'red')
ax.invert_yaxis()
#%%
absIatTime = absI[13]
L_0 = 3.0128 * 10**28 * u.W
L2019yft = L_0 * 10**(-0.4*absIatTime)


def calc_tmin(L, T, Vph):
    """
    L in erg/s
    T in K
    Vph in km/s"""
    return (4.3 * ((L/(1e42 * u.erg/u.s))**0.5) * 
            ((T/(1e4 * u.K))**(-2)) * (Vph/(1e4 * u.km/u.s))**(-1))

tmin2019yft = calc_tmin(L2019yft.to(u.erg/u.s), T2019yft, vel2019yft.to(u.km/u.s))

#%% for 2020chi

classDate2020chi = Time("2020-02-14 12:16:05", format='iso', scale='utc')

vel2020chi = 2 * 10**7 * u.m/u.s #m/s
T2020chi = 9900 * u.K#Kelvin

fig, ax = plt.subplots()
ax.scatter(absT, absI)
ax.axvline(classDate2020chi.jd-2457000, color='green')
plt.axvline(discdate, color = 'red')
plt.axvline(classDate2020chi.jd-2457000-8.5242)
ax.invert_yaxis()
#%%
absIatTime = -19.25
L_0 = 3.0128 * 10**28 * u.W
L2020chi = L_0 * 10**(-0.4*absIatTime)
print(L2020chi.to(u.erg/u.s))



tmin2020chi = calc_tmin(L2020chi.to(u.erg/u.s), T2020chi, vel2020chi.to(u.km/u.s))
print(tmin2020chi)



#%% 2018koy
classDate2018koy = Time("2018-12-31 05:17:41", format='iso', scale='utc')

vel2018koy = 9.3 * 10**6 * u.m/u.s #m/s
T2018koy = 13900 * u.K#Kelvin

fig, ax = plt.subplots()
ax.scatter(absT, absI)
ax.axvline(classDate2018koy.jd-2457000, color='green')
plt.axvline(discdate, color = 'red')
#plt.axvline(classDate2018koy.jd-2457000-8.5242)
ax.invert_yaxis()
#%%
absIatTime = -16.70
L_0 = 3.0128 * 10**28 * u.W
L2018koy = L_0 * 10**(-0.4*absIatTime)
print(L2018koy.to(u.erg/u.s))



tmin2018koy = calc_tmin(L2018koy.to(u.erg/u.s), T2018koy, vel2018koy.to(u.km/u.s))
print(tmin2018koy)






















