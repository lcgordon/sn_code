# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 15:44:08 2021

@author: conta
"""
#%% spectral stuff
import astropy.constants as const
#load in spectra
file = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/tns_2019yft_2019-12-30_03-23-38_LT_SPRAT_TCD.txt"
spectra = np.loadtxt(file)
#plot spectra
wavelength = (spectra[:,0] * u.Angstrom).to(u.nm)
strength = spectra[:,1]
plt.scatter(wavelength, strength)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux (arbitrary)")
#%%
#fit blackbody curve
T = 6900 * u.K
scale = 2.5e13
B = (((2 * const.h * const.c**2)/wavelength**5)/(
    np.exp(const.h * const.c/(wavelength * const.k_B * T)) - 1)).to(u.kg / (u.m * u.s**3))

B = B/scale
plt.scatter(wavelength, B, label = "Approximated fit")
plt.scatter(wavelength, strength, label = "raw spectra")
plt.title("2019yft TCD Spectra and Blackbody Fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux (arbitrary)")
plt.legend(loc="upper right")
peak_wavelength = 425 * u.nm
Teff = (const.b_wien/peak_wavelength).to(u.K)
spectra_folder = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/"
plt.savefig(spectra_folder + "2019yft-spectrafit.png")
#identify peak
#use wien's law to convert to temperature

#Teff = const.b_wien * peak_wavelength

#%% loading 2020chi
file2020chi = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/tns_2020chi_2020-02-14_12-16-05_P60_SEDM_ZTF.ascii"
f = open(file2020chi, 'r')
wav2020chi = []
strength2020chi = []
for line in f:
    line = line.strip()
    columns = line.split()
    wav = columns[0]
    strength = columns[1]
    print(name, j)
    wav2020chi.append(float(wav))
    strength2020chi.append(float(strength))

wav2020chi = (wav2020chi * u.Angstrom).to(u.nm)
    #%%

plt.scatter(wav2020chi, strength2020chi, label="raw spectra")
plt.title("2020chi TCD Spectra and Blackbody Fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux (arbitrary)")
plt.legend(loc="upper right")


T = 6900 * u.K
scale = 2.5e29
B = (((2 * const.h * const.c**2)/wav2020chi**5)/(
    np.exp(const.h * const.c/(wav2020chi * const.k_B * T)) - 1)).to(u.kg / (u.m * u.s**3))

B = B/scale
plt.scatter(wav2020chi, B, label = "Approximated fit")
plt.axvline(425, label="peak")

peak_wavelength = 425 * u.nm
Teff = (const.b_wien/peak_wavelength).to(u.K)
spectra_folder = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/"
plt.savefig(spectra_folder + "2020chi-spectrafit.png")

#%%
file2018koy = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/tns_2018koy_2018-12-31_05-17-41_ESO-NTT_EFOSC2-NTT_ePESSTO.asci"
f = open(file2018koy, 'r')
wav2018koy = []
strength2018koy = []
for line in f:
    line = line.strip()
    columns = line.split()
    wav = columns[0]
    strength = columns[1]
    print(name, j)
    wav2018koy.append(float(wav))
    strength2018koy.append(float(strength))

wav2018koy = (wav2018koy * u.Angstrom).to(u.nm)
#%%
plt.scatter(wav2018koy, strength2018koy, label="raw spectra")
plt.title("2018koy ePESSTO Spectra and Blackbody Fit")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Flux (arbitrary)")
plt.legend(loc="upper right")


T = 6900 * u.K
scale = 7e28
B = (((2 * const.h * const.c**2)/wav2018koy**5)/(
    np.exp(const.h * const.c/(wav2018koy * const.k_B * T)) - 1)).to(u.kg / (u.m * u.s**3))

B = B/scale
plt.scatter(wav2018koy, B, label = "Approximated fit")
plt.axvline(420, label="peak")

peak_wavelength = 420 * u.nm
Teff = (const.b_wien/peak_wavelength).to(u.K)
spectra_folder = "C:/Users/conta/UROP/plot_output/IndividualFollowUp/spectra/"
plt.savefig(spectra_folder + "2018koy-spectrafit.png")