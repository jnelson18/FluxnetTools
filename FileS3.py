"""
@author: Jacob A. Nelson
"""

# Uses the numpy package for matrix operations
import numpy as np

def DiurnalCentroid(flux,UnitsPerDay=48):
    '''DiurnalCentroid(flux)
    
    Diurnal centroid of sub-daily fluxes
    
    Calculates the daily flux weighted time of a sub-daily flux.
    

    Parameters
    ----------
    flux : list or list like
        sub-daily flux that must be condinuous and regular
    UnitsPerDay : integer
        frequency of the sub-daily measurements, 48 for half hourly measurements

     Returns
    -------
    array
        The diurnal centroid, in the same units as UnitsPerDay, at a daily frequency
    '''
 
    # calculate the total number of days
    days,UPD=flux.reshape(-1,UnitsPerDay).shape
    # create a 2D matrix providing a UPD time series for each day, used in the matrix operations.      
    hours=np.tile(np.arange(UPD),days).reshape(days,UPD)
    # calculate the diurnal centroid
    C=np.sum(hours*flux.reshape(-1,48),axis=1)/np.sum(flux.reshape(-1,48),axis=1)
    C=C*(24/UnitsPerDay)
    return(C)

def NormDiurnalCentroid(LE,Rg,UnitsPerDay=48):
    '''NormDiurnalCentroid(LE,Rg)
    
    Normalized diurnal centroid of latent energy (LE)
    
    Calculates the diurnal centroid of LE relative to the diurnal centroid of incoming radiation (Rg).
    

    Parameters
    ----------
    LE : list or list like
        Latend energy, can be any unit
    Rg : list or list like
        Incoming radiation, can be any unit
    UnitsPerDay : integer
        frequency of the sub-daily measurements, 48 for half hourly measurements

     Returns
    -------
    array
        The normalized diurnal centroid, in the same units as UnitsPerDay, at a daily frequency
    '''
    C_LE=DiurnalCentroid(LE,UnitsPerDay=UnitsPerDay)
    C_Rg=DiurnalCentroid(Rg,UnitsPerDay=UnitsPerDay)
    return(C_LE-C_Rg)










