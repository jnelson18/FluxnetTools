"""
Created on Fri Aug 26 17:32:13 2016

@author: Jacob A. Nelson and Martin Jung
"""

# Uses the numpy package for matrix operations
import numpy as np

def LEtoET(LE):
    # convert LE in W_m-2 to mm per half hour
    # 2300000 J per L, latent heat of vaporization
    # 1800    seconds per half hour
    ET=np.ma.getdata(LE)*((1800)/2300000)
    return(ET)

def MWB(precip,LE,s0=5,ConvertET=True):
    '''MWB(precip,LE,s0=5,ConvertET=True)
    
    Modified Water Balance (MWB)
    
    MWB forces a positive water water storage for any time-step with precipitation, yet has a maximum storage of s0.
    

    Parameters
    ----------
    precip : list or list like
        precipitation in mm
    LE : list or list like
        Latent energy in W_m-2 or ET in mm per half hour (must set ConvertET=False).
    s0 : value, float or int
        The maximum storage of the water balance
    ConvertET : bool
        Flag telling wheter to convert LE to ET.

     Returns
    -------
    array
        The modified water balance
    '''
    # insure that datasets are one dimensional
    precip=precip.reshape(-1)
    LE=LE.reshape(-1)
    # extract the data in case that precip has an associated mask
    precipcalc=np.ma.getdata(precip)
    
    # in case of an associated mask, fill all gap values with the maximum water storage
    if np.ma.is_masked(precip):    
        precipcalc[np.ma.getmask(precip)]=s0
    else:
    # in case of any negative values in precipitation fill with the maximum water storage
        precipcalc[(precip<0)]=s0
    # fill any other missing values with the maximum water storage
    precipcalc[np.isnan(precip)]=s0
    precipcalc[np.isinf(precip)]=s0

    # create an array of zeros the same shape as the precipitation data to hold the MWB
    mwb=np.zeros(precip.shape)
    # set the initial value of MWB to the max storage capacity
    mwb[0]=s0
    # if needed, convert the latent energy to mm per half hour
    if ConvertET:
        LEmm=LEtoET(LE)
    else:
        LEmm=LE

    # set any missing values in LE data to a number
    LEmm[np.isnan(LEmm)]=-9999
    LEmm[np.isinf(LEmm)]=-9999

    # loop through each timestep, skipping the inital condition        
    for j in range(precip.shape[0]-1):
        k=j+1
        # stepVal give either the current water balance or s0, causing s0 to be a ceiling 
        stepVal=min(mwb[j]+precipcalc[k]-LEmm[k],s0)
        # in case of a positive precip value, the current MWB is the max between the previous 
        # MWB and either the value of the precip or the s0 depending on which is smaller
        if precipcalc[k]>0:
            mwb[k]=max(stepVal,min(precipcalc[j],s0))
        # if there is no precip, the MWB is according to the stepVal,
        # causing simple water balance behaviour
        else:
            mwb[k]=stepVal
    return(mwb)
