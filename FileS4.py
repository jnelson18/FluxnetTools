# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:24:22 2017

@author: jnelson
"""

import FluxNetTools as fnt
import JMaths as jm

import numpy as np
from scipy.stats import ss

def daily_corr(x, y, Rg_pot):
    '''daily_corr(x, y)
    
    Daily correlation coefficient
    
    Calculates a daily correlation coefficient between two sub-daily timeseries
    

    Parameters
    ----------
    x : list or list like
        x variable
    y : list or list like
        y variable
    Rg_pot : list or list like
        potential radiation
     Returns
    -------
    array
        correlation coefficents at daily timescale
    '''
    x=x.reshape(-1,48)
    y=y.reshape(-1,48)
    Rg_pot=Rg_pot.reshape(-1,48)
    mask=Rg_pot<=0
    x=np.ma.MaskedArray(x,mask=mask)
    y=np.ma.MaskedArray(y,mask=mask)
    x=x/x.max(axis=1)[:,None]
    y=y/y.max(axis=1)[:,None]
    mx = x.mean(axis=1)
    my = y.mean(axis=1)
    xm, ym = x - mx[..., None], y - my[..., None]
    r_num = np.ma.add.reduce(xm * ym, axis=1)
    r_den = np.ma.sqrt(np.ma.sum(xm**2, axis=1) * np.ma.sum(ym**2, axis=1))
    r = r_num / r_den
    return(r**2)


def DWCIcalc(Rg_pot,LE,GPP,VPD,NEE,LE_sd,GPP_sd,NEE_fall,LE_fall):
    '''DWCIcalc(Rg_pot,LE,GPP,LE_sd,GPP_sd,NEE_fall,LE_fall)
    
    Diurnal water:carbon index (DWCI)
    
    DWCI measures the probability that the carbon and water are coupled within a given day. Method takes the correlation between evapotranspiration (LE) and gross primary productivity (GPP) and calculated the correlation within each day. This correlation is then compare to a distribution of correlations between artificial datasets built from the signal of potential radiation and the uncertainty in the LE and GPP.

    Parameters
    ----------
    Rg_pot : list or list like
        Sub-daily timeseries of potential radiation
    LE : list or list like
        Sub-daily timeseries of evapotranspiration or latent energy
    GPP : list or list like
        Sub-daily timeseries of gross primary productivity
    VPD : list or list like
        Sub-daily timeseries of vapor pressure deficit
    NEE : list or list like
        Sub-daily timeseries of net ecosystem exchange
    LE_sd : list or list like
        Sub-daily estimation of the uncertainty of LE
    GPP_sd : list or list like
        Sub-daily estimation of the uncertainty of GPP
    NEE_fall : list or list like
        Modeled sub-daily timeseries of net ecosystem exchange i.e. no noise
    LE_fall : list or list like
        Modeled sub-daily timeseries of evapotranspiration or latent energy i.e. no noise

     Returns
    -------
    array
        The diurnal water:carbon index (DWCI)
    '''

    # reshape all variables as number of days by number of half hours
    varList=[Rg_pot,LE,GPP,VPD,NEE,LE_sd,GPP_sd,NEE_fall,LE_fall]
    for j in range(len(varList)):
        varList[j]=varList[j].reshape(-1,48)
    Rg_pot,LE,GPP,VPD,NEE,LE_sd,GPP_sd,NEE_fall,LE_fall=varList
    # the number of artificial datasets to construct
    repeats=100
    # the number of days in the timeseries. Assumes data is half hourly
    days=int(LE.shape[0])
    # creates an empty 2D dataset to hold the artificial distributions
    StN=np.zeros([repeats,days])*np.nan
    corrDev=np.zeros([days,2,2])

    # create the daily cycle by dividing Rg_pot by the daily mean
    daily_cycle=Rg_pot/Rg_pot.mean(axis=1)[:,None]
    mean_GPP=GPP.mean(axis=1)
    mean_LE=LE.mean(axis=1)

    # Isolate the error of the carbon and water fluxes.
    NEE_err=NEE_fall-NEE
    LE_err=LE_fall-LE

    # loops through each day to generate an artificial dataset and calculate the associate correlation
    for d in range(days):#days
        if np.isnan(mean_GPP[d]) or np.isnan(mean_LE[d]):
            continue
        if np.isnan(NEE_err[d]).sum()>0 or np.isnan(LE_err[d]).sum()>0 or np.isnan(GPP_sd[d]).sum()>0 or np.isnan(LE_sd[d]).sum()>0:
            continue
        # find the correlation structure of the uncertanties to pass onto the artificial datasets

        if np.all(LE_err[d]==0) or np.all(NEE_err[d]==0):
            corrDev[d]=np.identity(2)
        else:
            corrDev[d] = np.corrcoef(-(NEE_err[d]),LE_err[d])

        # create our synthetic GPP and LE values for the current day
        synGPP  = np.zeros((repeats,48))*np.nan
        synLE   = np.zeros((repeats,48))*np.nan

        # this loop builds the artificial dataset using the covariance matrix between NEE and LE    
        for i in range(48):
            # compute the covariance matrix (s) for this half hour
            m   = [GPP_sd[d,i],LE_sd[d,i]]
            s   = np.zeros((2,2))*np.nan
            for j in range(2):
                for k in range(2):
                    s[j,k]    = corrDev[d,j,k]*m[j]*m[k]
        
            Noise    = np.random.multivariate_normal([0,0],s,100) # generate random 100 values with the std of this half hour and the correlation between LE and GPP
            synGPP[:,i] = daily_cycle[d,i]*mean_GPP[d]+Noise[:,0]    # synthetic gpp
            synLE[:,i]  = daily_cycle[d,i]*mean_LE[d]+Noise[:,1]     # synthetic le

        # calculate the 100 artificial correlation coefficients for the day       
        StN[:,d]=daily_corr(synGPP, synLE, np.tile(daily_cycle[d],100).reshape(-1,48))

    # calculate the real correlation array
    pwc=daily_corr(LE,GPP*np.sqrt(VPD),Rg_pot)
    
    # calculate the rank of the real array within the artificial dataset giving DWCI
    DWCI=(StN<np.tile(pwc,repeats).reshape(repeats,-1)).sum(axis=0)
    DWCI[np.isnan(StN).prod(axis=0).astype(bool)]=-9999
    
    return(DWCI)
