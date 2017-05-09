#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:08:24 2017
 
@author: nklongvessa
"""

import numpy as np
import pandas as pd
from trackpy import PandasHDFStoreSingleNode

def emsd_stream(path, mpp, fps, nlagtime, max_lagtime, framejump = 10, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals for the streaming function.

    Parameters
    ----------
    path : string path of the trajectory .h5
    mpp : microns per pixel
    fps : frames per second
    nlagtime : number of lagtime to which MSD is computed 
    max_lagtime : maximum intervals of frames out to which MSD is computed
    framejump : integer indicates the jump in t0 loop (to increase the speed) 
        Default : 10

    Returns
    -------
    DataFrame([<x^2>, <y^2>, msd, std, lagt])

    Notes
    -----
    Input units are pixels and frames. Output units are microns and seconds.
    """
    
    if pos_columns is None:
        pos_columns = ['x', 'y']
    result_columns = ['<{}^2>'.format(p) for p in pos_columns] + \
                      ['msd','std','lagt'] 
                      
    # define the lagtime to which MSD is computed. From 1 to fps, lagtime increases linearly with the step 1. 
    # Above fps, lagtime increases in a log scale until max_lagtime.
    lagtime = np.unique(np.append(np.arange(1,fps),(np.logspace(0,np.log10(max_lagtime/fps),nlagtime-fps)*fps).astype(int)))
    
    
    with PandasHDFStoreSingleNode(path) as traj: 
        Nframe = traj.max_frame # get number of frames
        
        result = pd.DataFrame(index = lagtime, columns = result_columns) # initialize the result Dataframe
        
        for lg in lagtime: # loop delta t
            lframe = range(0,Nframe + 1 - lg,framejump) # initialize t0
            msds = pd.DataFrame(index = range(len(lframe)),columns = result_columns) # initialize DataFrame for each t0
            
            for k,f in enumerate(lframe): # loop t0
                
                frameA = traj.get(f)
                frameB = traj.get(f+lg)
                # compute different position between 2 frames for each particle
                diff = frameB.set_index('particle')[pos_columns] - frameA.set_index('particle')[pos_columns]     
                msds[result_columns[0]][k] = np.nanmean((diff.x.values*mpp)**2) # <x^2>
                msds[result_columns[1]][k] = np.nanmean((diff.y.values*mpp)**2) # <y^2>
                    
            msds.msd = msds[result_columns[0]] + msds[result_columns[1]] # <r^2> = <x^2> + <y^2>
            
            result[result.index == lg] = [msds.mean()] # average over t0
            result.loc[result.index == lg,result.columns[3]] = msds.msd.std() # get the std over each t0
            
        result['lagt'] = lagtime/fps
        result.index.name = 'lagt'
          
        return result
    
    
def compute_drift_stream(path, smoothing=0, pos_columns=None):
    """Return the ensemble drift, xy(t).

    Parameters
    ----------
    path : string path of the trajectory .h5
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
       
     # Drift calculation 
    print('Drift calc')
    with PandasHDFStoreSingleNode(path) as traj: # open traj.h5
        Nframe = traj.max_frame
        dx = pd.DataFrame(data = np.zeros((Nframe+1,2)),columns = ['x','y'])    # initialize drift DataFrame     
        
        for f in range(Nframe): # loop frame
            frameA = traj.get(f)  # frame t
            frameB = traj.get(f+1) # frame t+1
            delta = frameB.set_index('particle')[pos_columns] - frameA.set_index('particle')[pos_columns]
            dx.iloc[f+1].x = np.nanmean(delta.x.values)
            dx.iloc[f+1].y = np.nanmean(delta.y.values) # compute drift
        
        if smoothing > 0:
            dx = pd.rolling_mean(dx, smoothing, min_periods=0)
        x = np.cumsum(dx)
    return x



#==============================================================================
# 
# def subtract_drift_stream(path_old, path_new, drift=None, inplace=False):
#     """Return a copy of particle trajectories with the overall drift subtracted
#     out.
# 
#     Parameters
#     ----------
#     path_old : savepath .h5 before the drift substraction
#     path_new : savepath .h5 after the drift substraction
#     drift : optional DataFrame([x, y], index=frame) like output of
#          compute_drift(). If no drift is passed, drift is computed from traj.
# 
#     Returns
#     -------
#     traj : a copy, having modified columns x and y
#     """
#     if drift is None:
#         drift = compute_drift(traj)
#     if not inplace:
#         traj = traj.copy()
#     traj.set_index('frame', inplace=True, drop=False)
#     traj.sort_index(inplace=True)
#     for col in drift.columns:
#         traj[col] = traj[col].sub(drift[col], fill_value=0, level='frame')
#     return traj
#==============================================================================
    
    
