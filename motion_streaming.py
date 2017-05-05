#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:08:24 2017
 
@author: nklongvessa
"""

import numpy as np
import pandas as pd
from trackpy import PandasHDFStoreSingleNode

def emsd_stream(savepath, mpp, fps, nlagtime, max_lagtime, framejump = 10, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals for the streaming function.

    Parameters
    ----------
    savepath : string path of the trajectory .h5
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
    
    
    with PandasHDFStoreSingleNode(savepath) as traj: 
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
    
    
def compute_drift_stream(savepath, smoothing=0, pos_columns=None):
    """Return the ensemble drift, xy(t).

    Parameters
    ----------
    traj : DataFrame of trajectories, including columns x, y, frame, and particle
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)

    Examples
    --------
    >>> compute_drift(traj).plot()
    >>> compute_drift(traj, 0, ['x', 'y']).plot() # not smoothed, equivalent to default.
    >>> compute_drift(traj, 15).plot() # Try various smoothing values.
    >>> drift = compute_drift(traj, 15) # Save good drift curves.
    >>> corrected_traj = subtract_drift(traj, drift) # Apply them.
    """
    if pos_columns is None:
        pos_columns = ['x', 'y']
    # the groupby...diff works only if the trajectory Dataframe is sorted along frame
    # I do here a copy because a "inplace=True" would sort the original "traj" which is perhaps unwanted/unexpected
    traj = pandas_sort(traj, 'frame')
    # Probe by particle, take the difference between frames.
    delta = traj.groupby('particle', sort=False).apply(lambda x :
                                    x.set_index('frame', drop=False).diff())
    # Keep only deltas between frames that are consecutive.
    delta = delta[delta['frame'] == 1]
    # Restore the original frame column (replacing delta frame).
    del delta['frame']
    delta.reset_index('particle', drop=True, inplace=True)
    delta.reset_index('frame', drop=False, inplace=True)
    dx = delta.groupby('frame').mean()
    if smoothing > 0:
        dx = pd.rolling_mean(dx, smoothing, min_periods=0)
    x = dx.cumsum(0)[pos_columns]
    return x
    
       
     # Drift calculation 
    print('Drift calc')
    drift = pd.DataFrame(data = np.zeros((Nframe+1,2)),columns = ['x','y'])    # initialize drift DataFrame     
    with tp.PandasHDFStoreSingleNode(savepath_temp.format(act)) as temp: # open temp.h5
        for f in range(0,Nframe): # loop frame
            frameA = temp.get(f)  # frame t
            frameB = temp.get(f+1) # frame t+1
            diff = frameB.set_index('particle')[pos_columns] - frameA.set_index('particle')[pos_columns]
            drift.iloc[f+1].x = np.nanmean(diff.x.values)
            drift.iloc[f+1].y = np.nanmean(diff.y.values) # compute drift
        drift = np.cumsum(drift)
        frameA = []
        frameB = []


    
    
