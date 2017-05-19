#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:19:25 2017

@author: nklongvessa
"""

from trackpy import PandasHDFStoreSingleNode



def list_traj(path, chunksize = 2**12):
    """A list of unique index of all particles in a movie
    
    Parameters
    ----------
    path: string
        path to the HDF5 file which contains DataFrames(['particle'])
    chunksize: integer, default is 2**12   
    
    Returns
    -------
    a list that contains unique particle indices   
    """
    
    with PandasHDFStoreSingleNode(path) as traj_cell:
        chunk_uniq = []
        for chunk in traj_cell.store.select_column(traj_cell.key,"particle", chunksize = chunksize):
            chunk_uniq.append(int(chunk))
            chunk_uniq = list(set(chunk_uniq)) # get unique values
            
    return chunk_uniq

                

def filter_stubs(path, savepath, threshold = 30, chunksize = 2**12):
    """Filter out trajectories which are shorter than the threshol value. 

    Parameters
    ----------
    path : string
        path to the HDF5 file which contains DataFrames(['particle'])
    savepath: string
        path to be saved the result file as a HDF5 file 
    threshold : integer, default 30
        minimum number of points (video frames) to survive
    chunksize : integer, default is 2**12         

    Returns
    -------
    a subset of DataFrame in path, to be saved in savepath
    """

    from numpy import zeros
    from pandas import DataFrame
    
    with PandasHDFStoreSingleNode(path) as traj:
        
        parindex = list_traj(path) # get a list of particle index
        print('1/2 Find trajectories length')
        trajsizes = DataFrame(zeros(len(parindex)),index = parindex) # initialize a Dataframe [particle index, no. of apperance (frame)] 
        for chunk in traj.store.select_column(traj.key,"particle", chunksize = chunksize): # find the length of each trajectory
            trajsizes.loc[chunk] += 1 # bin it
        
        print('2/2 Save to a new file')              
        with PandasHDFStoreSingleNode(savepath) as temp: # creat a new file to store the result after stubs
            for f in range(traj.max_frame + 1): # loop frame
                frame = traj.get(f) # get frame by frame data
                frame = frame[(trajsizes.loc[frame.particle.astype(int)] >= threshold).values]  # keep long enough trajectories
               # frame.reset_index(inplace=True)
                temp.put(frame) # store in temp.h5 file 
        
        print('Before:', len(parindex))
        print('After:',  len(list_traj(savepath)))





def filter_index(path, savepath, pindices):
    """Filter out particle by a set of indices. For HDF5 files.

    Parameters
    ----------
    path : string
        path to the HDF5 file which contains DataFrames(['particle'])
    savepath: string
        path to be saved the result file as a HDF5 file 
    pindices : list 
        list of particle index, to be removed from the DataFrame

    Returns
    -------
    a subset of tracks. Dataframe([x, y, frame, particle]), to be saved in savepath
    """
    
    
    with PandasHDFStoreSingleNode(path) as traj:
        with PandasHDFStoreSingleNode(savepath) as result:
            
            for f in range(traj.max_frame): # loop frame
                frame = traj.store.select(traj.key, "frame == {}".format(f), columns= ['x', 'y', 'frame', 'particle']) 
                frame.set_index('particle', drop=False, inplace = True)
                
                # list of removing index = intersection between particle in frame and pindices
                remove = list(set(frame.particle.values) & set(pindices)) 
                
                frame.drop(remove, inplace = True) #
                result.put(frame)
                
    print('Before:', len(list_traj(path)))
    print('After:',  len(list_traj(savepath)))
    
    

def par_char(path):
    """Get particle mass, size and ecc as a time average value
        *** will be improved to make it faster ***

    Parameters
    ----------
    path : string
        path to the HDF5 file which contains DataFrames(['mass','size', 'ecc', 'particle'])
        
    Returns
    -------
    DataFrame([mass, size, ecc, particle])
    """
    
    from pandas import DataFrame
    from numpy import zeros

    parindex = list_traj("test_traj2.h5") # get indices of all particles
    
    char_av = DataFrame(zeros((len(parindex),4)),index = parindex, columns = ['mass','size', 'ecc', 'particle']) # initialize result Dataframe
    char_av.particle = parindex
    
    with PandasHDFStoreSingleNode(path) as traj:
        
        for p in parindex: # loop by particle **can be improved?**
            char_t = traj.store.select(traj.key, "particle == p", columns= ['mass', 'size', 'ecc','particle']) # char of one particle in every frame
            char_av.loc[p] = char_t.mean() # time average
        
    return char_av






def emsd(path, mpp, fps, nlagtime, max_lagtime, framejump = 10, pos_columns=None):
    """Compute the mean displacement and mean squared displacement of one
    trajectory over a range of time intervals for the streaming function.

    Parameters
    ----------
    path : string 
        path to the HDF5 file which contains DataFrames(['particle'])
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
    import numpy as np
    import pandas as pd
    
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
          
        return result
    
    
def compute_drift(path, smoothing=0, pos_columns=None):
    """Return the ensemble drift, xy(t).

    Parameters
    ----------
    path : string p
        path to the HDF5 file which contains DataFrames(['x','y','particle'])
    smoothing : integer
        Smooth the drift using a forward-looking rolling mean over
        this many frames.

    Returns
    -------
    drift : DataFrame([x, y], index=frame)
    """
    import numpy as np
    import pandas as pd
    
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




def subtract_drift(path, savepath, drift=None):
    """Return a copy of particle trajectories with the overall drift subtracted
    out.

    Parameters
    ----------
    path : string 
        path to the HDF5 file which contains DataFrames(['x','y'])
    savepath : string
        path to be saved the result file as a HDF5 file 
    drift : optional 
        DataFrame([x, y], index=frame) 
        If no drift is passed, drift is computed from traj.

    Returns
    -------
    Dataframe, to be saved in savepath
    """
    if drift is None:
        drift = compute_drift(path)

    with PandasHDFStoreSingleNode(path) as traj_old: 
        Nframe = traj_old.max_frame
        with PandasHDFStoreSingleNode(savepath) as traj_new: 
            for f in range(0,Nframe+1): # loop frame
               frame = traj_old.get(f)
               frame['x'] = frame['x'].sub(drift['x'][f])
               frame['y'] = frame['y'].sub(drift['y'][f])
               traj_new.put(frame) # put in the new file 
                                     
   
 