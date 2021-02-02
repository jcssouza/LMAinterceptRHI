import csv
import pandas as pd
import numpy as np
from datetime import datetime, date, time, timedelta
import glob
from lmatools.io.LMA_h5_file import LMAh5File
from lmatools.coordinateSystems import RadarCoordinateSystem, GeographicSystem, TangentPlaneCartesianSystem, MapProjection

def search_files(dir, time_start, time_end):
    """
    This function return the lma files in the directory dir within a time interval.
    time_start and time end are in datetime format (datetime.datetime(YYYY, MM, DD, HH, MM))
    """
    lmafiles = sorted(glob.glob(dir + '*.h5'))
    # Start
    ind_s = 0
    ind_e = len(lmafiles)
    for i in np.arange(len(lmafiles)):
        lma = LMAh5File(lmafiles[i])
        if (lma.start_times[0] == time_start):
            ind_s = i
            break

    # End
    for i in np.arange(len(lmafiles)):
        lma = LMAh5File(lmafiles[i])
        if (lma.start_times[0] == time_end):
            ind_e = i
            break
            
    lmafiles = lmafiles[ind_s:ind_e]

    return lmafiles

import math
def closest_lma_rhi_time(lma_df, dt, radar_time):
    """
    Given a lma dataframe, a time threshold in seconds and the (radar) time,
    it returns the lma dataframe with the sources within the time entered +/- the threshold dt specified in seconds
    """
    # Converting the lma VHF sources times from seconds to datetime format
    hr = (lma_df.time.values//3600)
    mins = (lma_df.time.values%3600)//60
    seg = ((lma_df.time.values%3600)%60)
    frac = np.zeros(len(lma_df))
    whole = np.zeros(len(lma_df))
    for i in np.arange(len(lma_df)):
        frac[i], whole[i] = math.modf(seg[i])    
    microsegs = frac*(10**6)
    segs = whole

    lma_times = np.zeros(len(lma_df), dtype=object)
    for t in np.arange(len(lma_df)):
        lma_times[t] = datetime.combine(radar_time.date(), 
                                        time(int(hr[t]),int(mins[t]),int(segs[t]),int(microsegs[t])))
    # a['datetime'] = pd.to_datetime(lma_times)
    # pd.Timestamp(lma_times[i], tz=None).to_pydatetime()
    lma_df['datetime'] = lma_times

    # flashes on +/- second of the radar scan
    tmi = abs(lma_times-(radar_time-timedelta(seconds=dt)))
    tma = abs(lma_times-(radar_time+timedelta(seconds=dt)))
    flash_tmi = lma_df.flash_id[tmi.argmin() : tmi.argmin() + 1].values[0]
    flash_tma = lma_df.flash_id[tma.argmin() : tma.argmin() + 1].values[0]
    cond = np.logical_and(lma_df.flash_id >= flash_tmi, lma_df.flash_id <= flash_tma)

    cons_lma_df = lma_df[cond]
    return cons_lma_df


from radarlma2local import rcs_to_tps, geo_to_tps
from ortho_proj import rot_mat_lma
def closest_lma_rhi(lma_df, lma_ortho, dsi):
    """
    Given a lma dataframe lma_df, a distance dsi threshold in km from the (radar rhi) location, and the distance of each source from the RHI scan lma_ortho,
    it returns a dataframe like lma_df of the flashes that had at least one source within the threshold.
    """ 
    Xlma_ortho=np.abs(lma_ortho[:,0])
    Ylma_ortho=np.abs(lma_ortho[:,1])
    
    # Getting all the sources that is less than dsi (m) away from the x-axis
    ## IF ONLY ONE SOURCE MEETS THE REQUIREMENT, THE WHOLE FLASH IS INCLUDED 

    lma_df_close = pd.DataFrame(columns = lma_df.columns)  #----- creating new DF
    for i in np.arange(len(lma_df.flash_id.values[np.where(Ylma_ortho < dsi)])):
        a = lma_df[lma_df.flash_id.values == lma_df.flash_id.values[np.where(Ylma_ortho < dsi)][i]]
        lma_df_close = lma_df_close.append(a)
    return lma_df_close

def closest_lma_rhi_cs(lma_df, radar, dsi):
    """
    Given a lma dataframe, a distance ds threshold in km from the (radar rhi) location,
    it returns a 3D array of the location, height and time of lma sources within the threshold specified.
    """
    X, Y, Z = rcs_to_tps(radar)
    Xlma,Ylma,Zlma = geo_to_tps(lma_df, radar)
    lma_file_ortho  = ortho_proj_lma(radar, lma_df)
    Ylma_ortho=np.abs(lma_file_ortho[:,1])

    lma_file_ortho_xnew = lma_file_ortho[np.where(Ylma_ortho<dsi),0][0,:]
    Zlma_new = Zlma[np.where(Ylma_ortho<dsi)]
    time_new = lma_df.time.values[np.where(Ylma_ortho<dsi)]
    # putting 3 arrays together
    lma_ortho_new = np.zeros(shape=(len(lma_file_ortho_xnew), 3))
    lma_ortho_new[:,0] = lma_file_ortho_xnew
    lma_ortho_new[:,1] = Zlma_new
    lma_ortho_new[:,2] = time_new
    return lma_ortho_new

def one_flash(lma_df, flash_id):
    """
    Given a lma file and the flash id, it returns the lma dataframe with only the VHF sources with the specified flsah id.
    """
    return lma_df[lma_df.flash_id == flash_id]

def loc_1source(lma_df, flash_id):
    """
    Given lma file dataframe and the flash id, it returns a list of [latiude, longitude and altitude] 
    of the average of the 4 first VHF sources identified for the specific flash.
    """
    sel = lma_df[lma_df.flash_id.values == flash_id]
    sel = sel.sort_values(by=['time'])
    loc = []
    loc.append(np.mean(sel.lat.values[0:4]))
    loc.append(np.mean(sel.lon.values[0:4]))
    loc.append(np.mean(sel.alt.values[0:4]))
    return loc


