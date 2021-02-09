import numpy as np
from datetime import datetime, date, time, timedelta
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pyart
import os
import pytda

from radarlma2local import rcs_to_tps, geo_to_tps
from ortho_proj import ortho_proj_lma


def plot_locs_lma_rhi(radar, lma_df, zoom):
    """
    Plot the RHI scan and VHF sources locations in latitude and longitude given a radar file read by pyart and lma file pd.df
    """
    d = date(int(radar.time['units'][14:18]),int(radar.time['units'][19:21]),int(radar.time['units'][22:24]))
    t = time(int(radar.time['units'][25:27]),int(radar.time['units'][28:30]),int(radar.time['units'][31:33]))
    tt = datetime.combine(d,t)

    fig = plt.figure(figsize=(5, 5),dpi=150)
    projection = ccrs.PlateCarree()
    ax = fig.add_subplot(111, projection = projection)
    ax.coastlines(resolution='10m')
    gl = ax.gridlines(color='gray', linestyle='--', draw_labels=True)
    gl.xlabels_top = False
    gl.ylabels_right = False

    zoom = zoom
    ext = [ radar.longitude['data'] - zoom, radar.longitude['data'] + zoom, 
           radar.latitude['data'] - zoom, radar.latitude['data'] + zoom]
    ax.set_extent(ext)

    lma_df = lma_df.sort_values(by=['time'])
    llma = plt.scatter(x = lma_df.lon, y = lma_df.lat, transform=projection, marker = '*', s = 50,
               c = (lma_df.time.values - lma_df.time.values[0]), cmap = 'plasma',alpha=0.9)
    rradar = plt.scatter(radar.longitude['data'],radar.latitude['data'], 
            transform=projection,color = 'slategray', edgecolor='black', marker="o", s=40)
    cbar1 = plt.colorbar(llma, shrink = 0.82)
    cbar1.set_label('Time (s)', size=15)
    cbar1.ax.tick_params(labelsize=11)

    ax.text(-0.21, 0.55, 'Latitude (degree)', va='bottom', ha='center', rotation='vertical',rotation_mode='anchor',
            transform=ax.transAxes, fontsize = 12)
    ax.text(0.5, -0.13, 'Longitude (degree)', va='bottom', ha='center',rotation='horizontal', rotation_mode='anchor',
            transform=ax.transAxes, fontsize = 12)

    # Plot the RHI scan location
    rrhi = ax.plot([radar.longitude['data'], radar.gate_longitude['data'][0][-1]],
            [radar.latitude['data'], radar.gate_latitude['data'][0][-1]], 
            transform=projection, color = 'black')

    legend_elements1 = [Line2D([0], [0], color='gray',lw=0,linestyle = None,marker = '*',
                              markersize = 11, alpha=0.55,
                               label = 'VHF sources'),
                      Line2D([0], [0], lw=0,linestyle = None,marker = 'o', markeredgecolor = 'black', color = 'slategray',
                              markersize = 11,alpha=0.75,
                              label = 'Ka-band radar deployment'),
                       Line2D([0], [0], color='black',lw=2,linestyle ='solid',marker = None,
                               alpha=0.95,
                               label = 'RHI scan')]
    fig.legend(handles=legend_elements1, loc = 'lower center', ncol=3, fontsize = 12, framealpha = 0.9)   

    ax.set_title(tt.strftime('RHI scan and VHF sources locations \n' + 
                                 "%m/%d/%y %H:%M:%S") + ' UTC',y = 1.04,fontsize = 13)


from matplotlib import patches 
def plot_RHI_EDR_panel(rrcoords, zrcoords, values, rlmacoords, zlmacoords, ylma, xzoom_min, yzoom_min, xzoom_max, yzoom_max, radar_time, flash_id):
    """
    Plot overview and zoomed region of EDR for RHI scan with VHF sources given edges for x- and y-axis of radar coordinates, EDR values,
    and r-,z- and y-axis VHF point coords.
    xzoom_min, yzoom_min, xzoom_max, yzoom_max: specify region for zoom plot
    radar_time, flash_id: to plot the title, radar_time in datetime format (datetime.datetime(YYYY, MM, DD, HH, MM))
    """
    fig = plt.figure(figsize=(16, 8), dpi=300)
    # -- 1
    ax1 = fig.add_subplot(121)
    cs = ax1.pcolormesh(rrcoords, zrcoords, values, vmin = 0, vmax = 1)
    llma = plt.scatter(rlmacoords, zlmacoords, color = 'Red', edgecolor = 'black', marker = "X", s = 100, alpha = 1)
    # -- Plot box for target domain
    rect = patches.Rectangle((xzoom_min,yzoom_min), xzoom_max-xzoom_min, yzoom_max-yzoom_min, linewidth = 2, linestyle = '--', 
                             edgecolor = 'black',facecolor = 'none')
    ax1.add_patch(rect)
    
    xlim = (0, 24)
    ylim = (0, 15)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xticks(fontsize = 11)
    plt.yticks(fontsize = 11)

    # -- 2
    ax2 = fig.add_subplot(122)
    cs = ax2.pcolormesh(rrcoords, zrcoords, values, vmin = 0, vmax = 1)
    llma = plt.scatter(rlmacoords, zlmacoords, c = abs(ylma), edgecolor = 'black', cmap = 'Reds_r', marker = "X", s = 200, alpha = 0.5, vmin = 0, vmax = 100)

    plt.xlim(xzoom_min, xzoom_max)
    plt.ylim(yzoom_min, yzoom_max)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    fig.text(0.5, 0.93,radar_time.strftime("TTU Ka-band %d %B %Y %H:%M:%S") + ' \n flash id = ' + str(flash_id) + 
             ' - LMA sources within 100 m from the RHI scan', 
             va='center', ha='center', fontsize = 15)
    fig.text(0.5, 0.05, 'Distance from radar (km)', va='center', ha='center', fontsize = 15)
    fig.text(0.09, 0.5, 'Distance above radar (km)', va='center', ha='center', rotation='vertical', fontsize = 15)
    
    # -- cbar
    cb_ax = fig.add_axes([1.01, 0.13, 0.02, 0.75])
    cbar = fig.colorbar(cs, cax=cb_ax)
    cbar.set_label('$EDR^{0.33}$', size=15)
    cbar.ax.tick_params(labelsize=11)


    llma_ax = fig.add_axes([0.93, 0.13, 0.02, 0.75])
    cbar2 = fig.colorbar(llma, cax=llma_ax)
    cbar2.set_label('Orthogonal Distance (km)', size=15)
    cbar2.ax.tick_params(labelsize=11)


from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D
import matplotlib as mpl
def size_pt(altitude, min_alt, max_alt):
    """
    Gives point size depending on its altitude. 
    Higher(Lower) altitude = smaller(bigger) point 
    """
    coefa = (50/( min_alt - max_alt))
    coefb = 60 - (coefa * min_alt)
    return coefa * altitude + coefb
def plot_EDR_dVR(table):
    """
    Plot EDR and velocity spatial derivatives as a function of the distance between flash initiation and interception by the RHI scan given the results table.
    """
    fig = plt.figure(figsize=(24, 10), dpi=300)
    # For plotting the size: y = 0 x = 50, 
    coefa = (50/(table.init_alt.values.min() - table.init_alt.values.max()))
    coefb = 75 - (coefa * table.init_alt.values.min())
    # -- 1
    ax1 = fig.add_subplot(121)
    scatter = plt.scatter(x = table.dist_1.values, 
                          y = table.turbulence.values, 
                          marker = 'o', 
                          s = size_pt(table.init_alt.values,table.init_alt.values.min(), table.init_alt.values.max())*20,
                          c = np.log10(table.area.values),
                          cmap = 'viridis', vmin = 0, vmax = 3, alpha = 0.65,
                          edgecolor='dimgrey')
    ax1.axhline(0, color = 'k', linestyle = 'dotted')

    legend_elements = [Line2D([0], [0], color='dimgrey',lw=0,linestyle = None,marker = 'o',
                          markersize = np.sqrt(size_pt(5000,table.init_alt.values.min(), table.init_alt.values.max())*20),
                          label='5 km'),
                  Line2D([0], [0], color='dimgrey',lw=0,linestyle = None,marker = 'o',
                          markersize = np.sqrt(size_pt(9000,table.init_alt.values.min(), table.init_alt.values.max())*20),
                          label='9 km'),
                   Line2D([0], [0], color='dimgrey',lw=0,linestyle = None,marker = 'o',
                          markersize = np.sqrt(size_pt(13000,table.init_alt.values.min(), table.init_alt.values.max())*20),
                          label='13 km')]
    legend = plt.legend(handles = legend_elements, ncol = 1,loc = 'upper right',
                        labelspacing = 2,fontsize = 12, title='Altitude',
                        framealpha = 0, borderpad = 0.7)
    legend.set_title('Altitude',prop={'size':14})

    cbar = plt.colorbar(label='$log_{10}$ Flash Area ($km^2$)', pad = 0.01)
    cbar.set_label('$log_{10}$ Flash Area ($km^2$)', fontsize = 15)
    cbar.ax.tick_params(labelsize = 13)

    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.xlabel('Distance from the flash initiation point to the interception point on the RHI scan (m)', fontsize = 14)
    plt.ylabel('EDR $(m^2 s^{-3}) ^{1/3}$', fontsize = 14)

    # -- 2
    ax2 = fig.add_subplot(122)
    plt.scatter(x = table.dist_1.values[np.where(table.area.values>0)], 
                y = table.dVRde.values[np.where(table.area.values>0)], 
                marker = 'o', 
                s = size_pt(table.init_alt.values,table.init_alt.values.min(), table.init_alt.values.max())*20,
                c = np.log10(table.area.values[np.where(table.area.values>0)]),
                cmap='Blues_r',vmin=0, vmax = 3 ,alpha = 0.55, edgecolor='dimgrey')

    plt.scatter(x = table.dist_1.values[np.where(table.area.values>0)], 
                y = table.dVRdr.values[np.where(table.area.values>0)], 
                marker = 'o', 
                s = size_pt(table.init_alt.values,table.init_alt.values.min(), table.init_alt.values.max())*20,
                c = np.log10(table.area.values[np.where(table.area.values>0)]), 
                cmap = 'Reds_r', alpha=0.55, vmin = 0, vmax =3, edgecolor='dimgrey')

    ax2.axhline(0, color = 'k', linestyle = 'dotted')


    legend_elements1 = [Line2D([0], [0], color='dimgrey',lw=0,linestyle = None,marker = 'o',
                              markersize = np.sqrt(size_pt(5000,table.init_alt.values.min(), table.init_alt.values.max())*20),
                              label='5 km'),
                      Line2D([0], [0], color='dimgrey',lw=0,linestyle = None,marker = 'o',
                              markersize = np.sqrt(size_pt(9000,table.init_alt.values.min(), table.init_alt.values.max())*20),
                              label='9 km'),
                       Line2D([0], [0], color='dimgrey',lw=0,linestyle = None,marker = 'o',
                              markersize = np.sqrt(size_pt(13000,table.init_alt.values.min(), table.init_alt.values.max())*20),
                              label='13 km')]
    legend1 = plt.legend(handles=legend_elements1, ncol = 1, loc = 'upper right',
                     labelspacing=2,fontsize = 12, title='Altitude',
               framealpha = 0, borderpad=0.7)
    legend1.set_title('Altitude',prop={'size':14})
    plt.gca().add_artist(legend1)

    legend_elements2 = [Line2D([0], [0], color='blue',lw=0,linestyle = None,marker = 'o',
                              markersize = 14, alpha=0.55,
                               label = r'$\frac{1}{r} \frac{\partial V_R}{\partial \hat{\Phi}}$'),
                      Line2D([0], [0], color='red',lw=0,linestyle = None,marker = 'o',
                              markersize = 14,alpha=0.55,
                              label = r'$\frac{\partial V_R}{\partial \hat{r}}$')]
    legend2 = plt.legend(handles=legend_elements2, ncol = 1, loc = 'lower right',
                     labelspacing=0.7,fontsize = 20, 
               framealpha = 0, borderpad=0.7)

    cmap = plt.get_cmap('Greys_r',100)
    norm = mpl.colors.Normalize(vmin=0,vmax=3)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, label='$log_{10}$ Flash Area ($km^2$)',pad=0.01)
    cb.set_label('$log_{10}$ Flash Area ($km^2$)', size=15)
    cb.ax.tick_params(labelsize=13)

    plt.xlabel('Distance from the flash initiation point to the interception point on the RHI scan (m)', fontsize = 14)
    plt.ylabel('Velocity derivative in space ($s^{-1}$)', fontsize = 14)
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.ylim(-0.2, 0.2)
    
    td1 = fig.text(0.1274, 0.8898, "a", 
         fontsize = 18,fontweight = 'bold')

    td2 = fig.text(0.55, 0.8898, "b", 
             fontsize = 18,fontweight = 'bold')


def plot_sources_interp(df_nn, rm_nn, zm_nn, rstd_nn, zstd_nn, nn_dist1_weight_avg, nn_tur_weight_avg, nn_dist1_weight_std, nn_tur_weight_std,
                        df_oban1, rm_oban1, zm_oban1, rstd_oban1, zstd_oban1, oban1_dist1_weight_avg, oban1_tur_weight_avg, oban1_dist1_weight_std, oban1_tur_weight_std,
                        df_oban2, rm_oban2, zm_oban2, rstd_oban2, zstd_oban2, oban2_dist1_weight_avg, oban2_tur_weight_avg, oban2_dist1_weight_std, oban2_tur_weight_std):
    """
    Plot 3 panel with EDR value for all the sources within the threshold obtained using nearest neighbor (nn), objective analysis Barnes rcutoff1 (oban1),
    and objective analysis Barnes rcutoff2 (oban2), and their associated error based on the standard deviation from the average and from the distance weighted average.
    df_*: dataframe containing turbulence, dist_itp and dist_1
    rm_*: regular mean for dist_1 for nn, oban1, oban2
    zm_*: regular mean for turbulence for nn, oban1, oban2
    rstd_*: regular standard deviation for dist_1 for nn, oban1, oban2
    zstd_*: regular standard deviation for turbulence for nn, oban1, oban2
    *_dist1_weight_avg: weighted mean for dist_1 for nn, oban1, oban2
    *_tur_weight_avg: weighted mean for turbulence for nn, oban1, oban2
    *_dist1_weight_std: weighted standard deviation for dist_1 for nn, oban1, oban2
    *_tur_weight_std: weighted standard deviation for turbulence for nn, oban1, oban2
    """
    xlim = (min(df_nn.dist_1/1000) - 3, max(df_nn.dist_1/1000) + 3)
    ylim = (min(df_nn.turbulence) - 0.1, max(df_nn.turbulence) + 0.1)
    s1 = 100 # - markersize

    fig = plt.figure(figsize=(16, 8), dpi=300)
    ###############################################################################################
    # 1
    ax1 = fig.add_subplot(131)
    cs = plt.scatter(x = df_nn.dist_1/1000., y = df_nn.turbulence, marker = 'D', s = s1, cmap = 'Reds_r',
                    vmin = 0, vmax = 100, c = df_nn.dist_itp.values, alpha = 0.8, edgecolors = 'black',
                    label = 'LH[0]')

    # - Closest source
    r1_nn = df_nn.dist_1[df_nn.dist_itp.values == min(df_nn.dist_itp.values)].values[0]/1000.
    z1_nn = df_nn.turbulence[df_nn.dist_itp.values == min(df_nn.dist_itp.values)].values[0]
    css = plt.scatter(r1_nn, z1_nn, color = 'gray', edgecolor = 'black', marker = 'X', s = 160, label = 'LH[1]')
    # - Mean and std dev
    error = plt.errorbar(x = rm_nn, y = zm_nn, xerr = rstd_nn, yerr = zstd_nn, ecolor = 'blue', 
                 elinewidth = 2, color = 'blue', label = 'LH[2]', marker = 'o')

    # - Mean and std dev weighted
    error2 = plt.errorbar(x = nn_dist1_weight_avg, y = nn_tur_weight_avg, 
                          xerr = nn_dist1_weight_std, yerr = nn_tur_weight_std, 
                          ecolor = 'green', elinewidth = 2, color = 'green', label = 'LH[2]', marker = 'o')

    #
    plt.ylim(ylim)
    plt.xlim(xlim )
    plt.title('Nearest Neighbor', fontsize = 14)

    ###############################################################################################
    # 2
    ax2 = fig.add_subplot(132)
    cs = plt.scatter(x = df_oban1.dist_1/1000., y = df_oban1.turbulence, marker = 'D', s = s1, cmap = 'Reds_r',
                    vmin = 0, vmax = 100, c = df_oban1.dist_itp.values, alpha = 0.8, edgecolors = 'black',
                    label = 'LH[0]')
    # - Closest source
    r1_oban1 = df_oban1.dist_1[df_oban1.dist_itp.values == min(df_oban1.dist_itp.values)].values[0]/1000.
    z1_oban1 = df_oban1.turbulence[df_oban1.dist_itp.values == min(df_oban1.dist_itp.values)].values[0]
    css = plt.scatter(r1_oban1, z1_oban1, color = 'gray', edgecolor = 'black', marker = 'X', s = 160, label = 'LH[1]')
    # - Mean and std dev
    error = plt.errorbar(x = rm_oban1, y = zm_oban1, xerr = rstd_oban1, yerr = zstd_oban1, ecolor = 'blue', 
                 elinewidth = 2, color = 'blue', label = 'LH[2]', marker = 'o')
    # - Mean and std dev weighted
    error2 = plt.errorbar(x = oban1_dist1_weight_avg, y = oban1_tur_weight_avg, 
                          xerr = oban1_dist1_weight_std, yerr = oban1_tur_weight_std, 
                          ecolor = 'green', elinewidth = 2, color = 'green', label = 'LH[2]', marker = 'o')

    #
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title('Barnes Intepolation Rcutoff 1 ', fontsize = 14)

    ###############################################################################################
    # 3
    ax3 = fig.add_subplot(133)
    cs = plt.scatter(x = df_oban2.dist_1/1000., y = df_oban2.turbulence,  marker = 'D', s = s1, cmap = 'Reds_r',
                     vmin = 0, vmax = 100, c = df_oban2.dist_itp.values, alpha = 0.8, edgecolors = 'black',
                    label = 'LH[0]')
    # - Closest source
    r1_oban2 = df_oban2.dist_1[df_oban2.dist_itp.values == min(df_oban2.dist_itp.values)].values[0]/1000.
    z1_oban2 = df_oban2.turbulence[df_oban2.dist_itp.values == min(df_oban2.dist_itp.values)].values[0]
    css = plt.scatter(r1_oban2, z1_oban2, color = 'gray', edgecolor = 'black', marker = 'X', s = 160,
                     label = 'LH[1]')
    # - Mean and std dev
    error = plt.errorbar(x = rm_oban2, y = zm_oban2, xerr = rstd_oban2, yerr = zstd_oban2, ecolor = 'blue', 
                 elinewidth = 2, color = 'blue', label = 'LH[2]', marker = 'o')
    # - Mean and std dev weighted
    error2 = plt.errorbar(x = oban2_dist1_weight_avg, y = oban2_tur_weight_avg, 
                          xerr = oban2_dist1_weight_std, yerr = oban2_tur_weight_std, 
                          ecolor = 'green', elinewidth = 2, color = 'green', label = 'LH[2]', marker = 'o')

    #
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.title('Barnes Intepolation Rcutoff 2', fontsize = 14)
    ############################################################################################
    # - Colorbar
    cs_ax = fig.add_axes([0.92, 0.13, 0.02, 0.75])
    cbar = fig.colorbar(cs, cax=cs_ax)
    cbar.set_label('Orthogonal Distance (km)', size=13)

    # - Axis
    fig.text(0.5, 0.07, 'Distance from the flash initiation point to the interception point on the RHI scan (km)', 
             va='center', ha='center', fontsize = 15)
    fig.text(0.08, 0.5, '$EDR^{0.33}$', va='center', ha='center', rotation='vertical', fontsize = 15)
    # -- Tilte
    #fig.text(0.5, 0.96, radar_time.strftime("%d %B %Y %H:%M:%S") + '   flash id = ' + str(flashes_id[k]) +
    #         '   Closest source distance = {:.2f} m'.format(min(abs(Y_min))),
    #        va='center', ha='center', fontsize = 16)

    # - Create the legend
    leg = fig.legend(('Close LMA sources','Closest LMA source','Mean','Interception distance weighted mean'),
                     loc='lower center', ncol = 4, fontsize = 12)
    LH = leg.legendHandles
    LH[0].set_color('red')
    LH[0].set_edgecolor('black') 

