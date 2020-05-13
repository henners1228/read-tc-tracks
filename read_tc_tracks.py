"""
Script that downloads and processes ECMWF Tropical Cyclone BUFR track data and 
plots individual tracks, ensemble mean and probabilities

Todo:
    Include other models (GEFS/ICON-EPS) by tracking low pressure centres 
    within ax extent
    
    Use multiprocessing to speed probability calculation up
    
    Incorporate previous track data
    
Known Bugs:
    Problem with plotting data that crosses the dateline due to cartopy
    projection. Needs fixing

Takes model run (00/12) as argument, dependent on dissemination.ecmwf.int FTP
server as well as modules below

@author: Henry Bright
"""

import numpy as np
import datetime as dt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as mticker

from ftplib import FTP
import fnmatch
import sys
import os
import xarray as xr

from pybufrkit.decoder import Decoder
from  pybufrkit.renderer import FlatTextRenderer

from geopy.distance import distance

import math

def ensemble_mean(coords_df):

    x = 0.0
    y = 0.0
    z = 0.0
    
    for i, coord in coords_df.iterrows():
        latitude = math.radians(coord.Latitude)
        longitude = math.radians(coord.Longitude)
    
        x += math.cos(latitude) * math.cos(longitude)
        y += math.cos(latitude) * math.sin(longitude)
        z += math.sin(latitude)
    
    total = len(coords_df)
    
    x = x / total
    y = y / total
    z = z / total
    
    central_longitude = math.atan2(y, x)
    central_square_root = math.sqrt(x * x + y * y)
    central_latitude = math.atan2(z, central_square_root)
    
    mean_location = {
        'latitude': math.degrees(central_latitude),
        'longitude': math.degrees(central_longitude)
        }
    
    return mean_location

def downloadFile(directory,filename):
    
    if not os.path.exists(directory):
        
        os.mkdir(directory)
        
    if not os.path.exists(directory+filename):
    
        localfile = open(directory+filename, 'wb')
        ftp.retrbinary('RETR ' + filename, localfile.write)
    
        localfile.close()
    
def hasNumbers(inputString):

    return any(char.isdigit() for char in inputString)

def wind_10m_colourmap():

    precip_colors = [ 
        'dodgerblue',
			'lime',	#11-17kts
			'greenyellow',	#17-22kts
			'yellow',	#22-28kts
			'orange',	#28-34kts
			'darkorange',	#34-41kts
			'orangered',	#41-48kts
			'mediumorchid',	#48-56kts	
			'rebeccapurple',	#56-64kts
		]
		
    cmap = mpl.colors.ListedColormap(precip_colors)
    cmap.set_under('white')        
    cmap.set_over('grey')


    clevels = [0,11,17,22,28,34,41,48,56,64]
    norm = mpl.colors.BoundaryNorm(clevels,cmap.N)
    
    return cmap, norm, clevels

def prob_colourmap():

    precip_colors = [
            'lightgreen',
			'lime',	#11-17kts
			'greenyellow',	#17-22kts
			'yellow',	#22-28kts
			'orange',	#28-34kts
			'darkorange',	#34-41kts
			'orangered',	#41-48kts
			'mediumorchid',	#48-56kts	
			'rebeccapurple',	#56-64kts
            'grey',	#56-64kts
		]
		
    cmap = mpl.colors.ListedColormap(precip_colors)      
    cmap.set_under('white') 
    
    clevels = [5,10,20,30,40,50,60,70,80,90,100]
    norm = mpl.colors.BoundaryNorm(clevels,cmap.N)
    
    return cmap, norm, clevels

hfont = {'fontname': 'Segoe UI'}

model_run = sys.argv[1]

ftp = FTP("dissemination.ecmwf.int")

ftp.login("wmo","essential")

directory = dt.datetime.now().strftime("%Y%m%d{}0000/").format(model_run)

#directory = "20191031000000/" #comment out this line if wanting to download latest data

try:

    ftp.cwd(directory)
    
    filenames = ftp.nlst()
    
    bufr_files = fnmatch.filter(filenames, "*bufr4*")
    
    for filename in bufr_files:
        
        downloadFile(directory,filename)
    
    ftp.quit()

except:
    
    print("Directory does not exist on FTP server, checking locally")

    if os.path.isdir(directory):
        
        print("Found local directory, proceeding")
        
        filenames = os.listdir(directory)
    
        bufr_files = fnmatch.filter(filenames, "*bufr4*")
        
    else:
        
        sys.exit("Files not found locally")

sorted_bufr_files = sorted(bufr_files)

named_storm_files = []

for filename in sorted_bufr_files:
    if hasNumbers(filename[64:64+filename[64:].find("_")]):
        print("Not a named storm, skipping")
        
    else:
        named_storm_files.append(directory+filename)

if len(named_storm_files)==0:
    
    sys.exit("No named storms, exiting")

composite_storm_files = [named_storm_files[x:x+2] for x in range(0, len(named_storm_files),2)]

for storm in composite_storm_files:
    
    ens_path = storm[0]
    
    det_path = storm[1]
    
    # Decode ensemble bufr file
    decoder = Decoder()
    with open(ens_path, 'rb') as ins:
        bufr_message = decoder.process(ins.read())
    
    text_data = FlatTextRenderer().render(bufr_message)
    
    text_array = np.array(text_data.splitlines())
    
    for line in text_array:
        
        if "WMO LONG STORM NAME" in line:
            
            storm_name = line.split()[-1][:-1]
    
    section4 = text_array[np.where(text_array=="<<<<<< section 4 >>>>>>")[0][0]:np.where(text_array=="<<<<<< section 5 >>>>>>")[0][0]]
    
    list = []
    ens_subset = 0
    attribute = None
    tplus_hour = None
    
    for line in section4:
        if "subset" in line:
            ens_subset+=1
            tplus_hour=0
        elif "YEAR" in line:
            year = int(line.split()[-1])
        elif "MONTH" in line:
            month = int(line.split()[-1])
        elif "DAY" in line:
            day = int(line.split()[-1])
        elif "HOUR" in line:
            hour = int(line.split()[-1])
        elif "MINUTE" in line:
            minute = int(line.split()[-1])
        
        elif "METEOROLOGICAL ATTRIBUTE SIGNIFICANCE" in line:
        
            attribute = int(line.split()[-1])
        
        elif ((attribute==1) and ("LATITUDE" in line) and (tplus_hour==0)):
              
            lat_reported = line.split()[-1]
              
        elif ((attribute==1) and ("LONGITUDE" in line) and (tplus_hour==0)):
              
            lon_reported = line.split()[-1]
        
        elif "TIME PERIOD" in line:
            tplus_hour = int(line.split()[-1])
        
        elif (("LATITUDE" in line) and (attribute==4) and (tplus_hour==0)): #first analysis point
        
            lat = line.split()[-1]
            
        elif (("LONGITUDE" in line) and (attribute==4) and (tplus_hour==0)):
            
            lon = line.split()[-1]
             
        elif "PRESSURE" in line and (attribute==4):
            pressure = line.split()[-1]
            
        elif "WIND SPEED" in line and (attribute==3):
            if tplus_hour==0:
            
                time_0 = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)+dt.timedelta(hours=tplus_hour)
            
            time = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)+dt.timedelta(hours=tplus_hour)
    
            wind_speed = line.split()[-1]
            
            list.append([time,lat,lon,pressure,wind_speed,ens_subset])
            
        elif (("LATITUDE" in line) and (attribute==1) and (line.split()[-1]!="None")):
            
            lat = line.split()[-1]
            
        elif (("LONGITUDE" in line) and (attribute==1) and (line.split()[-1]!="None")):
            
            lon = float(line.split()[-1])
            
        elif "PRESSURE" in line and (attribute==1):
            pressure = line.split()[-1]
            
        elif "WIND SPEED" in line and (attribute==3):
            time = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)+dt.timedelta(hours=tplus_hour)
    
            wind_speed = line.split()[-1]
            
            list.append([time,lat,lon,pressure,wind_speed,ens_subset])
    
    df_ens = pd.DataFrame(list,columns=["Datetime","Latitude","Longitude","Central Pressure [hPa]","Max 10m Wind Speed within 750km of centre [m/s]","Ensemble member"]).set_index("Datetime")
    
    ens_member = df_ens["Ensemble member"]
    
    df_ens[df_ens[["Latitude","Longitude","Central Pressure [hPa]","Max 10m Wind Speed within 750km of centre [m/s]"]]=="None"]=np.nan
    
    df_ens["Central Pressure [hPa]"] = df_ens["Central Pressure [hPa]"].astype(float)/100
    
    df_ens["Ensemble member"] = ens_member
    
    # Decode a BUFR file
    decoder = Decoder()
    with open(det_path, 'rb') as ins:
        bufr_message = decoder.process(ins.read())
    
    text_data = FlatTextRenderer().render(bufr_message)
    
    text_array = np.array(text_data.splitlines())
    
    for line in text_array:
        
        if "WMO LONG STORM NAME" in line:
            
            storm_name = line.split()[-1][:-1]
    
    section4 = text_array[np.where(text_array=="<<<<<< section 4 >>>>>>")[0][0]:np.where(text_array=="<<<<<< section 5 >>>>>>")[0][0]]
    
    list = []
    det_subset = 0
    
    for line in section4:
        if "subset" in line:
            det_subset+=1
            tplus_hour=0
        elif "YEAR" in line:
            year = int(line.split()[-1])
        elif "MONTH" in line:
            month = int(line.split()[-1])
        elif "DAY" in line:
            day = int(line.split()[-1])
        elif "HOUR" in line:
            hour = int(line.split()[-1])
        elif "MINUTE" in line:
            minute = int(line.split()[-1])
        
        elif "METEOROLOGICAL ATTRIBUTE SIGNIFICANCE" in line:
        
            attribute = int(line.split()[-1])
        
        elif "TIME PERIOD" in line:
            tplus_hour = int(line.split()[-1])
            
        elif (("LATITUDE" in line) and (attribute==4) and (tplus_hour==0)): #first analysis point
        
            lat = line.split()[-1]
            
        elif (("LONGITUDE" in line) and (attribute==4) and (tplus_hour==0)):
            
            lon = line.split()[-1]
             
        elif "PRESSURE" in line and (attribute==5):
            pressure = line.split()[-1]
            
        elif "WIND SPEED" in line and (attribute==3):
            if tplus_hour==0:
            
                time_0 = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)+dt.timedelta(hours=tplus_hour)
            
            time = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)+dt.timedelta(hours=tplus_hour)
    
            wind_speed = line.split()[-1]
            
            list.append([time,lat,lon,pressure,wind_speed,det_subset])
            
        elif (("LATITUDE" in line) and (attribute==1) and (line.split()[-1]!="None")):
            
            lat = line.split()[-1]
            
        elif (("LONGITUDE" in line) and (attribute==1) and (line.split()[-1]!="None")):
            
            lon = float(line.split()[-1])
            
        elif "PRESSURE" in line and (attribute==1):
            pressure = line.split()[-1]
            
        elif "WIND SPEED" in line and (attribute==3):
            time = dt.datetime(year=year,month=month,day=day,hour=hour,minute=minute)+dt.timedelta(hours=tplus_hour)
    
            wind_speed = line.split()[-1]
            
            list.append([time,lat,lon,pressure,wind_speed,det_subset])
    
    df_det = pd.DataFrame(list,columns=["Datetime","Latitude","Longitude","Central Pressure [hPa]","Max 10m Wind Speed within 750km of centre [m/s]","Ensemble member"]).set_index("Datetime")
    
    ens_member = df_det["Ensemble member"]
    
    df_det[df_det[["Latitude","Longitude","Central Pressure [hPa]","Max 10m Wind Speed within 750km of centre [m/s]"]]=="None"]=np.nan
    
    df_det["Central Pressure [hPa]"] = df_det["Central Pressure [hPa]"].astype(float)/100
    
    df_det["Ensemble member"] = ens_member
    
    #calculate ensemble mean per timestep
    
    ens_mean_list = []
    
    for i in range(df_ens.index.drop_duplicates().shape[0]-1):
    
        ens_mean_list.append([df_ens.index.drop_duplicates()[i],ensemble_mean(df_ens[df_ens.index==df_ens.index.drop_duplicates()[i]][["Latitude","Longitude"]].astype(float).dropna())["latitude"], ensemble_mean(df_ens[df_ens.index==df_ens.index.drop_duplicates()[i]][["Latitude","Longitude"]].astype(float).dropna())["longitude"]])
    
    df_ens_mean = pd.DataFrame(ens_mean_list)
    
    df_ens_mean.columns = ["Datetime","Latitude","Longitude"]
    
    df_ens_mean = df_ens_mean.set_index("Datetime")
    
    #calculate ax extent
    x0,x1,y0,y1 = [df_ens["Longitude"].astype(float).min()-2,df_ens["Longitude"].astype(float).max()+2,df_ens["Latitude"].astype(float).min()-2,df_ens["Latitude"].astype(float).max()+2]
    
    #try and make plots square by having lat exent=lon extent
    
    if y1-y0 > x1-x0: #latitude bigger than longitude
        
        diff = (y1-y0) -  (x1-x0)
        
        x0_adj = x0 - diff/2
        
        x1_adj = x1 + diff/2
        
        y0_adj = y0
        y1_adj = y1
    
    elif y1-y0 < x1-x0: #latitude bigger than longitude
        
        diff = (x1-x0) -  (y1-y0)
        
        y0_adj = y0 - diff/2
        
        y1_adj = y1 + diff/2
        
        x0_adj = x0
        x1_adj = x1

    
    #calculate probabilities
    res = 1 #grid size resolution - decreasing dramatically increases computation time
    
    xx, yy = np.meshgrid(np.arange(x0,x1+res,res),np.arange(y0,y1+res,res))
    
    prob = np.zeros((xx.shape[0],xx.shape[1]))
    
    for j in range(xx.shape[1]-1):
        
        for i in range(xx.shape[0]-1):
            
            # consider points for each 6 hour period only rather, than the whole forecast time!
            for time in df_ens.index.drop_duplicates():
                
                count = 0
                nan_values = 0
            
                print(yy[i,j],xx[i,j])
            
                for lat,lon in df_ens[["Latitude","Longitude"]][df_ens.index==time].values:
                    
                    try:
                    
                        if distance((yy[i,j],xx[i,j]),(lat,lon)).km <= 120: #consider selected point on grid and loop through all other forecast centres to generate prob.
                        
                            count += 1
                            
                    except ValueError:
                        
                        print("nan lat or lon value, skipping") #exclude nan values where centre is not forecast by ensemble member etc.
                        
                        nan_values += 1
                
                if not prob[i,j]==0: #if value already stored in grid
                    
                    if prob[i,j]<((count/(df_ens[["Ensemble member"]].drop_duplicates().shape[0]-nan_values)) * 100 ): #only write to grid if greater than value already present
                    
                        prob[i,j] = ((count/(df_ens[["Ensemble member"]].drop_duplicates().shape[0]-nan_values)) * 100 )
                        
                    else:
                        
                        print("Calculated probability not higher than value already present")
                else:
                    
                    #if zero write to array
                    prob[i,j] = ((count/(df_ens[["Ensemble member"]].drop_duplicates().shape[0]-nan_values)) * 100 )
    
    #start plotting
    
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12,12), dpi=150)
    
    plot = ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.OCEAN.with_scale("50m"))
        
    ax.set_extent([x0_adj,x1_adj,y0_adj,y1_adj], ccrs.PlateCarree())

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.75, linestyle='--')
    
    x_list = np.arange(-180,180+5,5).tolist()
    
    y_list = np.arange(-90,90+5,5).tolist()
    
    gl.xlocator = mticker.FixedLocator(x_list)
    gl.ylocator = mticker.FixedLocator(y_list)
    
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','fontname': 'Segoe UI', "size": 14}
    gl.ylabel_style = {'color': 'k','fontname': 'Segoe UI', "size": 14}
    
    cmap, norm, clevels = wind_10m_colourmap()
    
    #plot storm tracks and max 10m wind
    # for member in np.arange(1,ens_subset+1):
    #     a = ax.plot(df_ens[df_ens["Ensemble member"]==member]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==member]["Latitude"].values.astype(float),"-",color="black",linewidth=0.5,alpha=0.75)
    #     s_ens = ax.scatter(df_ens[df_ens["Ensemble member"]==member]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==member]["Latitude"].values.astype(float),c=df_ens[df_ens["Ensemble member"]==member]["Max 10m Wind Speed within 750km of centre [m/s]"].values.astype(float)*1.94384,cmap=cmap,norm=norm)
    # s_det = ax.scatter(df_det[df_det["Ensemble member"]==1]["Longitude"].values.astype(float),df_det[df_det["Ensemble member"]==1]["Latitude"].values.astype(float),c=df_det[df_det["Ensemble member"]==1]["Max 10m Wind Speed within 750km of centre [m/s]"].values.astype(float)*1.94384,cmap=cmap,norm=norm) #MULTIPLY TO GET KNOTS
    
    #control
    #c = ax.plot(df_ens[df_ens["Ensemble member"]==51]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==51]["Latitude"].values.astype(float),"-",color="blue",linewidth=2)
    
    det = ax.plot(df_det[df_det["Ensemble member"]==1]["Longitude"].values.astype(float),df_det[df_det["Ensemble member"]==1]["Latitude"].values.astype(float),"-o",color="blue",linewidth=2,markersize=4)
    ens_mean = ax.plot(df_ens_mean["Longitude"],df_ens_mean["Latitude"],linestyle="dotted",color="blue",linewidth=2)
    
    reported_location = ax.plot(float(lon_reported),float(lat_reported),"x",color="black",mew=2.5, ms=7.5)
    
    title = "10 day Tropical Cyclone track forecast for {}".format((storm_name.lower()).capitalize())
    ax.set_title("{}".format(title), loc='center',**hfont, size=20)
    
    cmap, norm, clevels = prob_colourmap()
    
    ax.contourf(xx,yy,prob,levels=clevels,cmap=cmap,norm=norm)
    
    divider = make_axes_locatable(ax)
    
    #uncomment below if plotting probabilities
    cbaxes = divider.new_vertical(size="2%", pad=0.3, pack_start=True, axes_class=plt.Axes)
    cbaxes.set_aspect(.025)
    cbaxes.set_anchor('C')
    fig.add_axes(cbaxes)
    colourbar = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm, orientation='horizontal', ticks=[5,10,20,30,40,50,60,70,80,90,100])
    
    colourbar.set_label("Probability of TC centre passing within 120km over next 10 days",**hfont, size=16)
    colourbar.ax.set_xticklabels(["5%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])
    colourbar.ax.tick_params(labelsize=14)
    
    #comment out above and uncomment below if plotting ensemble tracks and max wind speeds
    # cbaxes = divider.new_vertical(size="2%", pad=0.3, pack_start=True, axes_class=plt.Axes)
    # cbaxes.set_aspect(.025)
    # cbaxes.set_anchor('C')
    # fig.add_axes(cbaxes)
    # colourbar = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
    
    # colourbar.set_label("Maximum 10m Wind Speed within 750km of forecast TC centre (kts)",**hfont, size=12)
    
    init_datetime = ax.annotate(time_0.strftime("Initialised: %a %d %b %Y %H UTC"), xy=(0.99, 0.99), xycoords='axes fraction', 
                            fontsize=14, **hfont ,horizontalalignment='right', 
                            verticalalignment='top', backgroundcolor="white",alpha=1, zorder=10)
    
    #for individual storm tracks
    #ax.legend([a[0],b[0],c[0]],["Ensemble Member","Deterministic","Control"],loc="upper left",framealpha=1)
    ax.legend([ens_mean[0],det[0],reported_location[0]],["Ensemble Mean","Deterministic","Reported Location"],loc="upper left",framealpha=1,fontsize=14)
    
                             
    if not os.path.exists(directory+"plot"):
        
        os.mkdir(directory+"plot")
    
    plt.savefig(directory+"plot"+"/{}_probs.png".format(storm_name),bbox_inches="tight")
    
    fig2, ax2 = plt.subplots(1, 1, subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12,12), dpi=150)
    
    ax2.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax2.add_feature(cfeature.OCEAN.with_scale("50m"))
        
    ax2.set_extent([x0_adj,x1_adj,y0_adj,y1_adj], ccrs.PlateCarree())

    gl = ax2.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                  linewidth=1, color='gray', alpha=0.75, linestyle='--')
        
    gl.xlocator = mticker.FixedLocator(x_list)
    gl.ylocator = mticker.FixedLocator(y_list)
    
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xlines = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'color': 'k','fontname': 'Segoe UI', "size": 14}
    gl.ylabel_style = {'color': 'k','fontname': 'Segoe UI', "size": 14}
    
    cmap, norm, clevels = wind_10m_colourmap()
    
    #plot storm tracks and max 10m wind
    for member in np.arange(1,ens_subset+1):
        a = ax2.plot(df_ens[df_ens["Ensemble member"]==member]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==member]["Latitude"].values.astype(float),"-",color="black",linewidth=0.5,alpha=0.75)
        s_ens = ax2.scatter(df_ens[df_ens["Ensemble member"]==member]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==member]["Latitude"].values.astype(float),c=df_ens[df_ens["Ensemble member"]==member]["Max 10m Wind Speed within 750km of centre [m/s]"].values.astype(float)*1.94384,cmap=cmap,norm=norm)
    s_det = ax2.scatter(df_det[df_det["Ensemble member"]==1]["Longitude"].values.astype(float),df_det[df_det["Ensemble member"]==1]["Latitude"].values.astype(float),c=df_det[df_det["Ensemble member"]==1]["Max 10m Wind Speed within 750km of centre [m/s]"].values.astype(float)*1.94384,cmap=cmap,norm=norm) #MULTIPLY TO GET KNOTS
    
    #control
    #c = ax.plot(df_ens[df_ens["Ensemble member"]==51]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==51]["Latitude"].values.astype(float),"-",color="blue",linewidth=2)
    
    det = ax2.plot(df_det[df_det["Ensemble member"]==1]["Longitude"].values.astype(float),df_det[df_det["Ensemble member"]==1]["Latitude"].values.astype(float),"-",color="blue",linewidth=2)
    #ens_mean = ax2.plot(df_ens_mean["Longitude"],df_ens_mean["Latitude"],linestyle="dotted",color="#011E41",linewidth=2)
    
    reported_location = ax2.plot(float(lon_reported),float(lat_reported),"x",color="black",mew=2.5, ms=7.5)
    
    title = "10 day Tropical Cyclone track forecast for {}".format((storm_name.lower()).capitalize())
    ax2.set_title("{}".format(title), loc='center',**hfont, size=20)
     
    divider = make_axes_locatable(ax2)
    
    #uncomment below if plotting probabilities
    cbaxes = divider.new_vertical(size="2%", pad=0.3, pack_start=True, axes_class=plt.Axes)
    cbaxes.set_aspect(.025)
    cbaxes.set_anchor('C')
    fig2.add_axes(cbaxes)
    
    colourbar = mpl.colorbar.ColorbarBase(cbaxes, cmap=cmap, norm=norm, extend='max', orientation='horizontal')
    
    colourbar.set_label("Maximum 10m Wind Speed within 750km of forecast TC centre (kts)",**hfont, size=16)
    colourbar.ax.tick_params(labelsize=14)
    
    init_datetime = ax2.annotate(time_0.strftime("Initialised: %a %d %b %Y %H UTC"), xy=(0.99, 0.99), xycoords='axes fraction', 
                            fontsize=14, **hfont ,horizontalalignment='right', 
                            verticalalignment='top', backgroundcolor="white",alpha=1, zorder=10)
    
    #for individual storm tracks
    ax2.legend([a[0],det[0],reported_location[0]],["Ensemble Member","Deterministic","Reported Location"],loc="upper left",framealpha=1,fontsize=14)
    
                             
    if not os.path.exists(directory+"plot"):
        
        os.mkdir(directory+"plot")
    
    plt.savefig(directory+"plot"+"/{}_tracks.png".format(storm_name),bbox_inches="tight")

    plt.close()    