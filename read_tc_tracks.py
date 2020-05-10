# -*- coding: utf-8 -*-
"""
Created on Sat May  2 12:18:02 2020

@author: Henry
"""

import numpy as np
import datetime as dt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

from ftplib import FTP
import fnmatch
import datetime as dt
import sys
import os
import sys

from geopy import distance

from pybufrkit.decoder import Decoder
from  pybufrkit.renderer import FlatTextRenderer

def downloadFile(directory,filename):
    
    if not os.path.exists(directory):
        
        os.mkdir(directory)
        
    if not os.path.exists(directory+filename):
    
        localfile = open(directory+filename, 'wb')
        ftp.retrbinary('RETR ' + filename, localfile.write)
    
        localfile.close()
    
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

ftp = FTP("dissemination.ecmwf.int")

ftp.login("wmo","essential")

model_run = sys.argv[1]

directory = dt.datetime.now().strftime("%Y%m%d{}0000/").format(model_run)

directory = "20191106000000/"

ftp.cwd(directory)

filenames = ftp.nlst()

bufr_files = fnmatch.filter(filenames, "*bufr4*")

for filename in bufr_files:
    
    downloadFile(directory,filename)

ftp.quit()

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

for storm in [composite_storm_files[0]]:
    
    ens_path = storm[0]
    
    det_path = storm[1]
    
    # Decode a BUFR file
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
    
    df_ens = pd.DataFrame(list,columns=["Datetime","Latitude","Longitude","Pressure [hPa]","10m Wind Speed [m/s]","Ensemble member"]).set_index("Datetime")
    
    ens_member = df_ens["Ensemble member"]
    
    df_ens[df_ens[["Latitude","Longitude","Pressure [hPa]","10m Wind Speed [m/s]"]]=="None"]=np.nan
    
    df_ens["Pressure [hPa]"] = df_ens["Pressure [hPa]"].astype(float)/100
    
    df_ens["Ensemble member"] = ens_member
        
    # for member in np.arange(1,subset+1):
    #     a = ax.plot(df[df["Ensemble member"]==member]["Longitude"].values.astype(float),df[df["Ensemble member"]==member]["Latitude"].values.astype(float),"-",color="black",linewidth=0.5)
    #     ax.scatter(df[df["Ensemble member"]==member]["Longitude"].values.astype(float),df[df["Ensemble member"]==member]["Latitude"].values.astype(float),c=df[df["Ensemble member"]==member]["Pressure [hPa]"].values.astype(float))
    
    #set extent using ensemble

    #now plot deterministic
    
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
    
    df_det = pd.DataFrame(list,columns=["Datetime","Latitude","Longitude","Pressure [hPa]","10m Wind Speed [m/s]","Ensemble member"]).set_index("Datetime")
    
    ens_member = df_det["Ensemble member"]
    
    df_det[df_det[["Latitude","Longitude","Pressure [hPa]","10m Wind Speed [m/s]"]]=="None"]=np.nan
    
    df_det["Pressure [hPa]"] = df_det["Pressure [hPa]"].astype(float)/100
    
    df_det["Ensemble member"] = ens_member
    
    # for member in np.arange(1,subset+1):
    #     b = ax.plot(df[df["Ensemble member"]==member]["Longitude"].values.astype(float),df[df["Ensemble member"]==member]["Latitude"].values.astype(float),"-",color="red",linewidth=1)
    #     ax.scatter(df[df["Ensemble member"]==member]["Longitude"].values.astype(float),df[df["Ensemble member"]==member]["Latitude"].values.astype(float),c=df[df["Ensemble member"]==member]["10m Wind Speed [m/s]"].values.astype(float))
    
    #fig,ax = plt.subplots(figsize=(10,10),dpi=150)
        
    # ax.coastlines()

    # #ax.colorbar(label="10m Wind Speed [m/s]",orientation="horizontal",pad=0.02)
    # ax.legend([a[0],b[0]],["Ensemble","Deterministic"])
    # plt.savefig("{}_plot_wind.png".format(storm_name),bbox_inches="tight")
    # plt.close()
    
    plt.figure(figsize=(10,10),dpi=150)
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"))
    ax.add_feature(cfeature.OCEAN.with_scale("50m"))
    
    x0, x1, y0, y1 = [df_ens["Longitude"].astype(float).min()-2,df_ens["Longitude"].astype(float).max()+2,df_ens["Latitude"].astype(float).min()-2,df_ens["Latitude"].astype(float).max()+2]
    
    ax.set_extent([x0,x1,y0,y1])
    
    for member in np.arange(1,ens_subset+1):
        a = ax.plot(df_ens[df_ens["Ensemble member"]==member]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==member]["Latitude"].values.astype(float),"-",color="black",linewidth=0.5)
        s_ens = ax.scatter(df_ens[df_ens["Ensemble member"]==member]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==member]["Latitude"].values.astype(float),c=df_ens[df_ens["Ensemble member"]==member]["Pressure [hPa]"].values.astype(float))
        
    b = ax.plot(df_det[df_det["Ensemble member"]==1]["Longitude"].values.astype(float),df_det[df_det["Ensemble member"]==1]["Latitude"].values.astype(float),"-",color="red",linewidth=1)
    s_det = ax.scatter(df_det[df_det["Ensemble member"]==1]["Longitude"].values.astype(float),df_det[df_det["Ensemble member"]==1]["Latitude"].values.astype(float),c=df_det[df_det["Ensemble member"]==1]["Pressure [hPa]"].values.astype(float))
    c = ax.plot(df_ens[df_ens["Ensemble member"]==51]["Longitude"].values.astype(float),df_ens[df_ens["Ensemble member"]==51]["Latitude"].values.astype(float),"-",color="blue",linewidth=1)
    plt.title("ECMWF storm track for {} initialised {}Z {}".format(storm_name,str(time_0.hour).zfill(2),time_0.strftime("%d/%m/%Y")))
    plt.colorbar(s_ens,label="Surface Pressure [hPa]",pad=0.05,orientation="horizontal")
    plt.legend([a[0],b[0],c[0]],["Ensemble","Deterministic","Control"])
    
    res = 0.25
    
    xx, yy = np.meshgrid(np.arange(x0,x1+res,res),np.arange(y0,y1+res,res))
    
    prob = np.zeros((xx.shape[0],xx.shape[1]))
    
    progress = 0
    
    for j in range(xx.shape[1]-1):
        for i in range(xx.shape[0]-1):
            
            count = 0
            
            print(yy[i,j],xx[i,j])
            
            for lat,lon in df_ens[["Latitude","Longitude"]].values:
                
                try:
                
                    if distance.distance((yy[i,j],xx[i,j]),(lat,lon)).nm <= 100:
                    
                        count += 1
                        
                except ValueError:
                    
                    print("nan lat or lon value, skipping")
            
            prob[i,j] = (count/df_ens[["Latitude","Longitude"]].shape[0]) * 100

    
    plot = ax.contourf(xx,yy,prob,levels=[10,20,30,40,50,60,70,80,90,100])
               
    # if not os.path.exists(directory+"plot"):
        
    #     os.mkdir(directory+"plot")
    
    # plt.savefig(directory+"plot"+"/{}.png".format(storm_name),bbox_inches="tight")
    plt.show()
