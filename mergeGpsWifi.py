## currently the files need to be cleaned up (remove decimals after gps Time values), changed to comma deliminated and given appropriate header names.  Merge is done on 'Time' field

import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import numpy as np
import argparse as ap
import os

def GpsHMStoStandard(gpsYear, gpsMonth, gpsDay, gpsHour, gpsMinute, gpsSecond, gpsDecimal=0):
    gpsStart = dt.datetime(year=1980, month=1, day=6, hour=0, minute=0, second=0)
    gpsThis = dt.datetime(year=gpsYear, month=gpsMonth, day=gpsDay, hour=gpsHour, minute=gpsMinute, second=gpsSecond, microsecond=int(1000000 * gpsDecimal))
    standardTime = gpsThis - gpsStart

    return standardTime.total_seconds()

def mergeGpsWifi(gpsFilePath, wifiFilePath, outputPath):

    gpsDF = pd.read_csv(gpsFilePath, index_col=False)
    wifiDF = pd.read_csv(wifiFilePath, index_col=False)

    gpsDF['TIME'] = pd.to_datetime(gpsDF[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']])
    wifiDF['TIME'] = pd.to_datetime(wifiDF[['YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND']])

    result = pd.merge(gpsDF, wifiDF, left_on='TIME', right_on='TIME')

    result.plot(x='QUALITY', y='TIME')
    result.plot(x='SIGNAL', y='TIME')
    #
    result.to_csv(outputPath + "/merged.csv")

    return

def formatWifiLog(wifiFilePath):
    wifiFile = open(wifiFilePath, "r")
    wifiVar = wifiFile.read()
    wifiFile.close()

    wifiVar = wifiVar.replace("\n  ", " ")
    wifiVar = wifiVar.replace("  ", " ")
    wifiVar = wifiVar.replace(" \n", "\n")
    wifiVar = wifiVar.replace(" ", ",")
    wifiVar = wifiVar.replace(".", "")
    wifiVarWithHead = "YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,QUALITY,SIGNAL\n" + wifiVar

    name, ext = os.path.splitext(wifiFilePath)
    newPath = name + "_converted.csv"
    newWifiFile = open(newPath, "w")
    newWifiFile.write(wifiVarWithHead)
    newWifiFile.close()

    return newPath

def formatGpsLLH(gpsFilePath):
    gpsFile = open(gpsFilePath, "r")
    gpsVar = gpsFile.read()
    gpsFile.close()

    gpsVar = gpsVar.replace("   ", " ")
    gpsVar = gpsVar.replace("  ", " ")
    gpsVar = gpsVar.replace(" \n", "\n")
    gpsVar = gpsVar.replace("/", " ")
    gpsVar = gpsVar.replace(":", " ")
    gpsVar = gpsVar.replace(" ", ",")
    gpsVarWithHead = "YEAR,MONTH,DAY,HOUR,MINUTE,SECOND,LATITUDE,LONGITUDE,ALTITUDE,FIX,NUM OF SATS,SDN,SDE,SDU,SDNE,SDEU,SDUN,AGE,RATIO\n" + gpsVar

    name, ext = os.path.splitext(gpsFilePath)
    newPath = name + "_converted.csv"
    newGpsFile = open(newPath, "w")
    newGpsFile.write(gpsVarWithHead)
    newGpsFile.close()

    return newPath

def pandsToGeopandas(pdDF):
    pdDF['geometry'] = pdDF.apply(lambda z: Point(z.LONGITUDE, z.LATITUDE), axis=1)
    gpDF = gp.GeoDataFrame(pdDF)

    return gpDF

if __name__ == "__main__":
    parser = ap.ArgumentParser(description='Run script to parse raw wifi logs and GPS "LLH" log from Rover Reach to generate wifi and GPS quality maps.')
    argGroup = parser.add_argument_group(title='Files')
    argGroup.add_argument('-iw', dest='wifiPath', required=False, help='input file path to wifi log file from robot (the one generated from "print-wifi.sh" execution).  ie "-iw path/to/20190725_wifi.log".')
    argGroup.add_argument('-ig', dest='gpsPath', required=False, help='input file path to gps LLH file from robot Reach module.  ie "-ig path/to/solution_201907252050.LLH".')
    argGroup.add_argument('-o', dest='outputPath', required=False, help='input directory path for output KMLs.')

    args = parser.parse_args()

    if args.wifiPath and args.gpsPath:
        newWifiPath = formatWifiLog(args.wifiPath)
        newGpsPath = formatGpsLLH(args.gpsPath)
        mergeGpsWifi(newGpsPath, newWifiPath, args.outputPath)
