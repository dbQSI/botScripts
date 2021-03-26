import math as m
import json as j
import os
import csv
import argparse as ap

def csv_dict_list(path):
    # Open variable-based csv, iterate over the rows and map values to a list of dictionaries containing key/value pairs
    reader = csv.DictReader(open(path, 'r'))
    dict_list = []
    for line in reader:
        dict_list.append(line)

    return dict_list

def convertHeading(heading):
    if (heading < 0):
       heading = heading + 360
    return heading

def haversineDistance(lat1, lat2, lon1, lon2):
    R = 6378137
    rLat1 = math.radians(lat1)
    rLat2 = math.radians(lat2)
    rDelLat= math.radians(lat2-lat1)
    rDelLon = math.radians(lon2-lon1)

    a = math.sin(rDelLat/2) * math.sin(rDelLat/2) +
        math.cos(rLat1) * math.cos(rLat2) *
        math.sin(rDelLon/2) * math.sin(rDelLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return (R * c)

def headingFromTwoPoints(lonStart, latStart, lonEnd, latEnd):
    lonStart = m.radians(lonStart)
    latStart = m.radians(latStart)
    lonEnd = m.radians(lonEnd)
    latEnd = m.radians(latEnd)

    X = m.cos(latEnd) * m.sin(lonEnd - lonStart)
    Y = (m.cos(latStart) * m.sin(latEnd)) - (m.sin(latStart) * m.cos(latEnd) * m.cos(lonEnd - lonStart))

    return round(m.degrees(m.atan2(X, Y)), 3)

def getStartEndFromGMPoints(globalMapperGeometryStr):
    #pull values from the Global Mapper "GEOMETRY" fields and strip any "LINESTRING" characters to get just LAT / LON values
    geom = globalMapperGeometryStr.strip('"LINESTRG)(')
    set1, set2 = geom.split(",")
    lonStart, latStart = (set1.strip()).split(" ")
    lonEnd, latEnd = (set2.strip()).split(" ")

    return {'latStart': float(latStart), 'lonStart': float(lonStart), 'latEnd': float(latEnd), 'lonEnd': float(lonEnd)}

def genInitialStations(csvDict):
    stationList = []
    #Build list of all station entries and their azimuth values based on start / end point positions
    for row in csvDict:
        geom = getStartEndFromGMPoints(row['GEOMETRY'])
        heading = convertHeading(headingFromTwoPoints(geom['lonStart'], geom['latStart'], geom['lonEnd'], geom['latEnd']))

        stationList.append({'INDEX': row['LABEL'], 'LAT': geom['latStart'], 'LON': geom['lonStart'], 'HEAD': [int(heading)]})

    return stationList

def removeDuplicatesMergeHeadings(stationList):

    # set 'DUPE' key to False
    for station in stationList:
        station['DUPE'] = False

    newStationList = []
    i = -1
    # Check for LAT / LON dupes, merge heading, and set DUPE to True
    for station in stationList:
        i += 1
        k = -1
        for check in stationList:
            k += 1
            if (i >= k):
                continue

            if (station['LAT'] == check['LAT']) and (station['LON'] == check['LON']):
                station['HEAD'].append(check['HEAD'][0])
                check['DUPE'] = True
    # remove the dupes from the list
    for rm in stationList:
        if rm['DUPE'] == True:
            continue
        else:
            newStationList.append(rm)

    return newStationList

def buildGeoJsonFromStations(stationList):
    ftrList = []
    for station in stationList:
        # print(makeGeoJson_PointFeatureEntry(station['LON'], station['LAT'], {'Directions': station['HEAD'], 'Name': station['INDEX']}))
        ftrList.append(makeGeoJson_PointFeatureEntry(station['LON'], station['LAT'], {'Directions': station['NEW_HEAD'], 'Name': station['INDEX']}))

    jsonStr = makeGeoJson_FeatureCollection(ftrList)

    return jsonStr

def transformHeadingValsForLeaflet(stationList):
    for station in stationList:
        station['NEW_HEAD'] = []
        for head in station['HEAD']:
            station['NEW_HEAD'].append(convertHeading(head - 180))
    return stationList

def makeGeoJson_PointFeatureEntry(X, Y, propertiesDict):
    ptFeature = { "type": "Feature", "properties": {}, "geometry": {"type": "Point", "coordinates": [X, Y]}}
    for k, prop in propertiesDict.items():
        ptFeature['properties'][str(k)] = prop

    return ptFeature

def makeGeoJson_FeatureCollection(listOfFeatures):
    ftrCollection = {"type": "FeatureCollection", "features": []}
    for ftr in listOfFeatures:
        ftrCollection['features'].append(ftr)

    return j.dumps(ftrCollection, indent=4, separators=(',', ': '))

def genStations(stationVectorsCsv):
    fileName, fileExt = os.path.splitext(stationVectorsCsv)

    csvList = csv_dict_list(stationVectorsCsv)
    stationList = genInitialStations(csvList)
    noDupesList = removeDuplicatesMergeHeadings(stationList)
    newHeadingsList = transformHeadingValsForLeaflet(noDupesList)
    jsonStr = buildGeoJsonFromStations(newHeadingsList)

    geoJsonFile = open(fileName + ".geojson", "w")
    geoJsonFile.write(jsonStr)
    geoJsonFile.close()

    return

if __name__ == "__main__":
    parser = ap.ArgumentParser(description='supply csv generated from vectors drawn and exported from Global Mapper with GEOGRAPHIC projection (talk to DB about instructions for generating correct file)')
    argGroup = parser.add_argument_group(title='Inputs')
    argGroup.add_argument('-csv', dest='csvPath', required=True, help='input path to Global Mapper vector CSV export.')

    args = parser.parse_args()

    if args.csvPath:
        genStations(args.csvPath)
    else:
        print("no input entered.  run with '-h' to learn about required inputs.")
