#!/usr/bin/env python
# -*- coding: utf-8 -*-

import boto3
import botocore
from collections import namedtuple
from operator import attrgetter
import pdb
import pandas as pd
import geopandas as gp
from shapely.geometry import Point
import sqlalchemy as sql
import os
import shutil
from PIL import Image, ImageDraw, ImageOps, ImageEnhance
import cv2
import json
import piexif
import numpy as np
import math
import csv
import progressbar as progress
import datetime as dt
import logging


S3Obj = namedtuple('S3Obj', ['key', 'mtime', 'size', 'ETag'])


def s3list(bucket, path, start=None, end=None, recursive=True, list_dirs=True, list_objs=True, limit=None):
    kwargs = dict()
    if start is not None:
        if not start.startswith(path):
            start = os.path.join(path, start)
        # note: need to use a string just smaller than start, because
        # the list_object API specifies that start is excluded (the first
        # result is *after* start).
        kwargs.update(Marker=__prev_str(start))
    if end is not None:
        if not end.startswith(path):
            end = os.path.join(path, end)
    if not recursive:
        kwargs.update(Delimiter='/')
        if not path.endswith('/'):
            path += '/'
    kwargs.update(Prefix=path)
    if limit is not None:
        kwargs.update(PaginationConfig={'MaxItems': limit})

    paginator = bucket.meta.client.get_paginator('list_objects')
    for resp in paginator.paginate(Bucket=bucket.name, **kwargs):
        q = []
        if 'CommonPrefixes' in resp and list_dirs:
            q = [S3Obj(f['Prefix'], None, None, None) for f in resp['CommonPrefixes']]
        if 'Contents' in resp and list_objs:
            q += [S3Obj(f['Key'], f['LastModified'], f['Size'], f['ETag']) for f in resp['Contents']]
        # note: even with sorted lists, it is faster to sort(a+b)
        # than heapq.merge(a, b) at least up to 10K elements in each list
        q = sorted(q, key=attrgetter('key'))
        if limit is not None:
            q = q[:limit]
            limit -= len(q)
        for p in q:
            if end is not None and p.key >= end:
                return
            yield p


def __prev_str(s):
    if len(s) == 0:
        return s
    s, c = s[:-1], ord(s[-1])
    if c > 0:
        s += chr(c - 1)
    s += ''.join(['\u7FFF' for _ in range(10)])
    return s


def generateBearingsFromEoList(eoList, offset=0):
    index = -1
    total = len(eoList)
    for row in eoList:
        index += 1
        if index == 0:
            continue

        bearing = headingFromTwoPoints(float(eoList[index - 1]['LON']), float(eoList[index - 1]['LAT']), float(eoList[index]['LON']), float(eoList[index]['LAT']))
        eoList[index - 1]['HEADING'] = convertHeading(bearing + offset)


    eoList[total - 1]['HEADING'] = eoList[total - 2]['HEADING']
    return eoList


def convertHeading(heading):
    if (heading < 0):
        heading = heading + 360
    elif (heading > 360):
        heading = heading - 360
    return heading


def headingFromTwoPoints(lonStart, latStart, lonEnd, latEnd):
    lonStart = math.radians(lonStart)
    latStart = math.radians(latStart)
    lonEnd = math.radians(lonEnd)
    latEnd = math.radians(latEnd)

    X = math.cos(latEnd) * math.sin(lonEnd - lonStart)
    Y = (math.cos(latStart) * math.sin(latEnd)) - (math.sin(latStart) * math.cos(latEnd) * math.cos(lonEnd - lonStart))

    return round(math.degrees(math.atan2(X, Y)), 3)


def convertDDtoDMS(decimalDeg):
    decimalDeg = abs(decimalDeg)
    deg = math.trunc(decimalDeg)
    decMin = (decimalDeg - deg) * 60
    min = math.trunc(decMin)
    sec = int(round(((decMin - min) * 60), 4) * 10000)

    return ((deg, 1), (min, 1), (sec, 10000))


def getDBtoDF(resource, keyPath, bucketName, filePath, sqlName):
    keyDir, keyFile = os.path.split(keyPath)
    resource.Bucket(bucketName).download_file(keyPath, keyFile)
    conn = sql.create_engine(f'sqlite:///{filePath}', echo = False)
    # conn = sqlite3.connect('NEBucket.db')
    db = pd.read_sql_table(sqlName, conn).drop(['index'], axis=1)
    return db


def sendDBfromDF(resource, keyPath, bucketName, dbDF, sqlName, filePath):
    conn = sql.create_engine(f'sqlite:///{sqlName}.db', echo = False)
    dbDF.to_sql(sqlName, con=conn, if_exists='replace')
    try:
        resource.Bucket(bucketName).upload_file(Filename=filePath, Key=keyPath)
    except botocore.exceptions.ClientError as e:
        logging.exception("ClientError")
        # print (str(e))
        raise
    return


def findNewObjs(dbDF, bucketDF, Subset=None):
    # newObjsDF = pd.concat([dbDF,bucketDF]).drop_duplicates(subset=Subset, keep=False)
    newObjsDF = dbDF.append(bucketDF, ignore_index=True).drop_duplicates(subset=Subset, keep=False)
    # addedNewDF = pd.concat([dbDF,bucketDF]).drop_duplicates(subset=Subset, keep='first')
    addedNewDF = dbDF.append(bucketDF, ignore_index=True).drop_duplicates(subset=Subset, keep='last')
    return (newObjsDF, addedNewDF)


def pandsToGeopandas(pdDF):

    pdDF['geometry'] = pdDF.apply(lambda z: Point(z.longitude, z.latitude), axis=1)
    gpDF = gp.GeoDataFrame(pdDF)
    gpDF.crs = 'epsg:4326'

    return gpDF


def addJSONfields(dbDF, urlPrefix):
    dbDF['lat'] = [z['latitude'] for index, z in dbDF.iterrows()]
    dbDF['lng'] = [z['longitude'] for index, z in dbDF.iterrows()]
    dbDF['url'] = [urlPrefix + z['filekey'] for index, z in dbDF.iterrows()]
    dbDF['thumbnail'] = [urlPrefix + z['thumbnail'] for index, z in dbDF.iterrows()]
    dbDF = dbDF.drop(['geometry'], axis=1)
    return dbDF


def DivideDFByTime(dbDF):
    uniqueTimeDF = dbDF.drop_duplicates(subset=['time'], keep='last')
    sortedList = []
    for index, z in uniqueTimeDF.iterrows():
         sortedList.append(dbDF[dbDF['time'] == z['time']])

    return sortedList


def generateJsonScriptForTimeSlider(sortedDFsList, fileName):
    layerGroup = ''
    with open(f"./{fileName}.json", 'w') as jf:
        jf.write('sidebar.open("imageView");\n')
        jf.write('var picViewer = L.map("picView", {\n')
        jf.write('  center: [0.0, 0.0], \n')
        jf.write('  zoom: 8, \n')
        jf.write('  maxBounds: [[-4.0, -3.0], [4.0, 3.0]], \n')
        jf.write('  minZoom: 8 \n')
        jf.write('});\n\n' )
        jf.write('sidebar.close();\n')

        jf.write('sidebar.open("thermView");\n')
        jf.write('var thermViewer = L.map("thermalView", {\n')
        jf.write('  center: [0.0, 0.0], \n')
        jf.write('  zoom: 12, \n')
        jf.write('  maxBounds: [[-0.35, -0.25], [0.47, 0.4]], \n')
        jf.write('  minZoom: 12 \n')
        jf.write('});\n\n' )
        jf.write('sidebar.close();\n')

        for sl in sortedDFsList:
            time = sl.iloc[0]['time'].replace('-', '_')
            timeName = f'meta_{time}'
            layerName = f'layer_{time}'
            layerGroup += f'{layerName}, '
            jf.write(f'var {timeName} = ')
            jf.write(sl.to_json(orient='records') + '\n')
            # if (sl.iloc[0]['spectral_prefix'] == 'LWIR'):
            layerSetup = f"""var {layerName} = L.photo.cluster().on('click', function(evt) {{
                                var photo = evt.layer.photo;
                                var template = photo.url;
                                var namearr = template.split('/');
                                var filename = namearr[namearr.length - 1];
                                if (photo.spectral_prefix == "LWIR") {{
                                    thermViewer.eachLayer(function (layer) {{
                                        thermViewer.removeLayer(layer);
                                    }});
                                    document.getElementById("dwnld2").href = template;
                                    ascPath = template.replace(".csv.tif", ".asc")
                                    sidebar.open('thermView');
                                    d3.text(ascPath, function (asc) {{
                                        var s = L.ScalarField.fromASCIIGrid(asc);
                                        newRange = [s.range[1], s.range[0]];
                                        var thermLayer = L.canvasLayer.scalarField(s, {{
                                            color: chroma.scale('thermal').domain(newRange),
                                        }}
                                        ).addTo(thermViewer);

                                        thermLayer.on('click', function (e) {{
                                            if (e.value !== null) {{
                                                let v = e.value.toFixed(3);
                                                let html = `<span class="popupText">${{v}} Deg (F)</span>`;
                                                let popup = L.popup().setLatLng(e.latlng).setContent(html).openOn(thermViewer);
                                            }}
                                        }});
                                        let colorBrewer = document.getElementById('colorBrewer');
                                        colorBrewer.addEventListener('change', function () {{
                                            var scale = chroma.scale(this.value).domain(newRange);
                                            thermLayer.setColor(scale);
                                        }});
                                    }});
                                    thermViewer.setView([0.06, 0.078], 11);
                                }}
                                else {{
                                    picViewer.eachLayer(function (layer) {{
                                        picViewer.removeLayer(layer);
                                    }});
                                    document.getElementById("dwnld").href = template;
                                    sidebar.open('imageView');
                                    var imageBounds = [[-1.0000, -1.0000], [1.0000, 1.00000]];
                                    L.imageOverlay(template, imageBounds).addTo(picViewer);
                                    picViewer.setView([0.0000, 0.0000], 8);
                                }};
                            }});\n"""
            # else:
            #     layerSetup = f"""var {layerName} = L.photo.cluster().on('click', function(evt) {{
            #                     picViewer.eachLayer(function (layer) {{
            #                         picViewer.removeLayer(layer);
            #                     }});
            #                     var photo = evt.layer.photo;
            #                     var template = photo.url;
            #                     var namearr = template.split('/');
            #                     var filename = namearr[namearr.length - 1];
            #                     document.getElementById("dwnld").href = template;
            #                     sidebar.open('imageView');
            #                     var imageBounds = [[-1.0000, -1.0000], [1.0000, 1.00000]];
            #                     L.imageOverlay(template, imageBounds).addTo(picViewer);
            #                     picViewer.setView([0.0000, 0.0000], 8);
            #                     }});\n"""
            jf.write(layerSetup)
            jf.write(f'{layerName}.add({timeName});\n')
            jf.write(f'{layerName}.options.time = "{sl.iloc[0]["time"]}"\n')

        jf.write(f"var multiLayers = L.layerGroup([{layerGroup}]);\n")

    return


def sendGJSHPfromDF(resource, keyPath, bucketName, dbDF, fileName):

    dbGDF = pandsToGeopandas(dbDF)
    dbGDF.to_file(f"./{fileName}.geojson", driver='GeoJSON')
    try:
        dbGDF.to_file(f"./{fileName}", driver='ESRI Shapefile')
    except UserWarning:
        logging.exception("SHP error")
        pass
    shutil.make_archive(fileName, 'zip', root_dir=fileName)
    shutil.rmtree(fileName)
    dbDF = addJSONfields(dbDF, 'https://dp4479dqtum7u.cloudfront.net/')
    sortedDFsList = DivideDFByTime(dbDF)
    generateJsonScriptForTimeSlider(sortedDFsList, fileName)

    try:
        resource.Bucket(bucketName).upload_file(Filename=f"./{fileName}.geojson", Key=f'{keyPath}.geojson', ExtraArgs={'ContentType': "text/javascript", 'ACL': "public-read"})
        resource.Bucket(bucketName).upload_file(Filename=f"./{fileName}.zip", Key=f'{keyPath}.zip')
        resource.Bucket(bucketName).upload_file(Filename=f"./{fileName}.json", Key=f'{keyPath}.json', ExtraArgs={'ContentType': "text/javascript", 'ACL': "public-read"})
    except botocore.exceptions.ClientError as e:
        logging.exception("ClientError")
        # print (str(e))
        raise

    return


def runCv2Clahe(PilImage, spherical=False, clipLimit=1.5, tileGridSize=(25,25)):
    #PIL image array to cv2
    npImage = np.asarray(PilImage)

    if spherical == True:
        height, width, chan = npImage.shape
        bufferSize = int(width / tileGridSize[0])
        npImage = cv2.copyMakeBorder(npImage,0,0,bufferSize,bufferSize,cv2.BORDER_WRAP)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    if len(npImage.shape) < 3:
        rgbImg = clahe.apply(npImage)

    else:
        # lab = cv2.cvtColor(npImage, cv2.COLOR_RGB2LAB)
        # lab_planes = cv2.split(lab)
        # lab_planes[0] = clahe.apply(lab_planes[0])
        # lab = cv2.merge(lab_planes)
        # rgbImg = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        hsv = cv2.cvtColor(npImage, cv2.COLOR_RGB2HSV)
        hsv_planes = cv2.split(hsv)
        hsv_planes[2] = clahe.apply(hsv_planes[2])
        hsv = cv2.merge(hsv_planes)
        rgbImg = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    if spherical == True:
        rgbImg = rgbImg[:, bufferSize:(width + bufferSize)]

    pilImage = Image.fromarray(rgbImg)

    return pilImage


def runCv2ClaheCb(PilImage, spherical=False, clipLimit=1.5, tileGridSize=(25,25)):
    #PIL image array to cv2
    npImage = np.asarray(PilImage)

    if spherical == True:
        height, width, chan = npImage.shape
        bufferSize = int(width / tileGridSize[0])
        npImage = cv2.copyMakeBorder(npImage,0,0,bufferSize,bufferSize,cv2.BORDER_WRAP)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    if len(npImage.shape) < 3:
        return PilImage

    hsvImg = cv2.cvtColor(npImage, cv2.COLOR_RGB2HSV)
    hsv_planes = cv2.split(hsvImg)
    hsv_planes[1] = clahe.apply(hsv_planes[1])
    hsvImg = cv2.merge(hsv_planes)
    rgbImg = cv2.cvtColor(hsvImg, cv2.COLOR_HSV2RGB)

    if spherical == True:
        rgbImg = rgbImg[:, bufferSize:(width + bufferSize)]

    pilImage = Image.fromarray(rgbImg)

    return pilImage


def setGpsTags(inputImg, outputDir, lat, lon, elev, heading, runClahe=True, claheClip=40, claheTile=(2,2), colorBoost=False, cbClip=0.5, cbTile=(2,2), jpegQuality=90):

    path, file = os.path.split(inputImg)
    fileName, fileExt = os.path.splitext(file)
    img = Image.open(inputImg)
    try:
        exifData = img.info["exif"]
    except KeyError:
        exifDict = {"0th": {}, "Exif": {}, "1st": {},
        "thumbnail": None, "GPS": {}}
    else:
        exifDict = piexif.load(exifData)
    if runClahe == True:
        img = runCv2Clahe(img, False, claheClip, claheTile)
    if colorBoost == True:
        img = runCv2ClaheCb(img, False, cbClip, cbTile)

    if (lat > 0):
        exifDict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "N"
    else:
        exifDict["GPS"][piexif.GPSIFD.GPSLatitudeRef] = "S"
    if (lon > 0):
        exifDict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "E"
    else:
        exifDict["GPS"][piexif.GPSIFD.GPSLongitudeRef] = "W"

    exifDict["GPS"][piexif.GPSIFD.GPSLatitude] = convertDDtoDMS(lat)
    exifDict["GPS"][piexif.GPSIFD.GPSLongitude] = convertDDtoDMS(lon)

    if elev < 0:
        exifDict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 1
    else:
        exifDict["GPS"][piexif.GPSIFD.GPSAltitudeRef] = 0

    exifDict["GPS"][piexif.GPSIFD.GPSAltitude] = (int(round(abs(elev), 2) * 100), 100)


    exifDict["GPS"][piexif.GPSIFD.GPSImgDirectionRef] = "T"
    exifDict["GPS"][piexif.GPSIFD.GPSImgDirection] = (int(round(convertHeading(heading), 2) * 100), 100)

    exifTo_bytes = piexif.dump(exifDict)
    if img.mode == 'I;16':
        # img = img.point(lambda i:i*(1./256)).convert('L')
        img = img.point(lambda i:i*(1./256)).convert('L')
    img.save(f'{outputDir}/{file}.jpg', 'jpeg', exif=exifTo_bytes, quality=jpegQuality)
    img.thumbnail((128,128))
    img.save(f'{outputDir}/{file}.thumbnail', 'jpeg')

    return


def csvToTiff(inPath, outPath, convert=False, exif=False):
    def C2F(cTemp):
    	return (cTemp * (9 / 5)) + 32

    xyzs = []
    max_x,max_y = 0,0
    x,y = 0,0
    for xyz in csv.reader(open(inPath)):
        if len(xyz)<50: continue # ignore [['header: 61445'], ['footer: 61440']]
        row = []
        for s in xyz:
            if len(s) > 2:
                row.append(float(s))
            else:
                continue

        xyzs.append(row)

        x = len(xyz)
        y += 1

        if x>max_x: max_x = x
        if y>max_y: max_y = y

    im = Image.new("F", (max_x,max_y), )
    with (open(f'{outPath}.asc', 'w') as ascFile):
        ascFile.write(f'ncols        {(max_x-1)}\nnrows        {max_y}\nxllcorner    0.0\nyllcorner    0.0\ncellsize     0.00050505\nNODATA_value  -9999')
        y = -1
        for row in xyzs:
            ascFile.write('\n')
            y += 1
            x = -1
            for val in row:
                x += 1
                if convert == True: val = C2F(val)
                ascFile.write(f' {val}')
                im.putpixel((x,y), val)

    im.save( open(outPath + ".csv.tif",'wb'), format='TIFF' )
    imT = im.convert('RGB')
    imT.thumbnail((128,128))
    imT.save( open(outPath + ".thumbnail",'wb'), format='jpeg' )

    if exif:
        im.save(outPath + ".jpg", quality=95)

    return


def evalJsonData(jsonData, keyFile):
    success = False
    metaobj = {}
    if jsonData.get('metadata'):
        try:
            metaobj['latitude'] = float(jsonData['metadata']['gps']['latitude'])
            metaobj['longitude'] = float(jsonData['metadata']['gps']['longitude'])
            metaobj['altitude'] = float(jsonData['metadata']['gps']['altitude'])
            metaobj['heading'] = float(jsonData['metadata']['ptu']['heading'])
            metaobj['tilt'] = float(jsonData['metadata']['ptu']['tilt'])
            metaobj['pan'] = float(jsonData['metadata']['ptu']['pan'])
            metaobj['W_human'] = jsonData['metadata']['metar']['human_readable']
            metaobj['W_timestamp'] = jsonData['metadata']['metar']['timestamp']
            metaobj['W_temperatureF'] = float(jsonData['metadata']['metar']['temp_f'])
            metaobj['W_temperatureC'] = float(jsonData['metadata']['metar']['temp_c'])
            metaobj['W_validation'] = jsonData['metadata']['metar']['valid']
            metaobj['W_visibility'] = jsonData['metadata']['metar']['visibility']
            metaobj['W_metar'] = jsonData['metadata']['metar']['metar_string']
            success = True
        except (KeyError, TypeError) as e:
            logging.exception("KeyError/TypeError")
            # print(f'KeyError/TypeError {e} at {keyFile}')
            success = False

    if jsonData.get('gps') and success == False:
        try:
            metaobj['latitude'] = float(jsonData['gps']['lat'])
            metaobj['longitude'] = float(jsonData['gps']['lng'])
            metaobj['altitude'] = float(jsonData['gps']['alt'])
            metaobj['heading'] = float(jsonData['imu']['yaw'])
            success = True
        except (KeyError, TypeError) as e:
            logging.exception("KeyError/TypeError")
            # print(f'KeyError/TypeError {e} at {keyFile}')
            success = False

    try:
        metaobj['spectral_prefix'] = jsonData['cdp']['spectral_prefix']
        metaobj['when'] = dt.datetime.isoformat(dt.datetime.strptime(jsonData['gps']['when'], '%Y-%m-%dT%H:%M:%S.%fZ'))
        metaobj['time'] = dt.datetime.strptime(jsonData['gps']['when'], '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
        metaobj['exposure_us'] = int(jsonData['cdp']['exposure_us'])
        metaobj['focal_length'] = float(jsonData['cdp']['focal_len_mm'])
        metaobj['px_H'] = int(jsonData['cdp']['pixel_height'])
        metaobj['px_W'] = int(jsonData['cdp']['pixel_width'])
    except (KeyError, TypeError) as e:
        logging.exception("KeyError/TypeError")
        # print(f'KeyError/TypeError {e} at {keyFile}')
        success = False

    if success == True:
        return metaobj
    else:
        return False


def streamImgProcessing(newObjsDF, resource, bucketName):
    #ugly function to stream files in/out of EC2 VM to watch AWS bucket for changes
    metaDF = []
    #iterate thru all new file objects
    progBar = progress.ProgressBar(max_value=len(newObjsDF))
    barIndex = 0
    with progBar as bar:
        for index, row in newObjsDF.iterrows():
            bar.update(barIndex)
            barIndex += 1

            keyDir, keyFile = os.path.split(row['key'])
            keyName, keyExt = os.path.splitext(keyFile)
            keySplit = keyFile.split('.')
            #skip all the LST bots for now as they have no metadata to speak of
            if ('qr-h001' or 'qr-h002' or 'qr-h003' or 'qr-h004' or 'qr-h005' or 'qr-h006' or 'qr-h007' or 'qr-j001') in row['key']:
                continue
            # skip empty keyFiles as this means the key is a directory name
            if keyFile == '':
                continue
            # exclude the thermal original '.tifs' which are improperly exposed (or maybe weird gamma settings?) versions of what the user see's when they take a thermal image: not useful
            if keyExt == '.tif' and 'LWIR_' in keySplit[0]:
                continue

            # grab all other files and filter after download...I may change this to only look at tifs/csvs/jsons if this is costly or taxing to the VM
            outAWSDir = keyDir.replace('ARIS_NextEra', 'ARIS_NextEra_out', 1)
            # we're only interested in TIFs and CSVs, along ith associated JSON files.  First we download each image and then the JSON, then we parse, process and organize
            if any(x == keyExt for x in ('.tif', '.csv')):
                jsonPath = keyDir + '/' + keySplit[0] + '.lraw.json'
                resource.Bucket(bucketName).download_file(row['key'], f'./tmp/{keyFile}')
                try:
                    resource.Bucket(bucketName).download_file(jsonPath, f'./tmp/{keySplit[0]}.lraw.json')
                except botocore.exceptions.ClientError as e:
                    if e.response['Error']['Code'] == "404":
                        logging.exception("The object does not exist.  skipping...")
                        # print("The object does not exist.  skipping...")
                        continue
                    else:
                        logging.exception("ClientError")
                        raise
                # Read the JSON in and start building DataFrame
                with open(f'./tmp/{keySplit[0]}.lraw.json', "r") as jsonFile:
                    jsonData = json.load(jsonFile)
                    metadata = evalJsonData(jsonData, row['key'])
                    #if the metadata is screwed up or missing, just skip the file and move on
                    if metadata == False:
                        logging.debug(f"metadata error.  skipping file {keySplit[0]}")
                        # print('metadata error.  skipping file.')
                        continue
                # '.gray.tif' files are 8-bit versions of the LWIR csv's.  These can be used on maps to visualize thermal data without opening it.
                if keyFile.endswith('.gray.tif'):
                    fileKey = f'{outAWSDir}/{keySplit[0]}.jpg'
                    fileThumbnail = f'{outAWSDir}/{keySplit[0]}.thumbnail'
                    # inject EXIF into image trying new metadata fields first, but if that fails using the old Lucint fields as fall back
                    try:
                        setGpsTags(f'./tmp/{keyFile}', './tmp/output', metadata['latitude'],
                                    metadata['longitude'], metadata['altitude'],
                                    metadata['heading'], runClahe=False, colorBoost=False)
                    except (KeyError, TypeError) as e:
                        logging.exception("KeyError/TypeError")
                        # print(f'no {e} key found')
                        continue
                    # Upload new images to output bucket on AWS
                    try:
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.jpg', Key=f'{outAWSDir}/{keySplit[0]}.jpg')
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.thumbnail', Key=f'{outAWSDir}/{keySplit[0]}.thumbnail')
                    except botocore.exceptions.ClientError as e:
                        logging.exception("ClientError")
                        # print (str(e))
                        raise

                # work on other tifs: NIR tifs & RGB tifs which are 1 channel uint16 and 3 channel 8-bit respectively
                elif keyExt == '.tif' and 'LWIR_' not in keySplit[0]:
                    fileKey = f'{outAWSDir}/{keySplit[0]}.jpg'
                    fileThumbnail = f'{outAWSDir}/{keySplit[0]}.thumbnail'
                    # inject EXIF and save to output folder
                    try:
                        setGpsTags(f'./tmp/{keyFile}', './tmp/output', metadata['latitude'],
                                    metadata['longitude'], metadata['altitude'], metadata['heading'])
                    except (KeyError, TypeError) as e:
                        logging.exception("KeyError/TypeError")
                        # print(f'no {e} key found')
                        continue
                    # try uploading to AWS output object
                    try:
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.jpg', Key=f'{outAWSDir}/{keySplit[0]}.jpg')
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.thumbnail', Key=f'{outAWSDir}/{keySplit[0]}.thumbnail')
                    except botocore.exceptions.ClientError as e:
                        logging.exception("ClientError")
                        # print (str(e))
                        raise
                # process the thermal csv's which actually represent the thermal values in celcius (which get converted to F during the 'csvTotiff' function)
                elif keyExt == '.csv':
                    fileKey = f'{outAWSDir}/{keySplit[0]}.csv.tif'
                    fileThumbnail = f'{outAWSDir}/{keySplit[0]}.thumbnail'
                    csvToTiff(f'./tmp/{keyFile}', f'./tmp/output/{keyFile}', convert=True)
                    # try to upload to AWS object
                    try:
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.asc', Key=f'{outAWSDir}/{keySplit[0]}.asc')
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.csv.tif', Key=f'{outAWSDir}/{keySplit[0]}.csv.tif')
                        resource.Bucket(bucketName).upload_file(Filename=f'./tmp/output/{keyFile}.thumbnail', Key=f'{outAWSDir}/{keySplit[0]}.thumbnail')
                    except botocore.exceptions.ClientError as e:
                        logging.exception("ClientError")
                        # print (str(e))
                        raise

                # add metadata to dataframe, trying new first, then old.  This will land in a GeoJson file for later consumption and a sqlite DB for general use
                metadata['fileOrigin'] = row['key']
                metadata['ETag'] = row['ETag']
                metadata['size'] = row['size']
                metadata['filekey'] = fileKey
                metadata['thumbnail'] = fileThumbnail
                metaDF.append(metadata)

                #delete the tmp folder to clear working data
                try:
                    shutil.rmtree('./tmp')
                except OSError as e:
                    logging.exception("OSError")
                    # print("Error: %s : %s" % ('./tmp', e.strerror))
                os.makedirs('./tmp/output', exist_ok=True)

    metadataDF = pd.DataFrame(metaDF)

    return metadataDF


def main(bucketName, resource):
    #setup variable names and paths, and check output / processing folder and delete
    bucket = resource.Bucket(name=bucketName)
    prefix = 'home/rclone/ARIS_NextEra'
    dbPath = 'home/rclone/ARIS_NextEra/NEBucket.db'
    dbName = 'NEBucket'
    metaDBPath = 'home/rclone/ARIS_NextEra_out/NEMetadata.db'
    metaDBName = 'NEMetadata'
    metaGJSHPPath = 'home/rclone/ARIS_NextEra_out/NEMetadata'


    try:
        shutil.rmtree('./tmp')
    except OSError as e:
        logging.exception("exception occurred")
        # print("Error: %s : %s" % ('./tmp', e.strerror))
    os.makedirs('./tmp/output', exist_ok=True)

    # get full listing of 'ARIS_NextEra' bucket and convert to DF
    listing = s3list(bucket, prefix)
    directory = [[o.key, o.mtime, o.size, o.ETag] for o in listing]
    listingDF = pd.DataFrame(directory, columns=['key', 'datetime', 'size', 'ETag'])
    # get DB of processed files from bucket and compare and check for new files
    dbDF = getDBtoDF(resource, dbPath, bucketName, f'{dbName}.db', dbName)
    newObjsDF, addedNewDF = findNewObjs(dbDF, listingDF, Subset=['key', 'size', 'ETag'])
    # run streaming image processing > download img > adjust img > inject geo EXIF data > collect metadata into DF > upload img and delete temp files > return DF
    newMetadataDF = streamImgProcessing(newObjsDF, resource, bucketName)
    # get current metadata DB to DF and merge metadata DFs and drop dupes
    metadataDF = getDBtoDF(resource, metaDBPath, bucketName, f'{metaDBName}.db', metaDBName)
    combinedMetadataDF = findNewObjs(metadataDF, newMetadataDF, Subset=['filekey', 'time', 'latitude', 'longitude'])[1]
    # append new metadata to current metadata DF and upload
    sendDBfromDF(resource, metaDBPath, bucketName, combinedMetadataDF, metaDBName, f'./{metaDBName}.db')
    # upload updated DB of newly processed files so they won't be processed again
    sendDBfromDF(resource, dbPath, bucketName, addedNewDF, dbName, f'./{dbName}.db')
    # generate GDF, spit out geojson / shpfile, and upload
    sendGJSHPfromDF(resource, metaGJSHPPath, bucketName, combinedMetadataDF, metaDBName)

    return

if __name__ == "__main__":
    s3_resource = boto3.resource('s3')
    bucketName = 'resources.qsisphere.com'
    logging.basicConfig(filename='botImageConvert.log', filemode='a', format='%(asctime)s - %(message)s', level=logging.DEBUG)

    main(bucketName, s3_resource)
