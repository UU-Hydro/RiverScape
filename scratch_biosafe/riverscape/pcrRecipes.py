#!/usr/bin/env python
# -*- coding: utf-8 -*-

#-modules
import os, sys, subprocess
import uuid
import time as tm
import numpy as np
import datetime
import glob
import pcraster as pcr
import pickle
import pandas as pd
import geopandas as gpd

from . import mapIO

############# numpy generic ############################################
def sortByNthColumn(arr, colIdx):
	'''sort a 2D array by the nth column'''
	return arr[arr[:,colIdx].argsort()]

def getMax(x,a):
	m= float(a.max())
	if x == None:
		return m
	else:
		return max(m,x)

def getMin(x,a):
	m= float(a.min())
	if x == None:
		return m
	else:
		return min(m,x)

def nashSutcliffe(Qobs,Qmod):
	'''compute the Nash-Sutcliffe model efficiency from observed and modelled data'''
	return 1 - ( np.sum((Qobs-Qmod)**2) ) / ( np.sum((Qobs - np.mean(Qobs))**2) )

def getFirstDay(startDate, deltaYears=0, deltaMonths=0):
	# d_years, d_months are "deltas" to apply to dt
    y, m = startDate.year + deltaYears, startDate.month + deltaMonths
    a, m = divmod(m-1, 12)
    return datetime.date(y+a, m+1, 1)

def getLastDay(dt):
    return getFirstDay(dt, 0, 1) + datetime.timedelta(-1)

def recursiveGlob(rootDir='.', suffix=''):
    ll = []
    for rootDir, dirNames, fileNames in os.walk(rootDir):
	    for fileName in fileNames:
		    if fileName.endswith(suffix):
			    ll.append(os.path.join(rootDir, fileName))
    return ll

def joinArraysNoKeep(array1, index1, array2, index2):
    """ Join two arrays based on common column index1 and index2
        removes the iteration of index column """
    nrColumns1 = np.shape(array1)[1]
    nrColumns2 = np.shape(array2)[1]
    outRows    = np.shape(array1)[0]
    outColumns = nrColumns1 + nrColumns2
    out = np.zeros((outRows, outColumns)) - 9999

    out[:,0:nrColumns1] = array1
    dict2 = array2dict(array2, index2)
    t=0
    for row in array1:
        key = row[index1]
        try:
            out[t,nrColumns1:] = dict2[key]
            t+=1
        except KeyError:
            t+=1
    delIndex = np.shape(array1)[1]+index2
    out = np.delete(out, delIndex, axis=1)
    return out

def array2dict(inArray, keyIndex):
    d = dict()
    for row in inArray:
        try:
            key = row[keyIndex]
            values = []
            for value in row:
                values.append(value)
            d[key] = values
        except KeyError:
            print('KeyError, passed')
    return d

def uniqueRows(a):
    """a = array of m rows n columns.
    """

    order = np.lexsort(a.T)
    a = a[order]
    diff = np.diff(a, axis=0)
    ui = np.ones(len(a), 'bool')
    ui[1:] = (diff != 0).any(axis=1)
    return a[ui]

########## PCRASTER ####################################################
def clone_attributes():
    """
    Get the map attributes from the PCRaster clone map.

    Returns a list with: xmin, xmax, ymin, ymax, nr_rows, nr_cols, cell_size.
    """
    nr_rows = pcr.clone().nrRows()
    nr_cols = pcr.clone().nrCols()
    cell_size = pcr.clone().cellSize()
    ymax = pcr.clone().north()
    xmin = pcr.clone().west()
    ymin = ymax - nr_rows * cell_size
    xmax = xmin + nr_cols * cell_size
    return xmin, xmax, ymin, ymax, nr_rows, nr_cols, cell_size

def map_edges(clone):
    """Boolean map true map edges, false elsewhere"""

    pcr.setglobaloption('unittrue')
    xmin, xmax, ymin, ymax, nr_rows, nr_cols, cell_size = clone_attributes()
    clone = pcr.ifthenelse(pcr.defined(clone), pcr.boolean(1), pcr.boolean(1))
    x_coor = pcr.xcoordinate(clone)
    y_coor = pcr.ycoordinate(clone)
    north = y_coor > (ymax - cell_size)
    south = y_coor < (ymin + cell_size)
    west = x_coor < (xmin + cell_size)
    east = x_coor > (xmax - cell_size)
    edges = north | south | west | east
    return edges

def representativePoint(nominalMap):
    """Select a representative point for a nominal map
    """
    pcr.setglobaloption('unitcell')
    filled = pcr.cover(nominalMap, 0)
    edges = pcr.windowdiversity(filled,3) > 1
    edges = pcr.ifthen(pcr.defined(nominalMap), edges)
    edges = map_edges(nominalMap) | edges
    dist = pcr.spread(edges, 0,1)
    dist = dist + pcr.uniform(pcr.defined(nominalMap))
    points = dist == pcr.areamaximum(dist, nominalMap)
    return pcr.ifthen(points, nominalMap)

def getCoordinates(cloneMap,MV= -9999):
	'''returns cell centre coordinates for a clone map as numpy array
	   return longitudes, latitudes '''
	cln = pcr.cover(pcr.boolean(cloneMap), pcr.boolean(1))
	xMap= pcr.xcoordinate(cln)
	yMap= pcr.ycoordinate(cln)
	return pcr.pcr2numpy(xMap,MV)[1,:], pcr.pcr2numpy(yMap,MV)[:,1]

def col2map(arr, cloneMapName, x=0, y=1, v=2, args = ''):
	''' creates a PCRaster map based on cloneMap
	x,y,v (value) are the indices of the columns in arr '''
	g = np.hstack((arr[:,x:x+1],arr[:,y:y+1],arr[:,v:v+1]))
	np.savetxt('temp.txt', g, delimiter=',')
	cmd = 'col2map --nothing --clone %s %s temp.txt temp.map'% (cloneMapName, args)
	print('\n', cmd)
	subprocess.call(cmd, shell=True)
	outMap = pcr.readmap('temp.map')
	os.remove('temp.txt')
	os.remove('temp.map')
	return outMap

def mapPercentile(scalarMap, percentile, area = False):
    """
    Compute the score at percentile for a scalar map.

    If area == True, the area is returned where scalarMap < score at percentile
    """

    arr = pcr.pcr2numpy(scalarMap, -9999).flatten()
    arr = arr[arr[:] > -9999]
    score = np.percentile(arr, percentile)
    if area == False:
        return score
    elif area == True:
        percentileArea = pcr.ifthen(scalarMap < score, scalarMap)
        return score, percentileArea

def pointPerClass(classMap):
    """ Select a single random point from each class in classMap"""
    rand1 = 100 * pcr.uniform(pcr.boolean(classMap))
    rand2 = 100 * pcr.uniform(pcr.boolean(classMap))
    rand3 = 100 * pcr.uniform(pcr.boolean(classMap))

    randomMap = pcr.scalar(classMap) * rand1 * rand2 * rand3
    pointMap  = pcr.ifthen(randomMap == pcr.areaminimum(randomMap, classMap), classMap)
    nrPointsPerClass = pcr.areatotal(pcr.scalar(pcr.boolean(pointMap)), classMap)
    assert pcr.cellvalue(pcr.mapmaximum(nrPointsPerClass), 0)[0] == 1
    return pointMap

def getCellValues(pointMap, mapList = [], columns = []):
    """ Get the cell values of the maps in mapList at the locations of pointMap
    """
    #-determine where the indices are True
    arr = pcr.pcr2numpy(pcr.boolean(pointMap),0).astype('bool')
    indices = np.where(arr == True)

    #-loop over the points in pointMap
    pcr.setglobaloption('unitcell')
    ll = []
    for rowIdx, colIdx in zip(indices[0], indices[1]):
        line = []
        line.append(rowIdx)
        line.append(colIdx)
        for pcrMap in mapList:
            line.append(pcr.cellvalue(pcrMap, np.int(rowIdx+1), np.int(colIdx +1))[0])
        ll.append(line)

    #-optionally add column names
    if len(columns) == len(mapList):
        columnNames = ['rowIdx', 'colIdx'] + columns
    else:
        columnNames = ['rowIdx', 'colIdx'] + \
                    ['map' + str(ii) for ii in range(1, 1 + len(mapList), 1)]

    #-return as Pandas DataFrame
    return pd.DataFrame(np.array(ll), columns = columnNames)

def reclassEcotopes(ecoMap, ecoDf, lookupDf, lookupColumn):
    """
    reclassify nominal values in an ecotope map based on the lookup column in
    the lookup data frame

    parameters:
        ecoMap: ecotope map with IDs in the legend
        ecoDf:  pandas data frame with 'ECO_CODE' en 'ID' columns
        lookupDf: pandas data frame with 'ECO_CODE' en lookupvalue columns
        lookupColumn: string with the name of the column to lookup
    """

    assert 'ECO_CODE' in ecoDf.columns.values
    assert 'ECO_CODE' in lookupDf.columns.values
    assert lookupColumn in lookupDf.columns.values

    #-determine the lookuptable for the ecotope IDs [ecoID, lookupValue]
    df = pd.merge(ecoDf, lookupDf, on = 'ECO_CODE')
    columnData = df.loc[:,['ID', lookupColumn]].values
    lutFile = 'lutReclass1%s.txt' % uuid.uuid4()
        # needs a unique file name due to bug in pcr.lookupnominal
    np.savetxt(lutFile, columnData, fmt = '%.0f')

    recoded = pcr.lookupnominal(lutFile, ecoMap)
    os.remove(lutFile)
    return recoded

def reclass_ecotopes(ecotopes, lookup_df, lookup_column):
    """
    reclassify nominal values in an ecotope map based on the lookup column in
    the lookup data frame

    parameters:
        ecotopes: LegendMap class pcr_map (ecotopes) and legend
        lookup_df: DataFrame with 'ECO_CODE' and lookupvalue columns
        lookupColumn: string with the name of the column to lookup
    """

    assert 'ECO_CODE' in ecotopes.legend_df.columns.values
    assert 'ECO_CODE' in lookup_df.columns.values
    assert lookup_column in lookup_df.columns.values

    #-determine the lookuptable for the ecotope IDs [ecoID, lookupValue]
    df = pd.merge(ecotopes.legend_df, lookup_df, on = 'ECO_CODE')
    column_data = df.loc[:,['values', lookup_column]].values
    lut_file = 'lut_reclass_%s.txt' % uuid.uuid4()
        # needs a unique file name due to bug in pcr.lookupnominal
    np.savetxt(lut_file, column_data, fmt = '%.0f')

    recoded = pcr.lookupnominal(lut_file, ecotopes.pcr_map)
    os.remove(lut_file)
    return recoded

########## IO ##########################################################
def writeCSVfile(outFileName, columnNames, dataArray):
	'''writes a data array to a file and adds a header'''
	assert len(columnNames) == dataArray.shape[1]
	outFile = open(outFileName, 'w')
	columnNames[-1] = columnNames[-1] + '\n'
	outFile.write(','.join(columnNames))
	for row in dataArray:
		outLine = ''
		for item in row:
			outLine  = outLine + ('%s,' % item)
		outLine = outLine[0:-1]
		outLine = outLine + '\n'
		outFile.write(outLine)
	outFile.close()

def makeDir(directoryName):
    try:
        os.makedirs(directoryName)
    except OSError:
        pass

def make_dir(directory_path):
    try:
        os.makedirs(directory_path)
    except OSError:
        pass



















