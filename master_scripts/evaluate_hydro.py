# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 10:26:17 2015

@author: Menno Straatsma
"""

import netCDF4
import math
import numpy as np
import pandas as pd
import string
import time
import datetime as dt
from collections import OrderedDict
import os
import shutil
import glob
import configparser

import matplotlib.pyplot as plt
import seaborn as sns

import geopandas as gpd

import pcraster as pcr
import pcrRecipes
import mapIO
from measures import *

def read_pli(pliFile):
    """Read a .pli file into a dictionary
    input 
    pliFile:    .pli, .pliz, or .pol input file for Delft3D-FM
    """
    def line2floats(line):
        return [float(i) for i in line.split()]
        
    with open(pliFile) as f:
        lines  = f.read().splitlines()
        
    #=loop over the polylines and store the data in a dictionary
    data = {}
    cnt = 0
    labels = []
    while cnt < len(lines):
        label = lines[cnt]
        if label in labels:
            label = label + str(cnt)
        nrow, ncol = lines[cnt+1].split()
        nrow, ncol = int(nrow), int(ncol)    
        polData = [line2floats(line) for line in lines[cnt+2: cnt+2+nrow]]
        cnt += nrow + 2
        data[label] = np.array(polData)
        labels.append(label)
    print len(data.keys()), 'polylines read in:\n\t %s' % (pliFile,)
    
    return data

def pliDict2GeoDataFrame(pliDict, outType = 'Point'):
    """Convert a dictionary with polyline data to a geopandas GeoDataFrame
    
    Input:
    pliDict :    dictionary with polyline data.
    outType :    can be 'Point', 'LineString', or 'Polygon' GeoDataFrame
    
    If 'Line' or 'Polygon' is selected, attribute data will be lost.
    """
    
    container = [] 
    for label, data in pliDict.iteritems():
        if outType == 'Point':
            points = gpd.GeoSeries([Point(x,y) for x,y in data[:,0:2]])
            geodata = gpd.GeoDataFrame(geometry = points)
            attributes = pd.DataFrame(data=data)
            gdf = pd.concat((geodata, attributes), axis=1)
            colNameList = ['geometry'] + list('abcdefghijklmnop')[0:data.shape[1]]
            gdf.columns = colNameList
            gdf['label'] = label
            
        elif outType == 'LineString':
            xyList = [[x,y] for x,y in data[:,0:2]]
            line = gpd.GeoSeries(LineString(xyList))
            gdf = gpd.GeoDataFrame(geometry = line)
            gdf['label'] = label
        
        elif outType == 'Polygon':
            try:
                xyList = [[x,y] for x,y in data[:,0:2]]
                line = gpd.GeoSeries(Polygon(xyList))
                gdf = gpd.GeoDataFrame(geometry = line)
                gdf['label'] = label
            except ValueError:
                pass

        container.append(gdf)
    
    outDf = pd.concat(container)
    outDf.index = np.arange(len(outDf))
    return gpd.GeoDataFrame(outDf)

def geoDataFrame2pli(geoDataFrame, pliFile, geometry = 'Point'):
    """Converts a geopandas.GeoDataFrame to a pli file on disk.
        
    Extension conventions are:
            .pli   for polyline files
            .pliz  for polyline files with addional attributes
            .pol   for polyline files with identical first and last coordinates
    
    input:
    geoDataFrame  :    geopandas GeoDataFrame
    pliFile       :    path to the output file
    geometry      :    shapely geometry (Point, LineString, Polygon)    
    
    Returns the file path.
    """
#    print 'Writing polyline data to:\n\t', pliFile
    with open(pliFile, 'w') as f:
        if geometry == 'Point':
            for label, df in geoDataFrame.groupby('label'):
                assert len(df.shape) == 2
                df = df.loc[:,['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']]
                f.write(label+'\n')
                f.write('%s %s\n' % df.shape)
                f.write(df.to_string(header=False, index=False))
                f.write('\n')
        elif geometry == 'LineString':
            pass
        elif geometry == 'Polygon':
            for index, pol, label in geoDataFrame.loc[:,['geometry', 'label']].itertuples():
                if pol.geom_type == 'MultiPolygon':
                    # TO DO: separate multipolygons and create new label
                    #        now the whole polygon is removed
                    pass
                else:
                    x,y = pol.exterior.coords.xy 
                    xyCoords = np.array([[xy[0], xy[1]] for xy in zip(x,y)])
                    f.write(label+'\n')
                    f.write('%s %s\n' % xyCoords.shape)
                    for xi, yi in xyCoords:
                        f.write('%.4f  %.4f\n' % (xi, yi))
    return pliFile

def updateTrachytopes(inFile, outFile, roughCodeMap):
    """
    input:
    inFile:       path to ascii trachytope file with columns x, y, 0, code, fraction
    outFile:      path to outfile with ascii trachytopes
    roughCodeMap: pcraster map with roughness codes
    
    output:
    new trachytope file
    """
    #-read the trachtope data into a DataFrame
    columns = ['x', 'y', 'z', 'code', 'frac']
    trachytopes = pd.DataFrame(data = np.loadtxt(inFile), columns = columns)
    nrCols = pcr.clone().nrCols()
    nrRows = pcr.clone().nrRows()
    #-get cell value for each line in trachytope DataFrame    
    for cnt in xrange(len(trachytopes)):
        xcoord = trachytopes.loc[cnt, 'x']
        ycoord = trachytopes.loc[cnt, 'y']
        row, col = pcr_coord(xcoord, ycoord)
        if col > nrCols:
            roughCode, validCell = 102, False
        elif row <= 0:
            roughCode, validCell = 102, False
        else:
            roughCode, validCell = pcr.cellvalue(roughCodeMap, row, col)
        if validCell == True:
            trachytopes.loc[cnt, 'code'] = roughCode
    
    groupedCodes = trachytopes.groupby(by = ['x', 'y', 'z', 'code']).sum()
    groupedCodes.reset_index(inplace = True)
    np.savetxt(outFile, groupedCodes.values, fmt = '%.4f %.4f %.0f %.0f %.4f')
    return groupedCodes

def updateNetNC(ncFile, dem):
    """Updates the Netnode_z variable in a fm '_net.nc' netCDF file 
    
    Input:
    ncFile : *_net.nc file that describes the netnodes of the computational grid
    dem    : pcraster object with new terrain heights at the location of the measures
    """
    
    ds = netCDF4.Dataset(ncFile, 'a')
    xcoords = ds.variables['NetNode_x'][:]
    ycoords = ds.variables['NetNode_y'][:]
    zcoords = ds.variables['NetNode_z'][:]
    
    #-update the z-values and sync with netCDF file on disk
    newZ = []
    nrCols = pcr.clone().nrCols()
    nrRows = pcr.clone().nrRows()
    for cnt in xrange(len(xcoords)):
        x,y,z = xcoords[cnt], ycoords[cnt], zcoords[cnt]
        row, col = pcr_coord(x, y)
        if col > nrCols:
            zMeasure, validCell = z, False
        elif row <= 0:
            zMeasure, validCell = z, False
        else:
            zMeasure, validCell = pcr.cellvalue(dem, row, col)            
        
        if validCell == True:
            newZ.append(zMeasure)
        else:
            newZ.append(z)    
    
    ds.variables['NetNode_z'][:] = np.array(newZ)
    ds.sync
    ds.close()
    del ds
    return newZ

def explode(geodataframe):
    """Explode a MultiPolygon geometry into individual Polygon geometries"""
    outdf = gpd.GeoDataFrame(columns=geodataframe.columns)
    for idx, row in geodataframe.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gpd.GeoDataFrame(columns=geodataframe.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf
               

def copy_fm_files(src_dir, dst_dir):
    """
    Copy the contents of the source directory to the destination directory.
    Creates the destination directory if it does not yet exist.
    """
    print "Copy files in {0} to {1}".format(src_dir, dst_dir)
    try:
        os.makedirs(dst_dir)
    except OSError:
        pass
    file_list = glob.glob(os.path.join(src_dir, '*.*'))
    for src_file in file_list:
        print '.',
        file_name = os.path.split(src_file)[-1]
        shutil.copyfile(src_file, os.path.join(dst_dir, file_name))
    print

def copy_fm_dirs(msr_names, ref_fm_dir, dst_root_dir):
    """
    """
    assert os.path.isdir(ref_fm_dir) == True
    assert os.path.isdir(dst_root_dir) == True
    for msr_name in msr_names:
        copy_fm_files(ref_fm_dir, os.path.join(dst_root_dir, msr_name))

def read_measure(msr_dir):
    """
    Read a measure (directory on disk) into a Measure.
    """
    settings = pd.read_csv(os.path.join(msr_dir, 'settings.txt'),
                           delimiter=':', index_col=0, header=None)
    area = pcr.readmap(os.path.join(msr_dir, 'area.map'))
    dem = pcr.readmap(os.path.join(msr_dir, 'dem.map'))
    ecotopes = read_map_with_legend(os.path.join(msr_dir, 'ecotopes.map'))
    trachytopes = pcr.readmap(os.path.join(msr_dir, 'trachytopes.map'))
    groyne_height = pcr.readmap(os.path.join(msr_dir, 'groyne_height.map'))
    minemb_height = pcr.readmap(os.path.join(msr_dir, 'minemb_height.map'))
    main_dike_height = pcr.readmap(os.path.join(msr_dir, 'main_dike_height.map'))
    msr = Measure(settings,
                 area, dem, ecotopes, trachytopes,
                 groyne_height, minemb_height, main_dike_height)
    return msr

def pcr_coord(xcoord, ycoord):
    """
    Get the row and column number for the clone map at xcoord, ycoord.
    Clone map should be defined explicitly, or through a previous command.
    """
    west = pcr.clone().west()
    north = pcr.clone().north()
    cell_size = pcr.clone().cellSize()
    xCol = (xcoord - west) / cell_size
    yRow = (north - ycoord) / cell_size
    col  = 1 + int(math.floor(xCol))
    row  = 1 + int(math.floor(yRow))
    return row, col 

def has_data(pcr_map):
    """
    Check if a PCRaster map has data or consists of missing values only.
    """
    MV = -1e20
    unique_values = np.unique(pcr.pcr2numpy(pcr.scalar(pcr_map), MV))
    unique_values = unique_values[unique_values > MV]
    has_data = len(unique_values) > 0
    return has_data

def update_trachytopes(fm_trachytopes, msr_trachytopes):
    """
    Update trachytopes in FM using the measure trachytopes. Each line in
    fm_trachytopes is checked whether the roughness has changed. Identical
    lines are merged into one.
    ----------
    Parameters:
        FM_trachytopes: DataFrame with [x, y, 0, code, fraction] - columns
        msr_trachytopes: PCRaster object with new trachytope distributions
    
    Returns a DataFrame with updated trachytope definitions
    """

    # Get cell value of msr_trachytopes for each line in trachytope DataFrame    
    updated_trachytopes = update_z(fm_trachytopes.loc[:,['x', 'y', 'code']],
                                   msr_trachytopes)
    df = fm_trachytopes.copy()
    df.loc[:,'code'] = updated_trachytopes.z
    grouped_codes = df.groupby(by = ['x', 'y', 'z', 'code']).sum()
    grouped_codes.reset_index(inplace = True)
    return grouped_codes

def update_z(df, pcr_map):
    """
    Update the 'z' column of the DataFrame with the 'x' and 'y' columns.
    Missing values in pcr_map are left unaltered
    ----------    
    Parameters:
        df: DataFrame with 'x', 'y', and 'z' columns
        pcr_map: PCRaster map with new values for 'z'. 
    """
    columns = list('xyz')
    df.columns = columns
    df_out = df.copy()    
    pcr_map = pcr.scalar(pcr_map)
    
    for idx, row in df.loc[:, columns].iterrows():
        x, y, z = list(row)
        row_nr, col_nr = pcr_coord(x,y)
        try:
            new_z, validity = pcr.cellvalue(pcr_map, row_nr, col_nr)
            if validity == True:
                df_out.loc[idx, 'z'] = new_z
            else:
                df_out.loc[idx, 'z'] = z
        except (ValueError, OverflowError):
            df_out.loc[idx, 'z'] = z
    return df_out

def lower_fxw(fixed_weirs, ref_height):
    """
    Lower fixed weirs using a PCRaster map.
    -----------
    Parameters:
        fixed_weirs: Fixed weir DataFrame with Point geometry
        ref_height: PCRaster map with new heights.
    
    Returns a fixed weir DataFrame
    """
    
    ref_z = update_z(fixed_weirs.loc[:,list('abc')], ref_height)
    fixed_weirs.loc[:,'ref_z'] = ref_z.z.values
    new_z = fixed_weirs.apply(lambda x: max([min([x.c, x.ref_z]), x.d, x.e]),
                                        axis=1)
    out_df = fixed_weirs.copy()
    out_df.loc[:, 'c'] = new_z
    return out_df.drop('ref_z', axis=1)

def lower_fxw_relative(fixed_weirs, ref_height):
    """
    Lower fixed weirs using a PCRaster map.
    -----------
    Parameters:
        fixed_weirs: Fixed weir DataFrame with Point geometry
        ref_height: PCRaster map with new heights.
    
    Returns a fixed weir DataFrame
    """
    fxw_abs = fixed_weirs.copy() # fxw GeoDataFrame with absolute toe heights
    fxw_abs['d_abs'] = fxw_abs.c - fxw_abs.d
    fxw_abs['e_abs'] = fxw_abs.c - fxw_abs.e
    ref_z = update_z(fxw_abs.loc[:,list('abc')], ref_height)
    fxw_abs['ref_z'] = ref_z.z
    fxw_abs['new_z'] = fxw_abs.apply(lambda x: max([min([x.c, x.ref_z]), 
                                                   x.d_abs, x.e_abs]),
                                     axis=1)
    out_fxw = fixed_weirs.copy()
    out_fxw.c = fxw_abs.new_z
    out_fxw.d = fxw_abs.new_z - fxw_abs.d_abs
    out_fxw.e = fxw_abs.new_z - fxw_abs.e_abs
    return out_fxw

def remove_fxw(fixed_weirs, removal_area):
    """
    Erase fixed weirs within the removal area. Line elements that get split 
    by the removal are relabeled.
    -------------
    Parameters
        fixed_weirs: Fixed weir DataFrame with Point geometry
        removal_area: Scalar map area, removal area indicated with -9999.
    
    Returns a fixed weir DataFrame.
    """
    # add label details as columns
    labels = fixed_weirs.label.str.split(':', expand = True)
    labels.columns = ['label_nr', 'label_type']
    fixed_weirs = pd.concat([fixed_weirs, labels], axis=1)
    
    # add reference value: 'selected' = 0:inside, 1:outside
    label_map = pcr.ifthenelse(pcr.defined(removal_area), pcr.scalar(0),
                               pcr.scalar(1))
    labeled_points = update_z(fixed_weirs.loc[:,list('abc')], label_map)
    fixed_weirs.loc[:,'selected'] = labeled_points.z 
    # 'parts' increases with 1 at every change from 1 to 0 and 0 to 1 in 'selected'
    fixed_weirs['parts'] =  (1 + fixed_weirs.selected.diff().abs().cumsum())
    fixed_weirs.fillna(0, inplace=True)

    # take the minimum value of 'parts' for each fixed weir and subtract
    # from the cumulative value to reset the counter per fixed weir
    nr_parts_cum = fixed_weirs.groupby(by = 'label').min().parts
    nr_parts_cum = nr_parts_cum.fillna(0).reset_index()
    
    fxw2 = pd.merge(fixed_weirs, nr_parts_cum,
                    on = 'label', suffixes = ['_l', '_r'])
    fxw2['parts'] = (1 + fxw2.parts_l - fxw2.parts_r)
    fxw2['new_label'] = fxw2.apply(lambda x: '{0}_{1}:{2}'.format(x.label_nr,
                                                           str(int(x.parts)),\
                                                           str(x.label_type)),\
                                                           axis=1)
        
    # clean up
    points_per_part = fxw2.loc[:,['new_label', 'selected']]\
                           .groupby('new_label')\
                           .count().reset_index()
    points_per_part.columns = ['new_label', 'nr_points']
    fxw3 = pd.merge(fxw2, points_per_part, on = 'new_label',
                    suffixes = ['_l', '_r'])
    fxw_out = fxw3[(fxw3.nr_points > 1) & (fxw3.selected == 1)]
    fxw_out.drop(['label', 'label_nr', 'label_type', 'selected',\
                  'parts_l', 'parts_r', 'parts', 'nr_points'],\
                   axis = 1, inplace=True)
    fxw_out.rename(columns = {'new_label':'label'}, inplace = True)
    fxw_out.index = range(len(fxw_out)) # to overwrite duplicate indices
    return fxw_out

def remove_thd(thin_dams, removal_area):
    """
    Remove the thin dams within the removal area.
    ----------
    Parameters
        thin_dams: Polygon GeoDataFrame with D-Flow FM thin dams
        removal_area: Boolean PCRaster map with area to be removed.
    Raster is converted to a polygon, and the difference with the thin dams
    is determined.
    """
    removal_area = pcr.nominal(removal_area+10000)
    pcr.report(removal_area, 'rem_area.map')
    mapIO.pcr2Shapefile('rem_area.map', 'mask.shp', 'areas')
    pols = gpd.GeoDataFrame.from_file('mask.shp')
    pols = pols[pols.areas > 0]
    pols = explode(gpd.GeoDataFrame(geometry=pols.buffer(0))) 
    # remove self-intersections from polygonized rasters
    
#    print 'Removing thin dams...'
    # select using polygon. This is the neat way of cutting out pieces of 
    # the thin dam polygons.
    
    thd_rem = thin_dams['geometry'].difference(pols.geometry.unary_union)
    thd_rem = gpd.GeoDataFrame(thd_rem[thd_rem.area > 2].reset_index())
    thd_rem.rename(columns = {0 : 'geometry'}, inplace=True)
    
    out_gdf = thd_rem.loc[:,['geometry', 'label']]
    out_gdf = explode(out_gdf)
    out_gdf['label'] = ['%s_%s'%(ii, ii+1) for ii in range(1,len(out_gdf)+1,1)]
    return out_gdf  
    
def update_FM(fm, msr, winter_bed, debug=False):
    """
    Update the a FM object with a measure.
    ---------
    Parameters:
        fm: FM object with attributes to be updated
        msr: Measure object with settings DataFrame and seven PCRaster maps
    
    Returns a FM object with altered attributes.
    """
    #-update trachytopes
    if has_data(msr.trachytopes) == True:
        trachytopes = update_trachytopes(fm.trachytopes, msr.trachytopes)
    else:
        trachytopes = fm.trachytopes
    
    #-update bathymetry
    if has_data(msr.dem) == True:
        bathymetry = update_z(fm.bathymetry, msr.dem)
    else:
        bathymetry = fm.bathymetry
    
    #-update fixed weirs
    labels = fm.fxw_points.label.str.split(':', expand=True)
    labels.columns = ['ID', 'type']
    groynes = fm.fxw_points[labels.type == 'type=1']
    minembs = fm.fxw_points[labels.type == 'type=2']
    hdiff = fm.fxw_points[labels.type == 'type=3']
    
    # groyne lowering
    lowering_height = pcr.ifthen(msr.groyne_height > -9999, msr.groyne_height)
    if has_data(lowering_height) == True:
        groynes_low = lower_fxw_relative(groynes, lowering_height)
    else:
        groynes_low = groynes
    
    # groyne removal
    removal_area = pcr.ifthen(msr.groyne_height == -9999, msr.groyne_height)
    if has_data(removal_area) == True:
        groynes_rem = lower_fxw_relative(groynes_low, removal_area)
    else:
        groynes_rem = groynes_low
    
    # minemb lowering
    lowering_height = pcr.ifthen(msr.minemb_height > -9999, msr.minemb_height)
    if has_data(lowering_height) == True:
        minembs_low = lower_fxw_relative(minembs, lowering_height)
    else:
        minembs_low = minembs
       
    # minemb removal
    removal_area = pcr.ifthen(msr.minemb_height == -9999, msr.minemb_height)
    if has_data(removal_area) == True:
        minembs_rem = remove_fxw(minembs_low, removal_area)
    else:
        minembs_rem = minembs_low   
    
    # hdiff removal
    removal_area = pcr.ifthen(pcr.defined(msr.dem), pcr.scalar(-9999))
    if has_data(removal_area) == True:
        hdiff_rem = remove_fxw(hdiff, removal_area)
    else:
        hdiff_rem = hdiff

    fxw_out = pd.concat([groynes_rem, minembs_rem, hdiff_rem], axis=0)

    # erase thin dams
    removal_area = pcr.ifthen(pcr.defined(msr.dem), pcr.scalar(-9999))
    if has_data(removal_area) == True:
        thd_rem = remove_thd(fm.thd_pols, removal_area)
    else:
        thd_rem = fm.thd_pols
    
    # erase dry areas, but only for relocation measures
    embanked_area = ~ winter_bed
    new_winbed = pcr.ifthen(embanked_area & msr.area, pcr.scalar(-9999))
    if (has_data(new_winbed) == True) and (msr.settings.loc['msr_type',1] == 'relocation'):
        dry_areas_rem = remove_thd(fm.dry_areas, new_winbed)
    else:
        dry_areas_rem = fm.dry_areas
    
    fm_updated = FM(fxw_out, fm.fxw_lines, thd_rem, dry_areas_rem,
                    trachytopes, bathymetry)
    return fm_updated

def read_FM(fm_dir, mdu_file):
    """
    Read the flexible mesh files that need updating into a class.
    fixed weirs: .pliz
    thin dams: .pli
    trachytopes: .arl
    dry areas: .pol
    bathymetry: .nc
    """
    #-read mdu file
    mdu = configparser.ConfigParser(allow_no_value=True,
                                    inline_comment_prefixes=('#'))
    mdu.read_file(open(os.path.join(fm_dir, mdu_file)))
    fxw_file = os.path.join(fm_dir, mdu['geometry']['FixedWeirFile'])
    thd_file = os.path.join(fm_dir, mdu['geometry']['ThinDamFile'])
    arl_file = os.path.join(fm_dir, mdu['trachytopes']['Trtl'])
    dry_areas_file = os.path.join(fm_dir, mdu['geometry']['DryPointsFile'])
    bathymetry_file = os.path.join(fm_dir, mdu['geometry']['NetFile'])
    
    fxw_points = pliDict2GeoDataFrame(read_pli(fxw_file), outType = 'Point')
    fxw_lines = pliDict2GeoDataFrame(read_pli(fxw_file), outType = 'LineString')
    thd_pols = pliDict2GeoDataFrame(read_pli(thd_file), outType = 'Polygon')
    dry_areas = pliDict2GeoDataFrame(read_pli(dry_areas_file),
                                     outType = 'Polygon')
    trachytopes = pd.DataFrame(data = np.loadtxt(arl_file),
                               columns = ['x', 'y', 'z', 'code', 'frac'])
    # bathymetry
    ds = netCDF4.Dataset(bathymetry_file, 'r')
    bathymetry = pd.DataFrame(data = {'x': ds.variables['NetNode_x'][:],
                                      'y': ds.variables['NetNode_y'][:],
                                      'z': ds.variables['NetNode_z'][:]})
    fm = FM(fxw_points, fxw_lines, thd_pols, dry_areas,
                    trachytopes, bathymetry)
    return fm

def write_FM(fm_out, fm_msr_dir, mdu_file):
    """
    Write FM files to disk
    --------
    Parameters:
        fm: FM object with updated attributues
        fm_msr_dir: path to the directory with the FM files
        mdu_file: file name of the Model Definition Unstructured_data file
    """
    # Read file names from the mdu-file.
    mdu = configparser.ConfigParser(allow_no_value=True,
                                    inline_comment_prefixes=('#'))
    mdu.read_file(open(os.path.join(fm_msr_dir, mdu_file)))
    arl_file = os.path.join(fm_msr_dir, mdu['trachytopes']['Trtl'])
    bathymetry_file = os.path.join(fm_msr_dir, mdu['geometry']['NetFile'])
    dry_areas_file = os.path.join(fm_msr_dir, mdu['geometry']['DryPointsFile'])
    fxw_file = os.path.join(fm_msr_dir, mdu['geometry']['FixedWeirFile'])
    thd_file = os.path.join(fm_msr_dir, mdu['geometry']['ThinDamFile'])
    
    # bathymetry
    ds = netCDF4.Dataset(bathymetry_file, 'a')
    ds.variables['NetNode_z'][:] = fm_out.bathymetry.z.values
    ds.sync; ds.close()
    del ds
    
    # dry areas
    geoDataFrame2pli(fm_out.dry_areas, dry_areas_file, geometry='Polygon')
    
    # fixed weirs
    geoDataFrame2pli(fm_out.fxw_points, fxw_file, geometry='Point')

    # thin dams
    geoDataFrame2pli(fm_out.thd_pols, thd_file, geometry='Polygon')

    # trachytopes    
    arl_data = fm_out.trachytopes.values
    np.savetxt(arl_file, arl_data, fmt = '%.4f %.4f %.0f %.0f %.4f')

def batch_FM(msr_fm_root, msr_names, nr_cores=7):
    """
    Create a batch file for single core FM-runs per measure.
    """
    # Remove old batch files
    for f in glob.glob(os.path.join(msr_fm_root,'runFM_*.bat')):
        os.remove(f)
    fmExe = r'C:\dflowfm-cli-x64-1.1.198.48568\dflowfm-cli.exe'
    nr_runs_per_file = len(msr_names) / nr_cores
    print nr_runs_per_file
    fm_bats = set()
    for ii, msr_name in enumerate(msr_names): 
        batch_nr = 1 + ii % nr_cores
        bat_file = os.path.join(msr_fm_root, 'runFM_%.0d.bat' % batch_nr)
        fm_bat = file(bat_file, 'a')
        
        #-compile the batch file
        msr_fm_dir = os.path.join(msr_fm_root, msr_name)
        fm_bat.write('cd %s\n' % msr_fm_dir)
        fm_bat.write("%s --autostartstop -t 1 msr.mdu\n" % fmExe)
        fm_bat.close()
        fm_bats.add(bat_file)

    all_file = os.path.join(msr_fm_root, 'start_all.bat')
    with open(all_file, 'w') as start_bat:
        for fm_bat in fm_bats:
            start_bat.write('start /B %s\n'% fm_bat)

  
class FM(object):
    """
    Read all D-Flow FM reference files
    Import in a measure
    update required files
    write/copy output files to fm_dst_dir.
    """
    def __init__(self, fxw_points, fxw_lines, thd_pols, dry_areas,
                    trachytopes, bathymetry):
        self.fxw_points = fxw_points
        self.fxw_lines = fxw_lines
        self.thd_pols = thd_pols
        self.dry_areas = dry_areas
        self.trachytopes = trachytopes
        self.bathymetry = bathymetry

def pliz2shp(inPlizFile, outShpFile, out_type = 'Point'):
    """Convert a Delft3D-FM polygon file with z coordinates (.pliz) to a ESRI
    shapefile with separate columns for label type
    """
    if out_type == 'Point':
        points = pliDict2GeoDataFrame(read_pli(inPlizFile), outType = 'Point')
        labels = points.label.str.split(':', expand = True)
        labels.columns = ['labelNr', 'labelType']
        labeledPoints = gpd.GeoDataFrame(pd.concat((points, labels), axis=1))       
        labeledPoints.to_file(outShpFile)
    elif out_type == 'LineString':
        lines = pliDict2GeoDataFrame(read_pli(inPlizFile), outType = 'LineString')
        labels = lines.label.str.split(':', expand = True)
        labels.columns = ['labelNr', 'labelType']
        labeledLines = gpd.GeoDataFrame(pd.concat((lines, labels), axis=1))       
        labeledLines.to_file(outShpFile)

def pli2shp(inPliFile, outShpFile, out_type = 'LineString'):
    """Convert a Delft3D-FM polygon file without z coordinates (.pli) to a ESRI
    shapefile with separate columns for label type
    """
    if out_type == 'LineString':
        lines = pliDict2GeoDataFrame(read_pli(inPliFile), outType = 'LineString')
        lines.to_file(outShpFile)    
    elif out_type == 'Polygon':
        polys = pliDict2GeoDataFrame(read_pli(inPliFile), outType = 'Polygon')
        polys.to_file(outShpFile)
    elif out_type == 'Point':
        points = pliDict2GeoDataFrame(read_pli(inPliFile), outType = 'Point')
        points.to_file(outShpFile)

def arl2shp(inArlFile, outShpFile):
    """Convert a Delft3D-FM area roughness file (.arl) to an ESRI
    shapefile with columns, x,y,z,code,fraction
    """
    df = pd.read_csv(inArlFile, sep = ' ', header = None)
    df.columns = ['x','y','z','code','fraction']
    df['geometry'] = df.apply(lambda row: Point(row.x, row.y), axis=1)
    gdf = gpd.GeoDataFrame(df)
    gdf.to_file(outShpFile)

def FM2GIS(fm_msr_dir, mdu_file, clone_file):
    """
    Convert update FM files to a GIS compatible format. Files are stored in
    a new GIS folder within the FM measure directory.
    --------
    Parameters:
        fm_msr_dir: path to the directory with the FM files
        mdu_file: file name of the Model Definition Unstructured_data file
    """
    GIS_dir = os.path.join(fm_msr_dir, 'GIS')
    pcrRecipes.make_dir(GIS_dir)
    # Read file names from the mdu-file.
    mdu = configparser.ConfigParser(allow_no_value=True,
                                    inline_comment_prefixes=('#'))
    mdu.read_file(open(os.path.join(fm_msr_dir, mdu_file)))
    arl_file = os.path.join(fm_msr_dir, mdu['trachytopes']['Trtl'])
    bathymetry_file = os.path.join(fm_msr_dir, mdu['geometry']['NetFile'])
    dry_areas_file = os.path.join(fm_msr_dir, mdu['geometry']['DryPointsFile'])
    fxw_file = os.path.join(fm_msr_dir, mdu['geometry']['FixedWeirFile'])
    thd_file = os.path.join(fm_msr_dir, mdu['geometry']['ThinDamFile'])
    
    # bathymetry
    ds = netCDF4.Dataset(bathymetry_file, 'r')
    x = ds.variables['NetNode_x'][:]
    y = ds.variables['NetNode_y'][:]
    z = ds.variables['NetNode_z'][:]
    del ds
    xyz = np.hstack([x.reshape(len(x),1), y.reshape(len(y),1),
                     z.reshape(len(z),1)])
    bathy = mapIO.col2map(xyz, clone_file, args=' -S ')
    pcr.report(bathy, os.path.join(GIS_dir, 'bathy.map'))
    
    # dry areas
    pli2shp(dry_areas_file, os.path.join(GIS_dir, 'dry_areas.shp'))
    
    # fixed weirs    
    pliz2shp(fxw_file, os.path.join(GIS_dir, 'fxw_points.shp'),
             out_type='Point')
    pliz2shp(fxw_file, os.path.join(GIS_dir, 'fxw_lines.shp'),
             out_type='LineString')

    # thin dams
    pli2shp(thd_file, os.path.join(GIS_dir, 'thd.shp'))

    # trachytopes    
    arl2shp(arl_file, os.path.join(GIS_dir, 'trachytopes.shp'))

def read_his_nc(netCDF_file, station_vars=[], xsec_vars=[]):
    """
    Read a Delft3D-FM history netcdf files for all stations and cross sections.
    ---------
    Parameters
        netCDF_file  :    path to the netCDF file on disk
        station_vars :    variable names of the stations 
        xsec_vars    :    variable names of the cross sections
    
    Returns a tuple of two lists with pandas data frames with time series
    """
    ds = netCDF4.Dataset(netCDF_file)
    #-extract time specs
    times = ds.variables['time']
    date_times = netCDF4.num2date(times[:], times.units)
    
    #-extract the names of the cross sections and observation points
    stat_names = [string.join(list(stat_name), '') \
                  for stat_name in ds.variables['station_name'][:]]
    xsec_names = [string.join(list(xsec_name), '') \
                  for xsec_name in ds.variables['cross_section_name'][:]]
    
    #-extract time series
    stat_list, xsec_list = [], []
    for vname in station_vars:
        stat_data = pd.DataFrame(data = ds.variables[vname][:],
                                columns = stat_names,
                                index=date_times)
        stat_list.append(stat_data)
    
    for vname in xsec_vars:
        xsec_data = pd.DataFrame(data = ds.variables[vname][:],
                                columns = xsec_names,
                                index=date_times)
        xsec_list.append(xsec_data)
    ds.close()
    del ds
    return stat_list, xsec_list

def read_wl_ensemble(msr_fm_root, msr_names, var_names=[], xsec_names=[], 
                     time_step=-1):
    """
    Read the water levels at the rivers axis for an ensemble of measures. 
    
    input
    ========
    msr_fm_root:  path to the ensemble folders
    msr_names:    string, folder name of the individual run
    var_names:    list, names of the fm-output variables, e.g. 'waterlevel'
    xsec_names:   list, names of the cross sections
    time_step:    datetime object, defaults to the last time step.
    """
    
    data = {}
    for msr_name in msr_names:
        f = os.path.join(msr_fm_root, msr_name, 'DFM_OUTPUT_msr/msr_his.nc.')
        if os.path.isfile(f)==True:
            his_all = read_his_nc(f, station_vars=['waterlevel'])
            if time_step == -1:
                wl = his_all[0][0].iloc[time_step,:]
            else:
                wl = his_all[0][0].loc[time_step,:]
            data[msr_name] = wl
        else:
            pass
    return pd.DataFrame(data=data)

#%% initial settings
# Waal XL
ref_fm_dir = r'D:\Projecten\RiverScapeWaal2\input\reference_FM_dQdH'
ref_map_dir = r'D:\Projecten\RiverScapeWaal2\input\reference_maps'
msr_fm_root = r'D:\Projecten\RiverScapeWaal2\output\waal_XL\measures_ensemble04\hydro'
msr_map_dir = r'D:\Projecten\RiverScapeWaal2\output\waal_XL\measures_ensemble04\maps'
pcr.setclone(os.path.join(ref_map_dir, 'clone.map'))
scratch_dir = r'D:\Projecten\RiverScapeWaal2\scratch'
os.chdir(scratch_dir)

msr_names = next(os.walk(msr_map_dir))[1]
print msr_names
ppp
#%% update fm based on directory with measures
start_time = time.time()
batch_FM(msr_fm_root, msr_names, 6)
copy_fm_dirs(msr_names, ref_fm_dir, msr_fm_root)
fm = read_FM(ref_fm_dir, 'ref.mdu')

winbed_file = os.path.join(ref_map_dir, 'winter_bed.map')
winter_bed = pcr.readmap(winbed_file)
for msr_name in msr_names[:]:
    print msr_name
    msr = read_measure(os.path.join(msr_map_dir, msr_name))
    fm_new = update_FM(fm, msr, winter_bed)
    fm_msr_dir = os.path.join(msr_fm_root, msr_name)
    write_FM(fm_new, fm_msr_dir, 'msr.mdu')
#    FM2GIS(os.path.join(msr_fm_root, msr_name), 'msr.mdu', winbed_file)
end_time = time.time()
print 'Processing duration in seconds: ', end_time - start_time
ppp
#%% extract water level lowering at the river axis for all measures
def rkm_to_float(in_arr):
    return np.array([float(rkm.split('_')[0]) for rkm in in_arr])

#-select three time steps, for Qref, dQ, and dQdh
t_Qref = dt.datetime(2016,1,3,0,0,0)
t_dQ = dt.datetime(2016,1,6,0,0,0)
t_dQdh = dt.datetime(2016,1,9,0,0,0)

#-extract water levels at the rivrer's axis for three scenarios
msr_names = next(os.walk(msr_fm_root))[1]
wl_Qref = read_wl_ensemble(msr_fm_root, msr_names, var_names=['waterlevel'],
                          time_step=t_Qref)
wl_dQ = read_wl_ensemble(msr_fm_root, msr_names, var_names=['waterlevel'],
                          time_step=t_dQ)
wl_dQdh = read_wl_ensemble(msr_fm_root, msr_names, var_names=['waterlevel'],
                          time_step=t_dQdh)

#-calculate the difference in water level with Qref without any measures.
wl_ref = wl_Qref['reference_FM_dQdh']
dwl_Qref = wl_Qref.subtract(wl_ref, axis=0).iloc[29:,:]
dwl_Qref.index = rkm_to_float(dwl_Qref.index.values)
dwl_Qref.to_csv('dwl_Qref.csv')

dwl_dQ = wl_dQ.subtract(wl_ref, axis=0).iloc[29:,:]
dwl_dQ.index = rkm_to_float(dwl_dQ.index.values)
dwl_dQ.to_csv('dwl_dQ.csv')

dwl_dQdh = wl_dQdh.subtract(wl_ref, axis=0).iloc[29:,:]
dwl_dQdh.index = rkm_to_float(dwl_dQdh.index.values)
dwl_dQdh.to_csv('dwl_Qref.csv')

wl_Qref = wl_Qref.iloc[29:,:]
wl_Qref.index = rkm_to_float(wl_Qref.index.values)
wl_Qref.to_csv('wl_Qref.csv')

wl_dQ = wl_dQ.iloc[29:,:]
wl_dQ.index = rkm_to_float(wl_dQ.index.values)
wl_dQ.to_csv('wl_dQ.csv')

wl_dQdh = wl_dQdh.iloc[29:,:]
wl_dQdh.index = rkm_to_float(wl_dQdh.index.values)
wl_dQdh.to_csv('wl_dQdh.csv')


#%% Plot water levels at the river axis
sns.set_style('ticks')
colors = ["pale red", "amber", "windows blue", "faded green", "dusty purple"]
sns.set_palette(sns.xkcd_palette(colors))
fig, [ax1, ax2] = plt.subplots(1,2, figsize=(8, 3.5))
wl_dQdh.reference_FM_dQdh.plot(ax=ax1, linewidth=3, label='Q18_dh1.8', zorder=100)
wl_dQ.reference_FM_dQdh.plot(ax=ax1, linewidth=3, label='Q18_dh0.0')
wl_Qref.reference_FM_dQdh.plot(ax=ax1, linewidth=3, label='Q16_dh0.0')


x = wl_Qref.index.values
ax1.fill_between(x=x, y1=wl_dQdh.min(axis=1), y2=wl_dQdh.max(axis=1),
                alpha=0.5, color=sns.xkcd_palette(['pale red']),
                label='Q18 dh1.8 with measures')
ax1.fill_between(x=x, y1=wl_dQ.min(axis=1), y2=wl_dQ.max(axis=1),
                alpha=0.5, color=sns.xkcd_palette(['amber']),
                label='Q18_dh0.0 with measures')
ax1.fill_between(x=x, y1=wl_Qref.min(axis=1), y2=wl_Qref.max(axis=1),
                alpha=0.5, color=sns.xkcd_palette(['windows blue']),
                label='Q16_dh0.0 with measures')

ax1.set_ylabel('water level (m + ordnance datum)')
ax1.set_xlabel('river kilometer (km)')

dwl_dQdh.reference_FM_dQdh.plot(ax=ax2, linewidth=3, label='Q18_dh1.8', zorder=100)
dwl_dQ.reference_FM_dQdh.plot(ax=ax2, linewidth=3, label='Q18_dh0.0')
dwl_Qref.reference_FM_dQdh.plot(ax=ax2, linewidth=3, label='Q16_dh0.0')


x = dwl_Qref.index.values
ax2.fill_between(x=x, y1=dwl_dQdh.min(axis=1), y2=dwl_dQdh.max(axis=1),
                alpha=0.25, color=sns.xkcd_palette(['pale red']),
                label='Q18 dh1.8 with measures')
ax2.fill_between(x=x, y1=dwl_dQ.min(axis=1), y2=dwl_dQ.max(axis=1),
                alpha=0.25, color=sns.xkcd_palette(['amber']),
                label='Q18 with measures')
ax2.fill_between(x=x, y1=dwl_Qref.min(axis=1), y2=dwl_Qref.max(axis=1),
                alpha=0.25, color=sns.xkcd_palette(['windows blue']),
                label='Q16 with measures')

ax2.set_ylabel('Water level relative to Q16_dh0.0 (m)')
ax2.set_xlabel('river kilometer (km)')

dwl_Qref.plot(ax=ax2, color='#3778bf', linewidth=0.5, alpha=0.5, legend=False)
dwl_dQ.plot(ax=ax2, color='#feb308', linewidth=0.5, alpha=0.5, legend=False)
dwl_dQdh.plot(ax=ax2, color='#d9544d', linewidth=0.5, alpha=0.5, legend=False)
#ax2.legend_.remove()

ax1.legend(bbox_to_anchor=(.75, 0.5))
ax1.text(850, -0.5, '(a)', fontsize=11)
ax2.text(847, -1.95, '(b)', fontsize=11)

plt.ylim(-1.4, 2.15)
plt.subplots_adjust(left=0.08, right=0.98, bottom=0.155, top=0.97, wspace=0.35)
plt.savefig('delta_water_levels.png', dpi=300)

#%% Plot water level lowering per scenario and measure type
#-prepare a single dataframe for plotting
def dwl_prep(dwl_df, hydro_label):
    df = dwl_df.copy()
    cols = ['dikeraising_evr_smooth', 'dikeraising_lrg_smooth', 
            'reference_FM_dQdh']
    df.drop(cols, axis=1, inplace=True)
    df.columns = [hydro_label + '_' + col for col in df.columns]
    df = df.unstack().to_frame(name='dwl')
    df['run'] = df.index.get_level_values(0)
    labels=df.run.str.split('_', expand=True)
    labels.columns = ['hydro', 'msr', 'area', 'eco']
    df_all = pd.concat([df, labels], axis=1)
    df_all.drop('run', axis=1, inplace=True)
    df_all.index = df_all.index.droplevel(level=0)
    df_all['rkm'] = df_all.index.values
    return df_all

dwl_tidy = pd.concat([dwl_prep(dwl_Qref, 'Qref'),
                      dwl_prep(dwl_dQ, 'dQ'),
                      dwl_prep(dwl_dQdh, 'dQdh')],
                      )
dwl_tidy.reset_index(drop=True, inplace=True)
dwl_tidy['area_eco'] = dwl_tidy.area + '_' + dwl_tidy.eco
dwl_tidy.area_eco.replace({'lrg_natural': 'LS_natural',
                           'lrg_smooth': 'LS_smooth',
                           'evr_natural': 'AS_natural',
                           'evr_smooth': 'AS_smooth'},
                           inplace=True)

dwl_tidy.rename(columns={'dwl':'dh (m)'}, inplace=True)
#-plot the data
colors = [ 'blue', "red", 'flat blue', "amber"]
sns.set_palette(sns.xkcd_palette(colors))
#colors = ['#b2df8a', '#a6cee3','#33a02c','#1f78b4']
sns.set_palette('Set1', 4)
g = sns.FacetGrid(dwl_tidy, 
                  col='hydro',
                  col_order = ['Qref', 'dQ', 'dQdh'],
                  row='msr',
                  row_order = ['groynelowering', 'minemblowering', 'smoothing',
                               'sidechannel', 'lowering'],
                  hue='area_eco',
                  hue_order=['LS_natural', 'LS_smooth', 
                             'AS_natural', 'AS_smooth'],
                  margin_titles=True,
                  size=2,
                  legend_out=False)
g = g.map(plt.plot, 'rkm', 'dh (m)')

g.set(ylim=(-1.4, 1.7), xlim=(865, 962))
g.fig.subplots_adjust(wspace=.05, hspace=.05)

for r in range(5):
    for c in range(3):
        g.axes[r,c].axhline(0, c='k', linewidth=1, zorder=0)
        g.axes[r,c].axvline(932, c='k', linewidth=10, zorder=1, alpha=0.1)
        g.axes[r,c].axvline(881, c='k', linewidth=10, zorder=2, alpha=0.1)
        if c == 1:
            g.axes[r,c].plot(dwl_dQ.index.values, dwl_dQ.reference_FM_dQdh.values,
                  c=(.7, .7, .7), zorder=0)
        if c == 2:
            g.axes[r,c].plot(dwl_dQdh.index.values, dwl_dQdh.reference_FM_dQdh.values,
                  c=(.7, .7, .7), zorder=0)
g.add_legend(title='')
plt.savefig('faceted_hydro.png', dpi=200)



#%% Extract key statistics on water level changes
# water level changes per reach
reaches = ['upp'] * 26 + ['mid'] * 26 + ['low'] * 26 + ['bc'] * 16
ll = []
for dwl, label in zip([dwl_Qref, dwl_dQ, dwl_dQdh],
                      ['dwl_Qref', 'dwl_dQ', 'dwl_dQdh']):
    dwl['hydro'] = label
    dwl['reaches'] = reaches
    means = dwl.groupby(['hydro', 'reaches']).mean()
    means.drop('bc', level=1, inplace=True)
    ll.append(means)
    
hydro_stats = pd.concat(ll)
hydro_stats.to_csv('hydro_stats.csv')

#-Nr of km exceeding Q16000 reference water levels
data = {'dikeraisekm_Qref': dwl_Qref[dwl_Qref > 0.0].count(),
        'dikeraisekm_dQ': dwl_dQ[dwl_dQ > 0.0].count(),
        'dikeraisekm_dQdh': dwl_dQdh[dwl_dQdh > 0.0].count()}
dikeraise_km = pd.DataFrame(data=data)
dikeraise_km.drop(['hydro', 'reaches'],inplace=True)
dikeraise_km.rename(index={'reference_FM_dQdh':'reference'}, inplace=True)
dikeraise_km.to_csv('dikeraise_km.csv')

#%%
#-extract water levels at the river's axis for three scenarios
msr_names = next(os.walk(msr_fm_root))[1]
wl_Qref = read_wl_ensemble(msr_fm_root, msr_names, var_names=['waterlevel'],
                          time_step=t_Qref)
wl_dQ = read_wl_ensemble(msr_fm_root, msr_names, var_names=['waterlevel'],
                          time_step=t_dQ)
wl_dQdh = read_wl_ensemble(msr_fm_root, msr_names, var_names=['waterlevel'],
                          time_step=t_dQdh)

#-calculate the difference in water level with 'no measures' 
wl_ref = wl_Qref['reference_FM_dQdh']
dwl_Qref = wl_Qref.subtract(wl_ref, axis=0).iloc[29:,:]
dwl_Qref.index = rkm_to_float(dwl_Qref.index.values)
dwl_Qref.to_csv('dwl_Qref.csv')

wl_ref = wl_dQ['reference_FM_dQdh']
dwl_dQ = wl_dQ.subtract(wl_ref, axis=0).iloc[29:,:]
dwl_dQ.index = rkm_to_float(dwl_dQ.index.values)
dwl_dQ.to_csv('dwl_dQdQ.csv')

wl_ref = wl_dQdh['reference_FM_dQdh']
dwl_dQdh = wl_dQdh.subtract(wl_ref, axis=0).iloc[29:,:]
dwl_dQdh.index = rkm_to_float(dwl_dQdh.index.values)
dwl_dQdh.to_csv('dwl_dQdhdQdh.csv')


df = pd.DataFrame({'Q16':dwl_Qref.mean(),
                   'Q18':dwl_dQ.mean(),
                   'Q18dh':dwl_dQdh.mean(),})
df.plot.scatter('Q16', 'Q18')
df.plot.scatter('Q18dh', 'Q18')


#%% Plot lowering for each measure type
msr_types = ['smoothing', 'sidechannel', 'lowering',
             'dikeraising', 'groynelowering', 'minemblowering']
msr_type_dwl = pd.Series([col_name.split('_')[0] for col_name in dwl_Qref.columns])
for dwl, label_ii in zip ([dwl_Qref,  dwl_dQ, dwl_dQdh], ['1 Qref', '2 dQ', '3 dQdh']):
    for msr_type in msr_types:
        cols = msr_type_dwl.isin([msr_type])
        df = dwl.loc[:,cols.values]
        fig, ax = plt.subplots()
        df.plot(ax=ax, legend=False, linewidth=4)
        ax.plot(dwl.reference_FM_dQdh, linewidth=4, label='no msr')
        plt.title(msr_type + '  ' + label_ii)
        plt.ylabel('delta_wl_axis (m)')
        plt.legend(ncol=2, fontsize=10)
        plt.savefig('m_%s_%s.png' % (msr_type, label_ii), dpi=150)
    






















