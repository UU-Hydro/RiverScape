import os 
import pcraster as pcr
import fiona
import subprocess
import collections
import ogr, gdal
import numpy as np
import pysal as ps
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point, LineString, Polygon, MultiPolygon

################################################################################
#%% IO with vector data #########################################################
################################################################################

def dbf2df(dbf_path, index=None, cols=False, incl_index=False):
    '''
    Read a dbf file as a pandas.DataFrame, optionally selecting the index
    variable and which columns are to be loaded.
    __author__  = "Dani Arribas-Bel <darribas@asu.edu> "
    ...
    Arguments
    ---------
    dbf_path    : str
                  Path to the DBF file to be read
    index       : str
                  Name of the column to be used as the index of the DataFrame
    cols        : list
                  List with the names of the columns to be read into the
                  DataFrame. Defaults to False, which reads the whole dbf
    incl_index  : Boolean
                  If True index is included in the DataFrame as a
                  column too. Defaults to False
    Returns
    -------
    df          : DataFrame
                  pandas.DataFrame object created
    '''
    db = ps.open(dbf_path)
    if cols:
        if incl_index:
            cols.append(index)
        vars_to_read = cols
    else:
        vars_to_read = db.header
    data = dict([(var, db.by_col(var)) for var in vars_to_read])
    if index:
        index = db.by_col(index)
        db.close()
        return pd.DataFrame(data, index=index)
    else:
        db.close()
        return pd.DataFrame(data)

def vector2raster(fileFormat, cloneFile, inFile, attribute, layer = None):
    """Rasterize a vector dataset using the map attributes of the cloneFile.
    
    fileFormat: ogr vector format string of inFile
    cloneFile:  PCRaster .map file
    inFile:     input vector file, or ESRI geodatabase
    topo:       topology type: polygon, line, point
    attribute:  item name of the attribute table
    layer:      'PAL', 'LAB', or 'ARC' for AVCBin polygon, point, or line coverages
    layer:      layer in the fileGDB 
    """
    print '\n\tvector2raster of %s ' % inFile
    xmin, xmax, ymin, ymax = getBoundingBox(cloneFile)
    cellLength = getMapAttr(cloneFile)['cell_length']
        
    try:
        os.remove('tmp.tif')
        os.remove('tmp.map')
    except WindowsError:
        pass
    if fileFormat in ['AVCBin', 'FileGDB']:
        cmd = (""""c:\Program Files\GDAL\gdal_rasterize" -te %s %s %s %s """
                              '-tr %s %s '
                              '-ot Float64 '
                              '-a_nodata -9999 '
                              '-l %s '
                              '-a %s ' 
                              '-of GTiff ' 
                              '%s tmp.tif ') %\
              (xmin, ymin, xmax, ymax, cellLength, cellLength, layer, attribute, inFile)
    elif fileFormat == 'ESRI Shapefile':
        cmd = (""""c:\Program Files\GDAL\gdal_rasterize" -te %s %s %s %s """
                              '-tr %s %s '
                              '-ot Float64 '
                              '-a_nodata -9999 '
                              '-a %s ' 
                              '-of GTiff ' 
                              '%s tmp.tif ' ) %\
              (xmin, ymin, xmax, ymax, cellLength, cellLength, attribute, inFile)
    print '\n', cmd
    subprocess.call(cmd, shell=True)
    cmd = 'gdal_translate -of PCRaster -ot Float64 tmp.tif tmp.map'
    subprocess.call(cmd, shell=True)
    outMap = pcr.readmap('tmp.map')
#    os.remove('tmp.tif')
#    os.remove('tmp.map')
    return outMap

def rasterizeFID(fileFormat, cloneFile, inFile, layer = None):
    """Rasterize a vector dataset using the map attributes of the cloneFile.
    
    fileFormat: ogr vector format string of inFile
    cloneFile:  PCRaster .map file
    inFile:     input vector file, or ESRI geodatabase
    attribute:  item name of the attribute table
    layer:      'PAL', 'LAB', or 'ARC' for AVCBin polygon, point, or line coverages
    layer:      layer in the fileGDB 
    """
    print '\n\trasterizeFID of layer %s in %s' % (layer, inFile)
    xmin, xmax, ymin, ymax = getBoundingBox(cloneFile)
    cellLength = getMapAttr(cloneFile)['cell_length']
        
    try:
        os.remove('tmp.tif')
        os.remove('tmp.map')
    except WindowsError:
        pass
    if fileFormat in ['AVCBin', 'FileGDB']:
        cmd = (""""c:\Program Files\GDAL\gdal_rasterize" -te %s %s %s %s """
                              '-tr %s %s '
                              '-ot Float32 '
                              '-a_nodata -9999 '
                              '-a FID -sql "select FID, * from %s" ' 
                              '-of GTiff ' 
                              '%s tmp.tif ') %\
              (xmin, ymin, xmax, ymax, cellLength, cellLength, layer, inFile)
    elif fileFormat == 'ESRI Shapefile':
        cmd = (""""c:\Program Files\GDAL\gdal_rasterize" -te %s %s %s %s """
                              '-tr %s %s '
                              '-ot Float32 '
                              '-a_nodata -9999 '
                              '-a FID -sql "select FID, * from %s" ' 
                              '-of GTiff ' 
                              '%s tmp.tif ') %\
              (xmin, ymin, xmax, ymax, cellLength, cellLength, layer, inFile)
        print cmd
    #stop
    subprocess.call(cmd, shell=True)
    cmd = 'gdal_translate -of PCRaster -ot Float32 tmp.tif tmp.map'
    subprocess.call(cmd, shell=True)
    outMap = pcr.readmap('tmp.map')
    #os.remove('tmp.tif')
    #os.remove('tmp.map')
    return outMap

def shapeString2Ordinal(inFile, outFile, inFieldName, outFieldName):
    """
    convert the string input of inFieldName in inFile to a copy of the shapefile
    with an ordinal field in the outFile as outFieldName
    """

    #-get the unique set of values in a column of the shapefile
    ll = []
    source = fiona.open(inFile)
    for ii in xrange(len(source)):
        rec = next(source)
        ll.append((rec['properties'][inFieldName]))  
    stringClasses = list(set(ll))
    source.close()
    
    #-create a dictionary with strings a keys and integers as value
    lut = {stringClasses[k]:k+1 for k in range(len(stringClasses))}

    #-determine that outFile driver, and coor ref system based on input
    with fiona.open(inFile) as source:
        dstDriver = source.driver
        dstSchema = source.schema.copy()
        # base the schema on the source and the new fields
        inFieldDefinition = dstSchema['properties'][inFieldName]
        dstSchema['properties'] = collections.OrderedDict([(inFieldName, inFieldDefinition),\
                                                            (outFieldName, 'int:8')])
        source.close()
    
    #-fill output with records
    with fiona.open(outFile, 'w', dstDriver, dstSchema) as output:
        source = fiona.open(inFile)
        for ii in xrange(len(source)):
            srcRecord = next(source)
            dstRecord = {}
            dstRecord['geometry'] = srcRecord['geometry']
            dstRecord['id'] = srcRecord['id']
            ecoCode = srcRecord['properties'][inFieldName]
            ecoNr = lut[ecoCode]
            dstRecord['properties'] = collections.OrderedDict([(inFieldName, ecoCode),\
                                                               (outFieldName, ecoNr),\
                                                               ])
            output.write(dstRecord)
    output.close()
    
    return lut

def pcr2Shapefile(srcMap, dstShape, fieldName):
    """
    polygonize a pcraster map into a shapefile
    
    srcMap:       string, path to PCRaster map
    dstShape:     string, path to shapefile
    fieldName:    string, name of the item in the shapefile
    """
    gdal.UseExceptions()
    
    #-create a new shapefile
    driver        = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(dstShape):
        driver.DeleteDataSource(dstShape)
        
    destinationDataSet = driver.CreateDataSource(dstShape)
    destinationLayer   = destinationDataSet.CreateLayer('TEMP', srs = None )
    newField           = ogr.FieldDefn(fieldName, ogr.OFTInteger)
    destinationLayer.CreateField(newField)
    
    #-open the source data and polygonize into new field
    sourceDataSet = gdal.Open(srcMap)
    sourceBand    = sourceDataSet.GetRasterBand(1)
    gdal.Polygonize(sourceBand, None, destinationLayer, 0, [], callback=None)
    
    #-Destroy the destination dataset
    destinationDataSet.Destroy()

def stringVector2raster(inShp, inField, cloneFile, outMap, title = None):
    """Rasterizes a string field in inShp to a PCRaster map. The unique values
    in the shapefile are converted to nominal values. The values and strings
    are contained in the legend of the output map on disk
    
    Input:
    inShp:   input shapefile,
    inField: field to be rasterized
    outMap:  PCRaster map written to disk
    """
    
    # add a column to a copy of the shapefile
    gdf = gpd.read_file(inShp)
    classes = gdf[inField].unique()
    outField = 'TMP_FIELD'
    lut = {classes[k]:k+1 for k in range(len(classes))}
    gdf[outField] = gdf.apply(lambda row: lut[row[inField]], axis=1)
    gdf_subset = gdf.loc[:,['geometry', inField, 'TMP_FIELD']]
    gdf_subset.to_file('tmp.shp')
    
    #-rasterize the copy of the shapefile and report the map with the legend.
    pcrMap = vector2raster('ESRI Shapefile', cloneFile, 'tmp.shp', outField)
    reportMapWithLegend(pcr.nominal(pcrMap), outMap,\
                              {v:k for k,v in lut.iteritems()},\
                              title = title)
    return pcr.readmap(outMap)

################################################################################
#%% IO with raster data #########################################################
################################################################################
  
def cropRaster(inFile, cloneFile):
    """Crop a raster using gdal_translate, return a pcraster object.
    
    cloneFile:  PCRaster .map file
    inFile:     input raster map 
    """
    
    try:
        os.remove('tmp.map')
    except WindowsError:
        pass
    
    xmin, xmax, ymin, ymax = getBoundingBox(cloneFile)
    cmd = ('gdal_translate -projwin %s %s %s %s '
                              '-ot Float32 '
                              '-of PCRaster ' 
                              '%s tmp.map ' ) %\
              (xmin, ymax, xmax, ymin, inFile)
    print '\n', cmd
    subprocess.call(cmd, shell=True)
    outMap = pcr.readmap('tmp.map')
    os.remove('tmp.map')
    return outMap


def emptyMap(cloneMap):
    """Create a boolean map with missing values only"""
    return pcr.ifthen(pcr.scalar(cloneMap) == -9999, pcr.boolean(0))

def empty_map(clone_map):
    """Create a boolean map with missing values only"""
    return pcr.ifthen(pcr.scalar(clone_map) == -9999, pcr.boolean(0))

def makeMap(xmin, ymax, cellLength, nrRows, nrCols, outFile):
    """
    Create a boolean PCRaster map using mapattr.exe

    xmin,xmax,ymin,ymax: bounding box of the area
    resolution:          cell length of the grid
    outFile:             resulting file on disk
    """
    try:
        os.remove(outFile)
    except WindowsError:
        pass
    cmd = 'mapattr -s -B -x %s -y %s -l %s -R %s -C %s %s' %\
          (xmin, ymax, cellLength, nrRows, nrCols, outFile)
    
    subprocess.call(cmd, shell=True)

def make_clone(nrRows, nrCols, cellSize, west, north):
    pcr.setclone(nrRows, nrCols, cellSize, west, north)
    clone = pcr.boolean(1)
    return clone

def col2map(arr, cloneMapName, x=0, y=1, v=2, args = ''):
    """creates a PCRaster map based on cloneMap
	x,y,v (value) are the indices of the columns in arr"""
    g = np.hstack((arr[:,x:x+1],arr[:,y:y+1],arr[:,v:v+1]))
    np.savetxt('temp.txt', g, delimiter=',')
    cmd = 'col2map --clone %s %s temp.txt temp.map'% (cloneMapName, args)
    print '\n', cmd
    subprocess.call(cmd, shell=True)
    outMap = pcr.readmap('temp.map')
    os.remove('temp.txt')
    os.remove('temp.map')
    return outMap


def pcr2col(listOfMaps, MV, selection = 'ONE_TRUE'):
    """converts a set of maps to a column array: X, Y, map values
       selection can be set to ALL, ALL_TRUE, ONE_TRUE"""
    
    #-intersect all maps and get X and Y coordinates
    intersection = pcr.boolean(pcr.cover(listOfMaps[0],0))
    for mapX in listOfMaps[1:]:
        intersection = intersection | pcr.boolean(pcr.cover(mapX,0))
    pcr.setglobaloption("unittrue")
    xCoor = pcr.ifthen(intersection, pcr.xcoordinate(intersection))
    yCoor = pcr.ifthen(intersection, pcr.ycoordinate(intersection))
    pcr.setglobaloption("unitcell")
    
    #-initiate outArray with xCoor and yCoor
    xCoorArr = pcr.pcr2numpy(xCoor, MV)
    yCoorArr = pcr.pcr2numpy(yCoor, MV)
    nRows, nCols = xCoorArr.shape
    nrCells  = nRows * nCols
    outArray = np.hstack((xCoorArr.reshape(nrCells,1), yCoorArr.reshape(nrCells,1)))
    
    #-add subsequent maps
    for mapX in listOfMaps:
        arr = pcr.pcr2numpy(mapX, MV).reshape(nrCells,1)
        outArray = np.hstack((outArray, arr))
    
    #-subset output based on selection criterium
    ll = []
    nrMaps = len(listOfMaps)
    if selection == 'ONE_TRUE':
        for line in outArray:
            nrMV = len(line[line == MV])
            if nrMV < nrMaps:
                ll.append(line)
            else:
                pass
        outArray = np.array(ll)
    elif selection == 'ALL_TRUE':
        for line in outArray:
            if MV not in line:
                ll.append(line)
            else:
                pass
        outArray = np.array(ll)
    elif selection == 'ALL':
        pass
    return outArray


def xyzWAQUA2Pcr(xyzFile, gridIDMap, cloneFile):
    """read an WAQUA .xyz file into a raster map
    
    xyzFile:    x,y,z,m,n,id column file with a one line header
    gridIDmap:  pcraster map with the rgf grid cell IDs
    cloneFile:  pcraster clonefile to be used in col2map
    """
    
    xyz = np.loadtxt(xyzFile, delimiter = ',', skiprows=1)
    xyzPointMap = col2map(xyz, cloneFile, args = '-S -a -m -999999')
    xyzAreaMap = pcr.areaaverage(xyzPointMap, pcr.nominal(gridIDMap))
    return xyzAreaMap

def getMapAttr(fileName):
    ''' list the map attributes in a dictionary, keys are mapattr output codes '''
    d={}
    cmd = 'mapattr -p %s' % fileName
    raw = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    for row in raw.split('\n')[:-1]:
        
        items = row.split()
        k,v = items[0], items[1]
        try:
            vScalar = float(v)
            d[k] = vScalar
        except ValueError:
            d[k] = v
    return d

def getBoundingBox(mapFile):
    mapAtt = getMapAttr(mapFile)
    cellLength = mapAtt['cell_length']
    nCols  = mapAtt['columns']
    nRows  = mapAtt['rows']
    xmin   = mapAtt['xUL']
    ymax   = mapAtt['yUL']
    xmax   = xmin + nCols * cellLength
    ymin   = ymax - nRows * cellLength
    return xmin, xmax, ymin, ymax

def readLegend(mapFile):
    """
    Read a pcraster legend into a dictionary.
    """
    cmd = 'legend -w lut.tmp %s' % mapFile
    subprocess.call(cmd, shell = True)
    lut = {}
    for line in file('lut.tmp', 'r').readlines()[1:]:
        items = line.split(' ')
        key = items[0]
        value = items[1].split('_')[1].rstrip()
        lut[key] = value
    title = file('lut.tmp', 'r').readline().split(' ')[1].rstrip()
    return lut, title

def reportLegend(mapFile, legendDict, title = None):
    """
    attaches a legend to a PCRaster map
    """
    legendFile = open('tmp.legend', 'w')
    legendFile.write('-0 %s\n' % title)
    for key in legendDict.keys():
        legendFile.write('%s %s_%s\n' % (key, key, legendDict[key]))
    legendFile.close()
    cmd = 'legend -f %s %s' % ('tmp.legend', mapFile)
    subprocess.call(cmd, shell = True)

def readMapWithLegend(pcrFile):
    """
    Read a PCRaster map file with a legend into a PCRaster object and a dictionary
    The dictionary keys represent the nominal values of the map, and the values
    represent the labels of the legend, separated by the underscore
    """
    pcrMap = pcr.readmap(pcrFile)
    pcrLegend, title = readLegend(pcrFile)
    return pcrMap, pcrLegend, title

def reportMapWithLegend(pcrMap, pcrFile, legendDict, title = None):
    """
    Repor a PCRaster file and attach a legend to it
    
    pcrMap: ordinal map
    legendDict:     keys = nominal values present in pcrMap
                    values = labels as strings
    labels are given as: <classNr>_<label> to optimize readability of the map
    """
    pcr.report(pcrMap, pcrFile)
    reportLegend(pcrFile, legendDict, title = title)

def read_map_with_legend(pcr_file):
    """
    Read map and legend into LegendMap class for nominal or ordinal data.
    The legend needs 'key_label' pairs, separated by an underscore. For example
    '1_UM-1' links map values of 1 to 'UM-1'
    
    Returns a MapLegend class
    """
    legend = read_legend(pcr_file)
    pcr_map = pcr.readmap(pcr_file)
    return LegendMap(pcr_map, legend)

def read_legend(pcr_file):
    """
    Read a pcraster legend into a data frame.
    """
    cmd = 'legend -w legend.tmp %s' % pcr_file
    subprocess.call(cmd, shell = True)
    df = pd.read_csv('legend.tmp', sep=' ')
    title = df.columns[1]
    data = {'values':df.iloc[:,0],
             title: df.iloc[:,1].str.split('_', expand=True).iloc[:,1]}
    legend = pd.DataFrame(data=data)
    return legend

def report_map_with_legend(legend_map_class, pcr_file):
    """
    Report an ordinal, or nominal map and attach a legend to it.
    """
    # Report the map
    pcr.report(legend_map_class.pcr_map, pcr_file)    
    
    # Create the legend DataFrame
    df = legend_map_class.legend_df
    columns = list(df.columns.values)
    columns.remove('values')
    title = columns[0]
    
    df['labels'] = df.apply(lambda row: str(row['values']) + '_' + row[title],
                                   axis=1)
    legend = df.loc[:,['values', 'labels']]
    legend.columns = ['-0', title]
    
    # Attach the legend
    legend.to_csv('tmp.legend', sep=' ', index=False)
    cmd = 'legend -f %s %s' % ('tmp.legend', pcr_file)
    subprocess.call(cmd, shell = True)

class LegendMap(object):
    """
    Class for storing a PCRaster map with legend and legend title.
    -----------
    Parameters:
    pcr_map: PCRaster map object
    legend: pandas DataFrame with map values and labels
    title: string
    """
    def __init__(self, pcr_map, legend_df):
        self.pcr_map = pcr_map
        self.legend_df = legend_df

def read_pli(pli_file):
    """
    Read a .pli file into a dictionary
    -------
    input 
        pliFile: .pli, .pliz, or .pol input file for Delft3D-FM
    """
    def line2floats(line):
        return [float(i) for i in line.split()]
        
    with open(pli_file) as f:
        lines  = f.read().splitlines()
        
    #=loop over the polylines and store the data in a dictionary
    data = {}
    cnt = 0
    while cnt < len(lines):
        label = lines[cnt]
        nrow, ncol = lines[cnt+1].split()
        nrow, ncol = int(nrow), int(ncol)    
        pol_data = [line2floats(line) for line in lines[cnt+2: cnt+2+nrow]]
        cnt += nrow + 2
        data[label] = np.array(pol_data)
    print len(data.keys()), 'polylines read in:\n\t %s' % (pli_file,)
    return data

def pli_dict2geodataframe(pli_dict, out_type = 'Point'):
    """Convert a dictionary with polyline data to a geopandas GeoDataFrame
    -------
    Input:
        pliDict : dictionary with polyline data.
        outType : can be 'Point', 'LineString', or 'Polygon' GeoDataFrame
    
    If 'LineString' or 'Polygon' is selected, attribute data will be lost.
    """
    
    container = [] 
    for label, data in pli_dict.iteritems():
        if out_type == 'Point':
            points = gpd.GeoSeries([Point(x,y) for x,y in data[:,0:2]])
            geodata = gpd.GeoDataFrame(geometry = points)
            attributes = pd.DataFrame(data=data)
            gdf = pd.concat((geodata, attributes), axis=1)
            colNameList = ['geometry'] + list('abcdefghijklmnop')[0:data.shape[1]]
            gdf.columns = colNameList
            gdf['label'] = label
            
        elif out_type == 'LineString':
            xyList = [[x,y] for x,y in data[:,0:2]]
            line = gpd.GeoSeries(LineString(xyList))
            gdf = gpd.GeoDataFrame(geometry = line)
            gdf['label'] = label
        
        elif out_type == 'Polygon':
            try:
                xyList = [[x,y] for x,y in data[:,0:2]]
                line = gpd.GeoSeries(Polygon(xyList))
                gdf = gpd.GeoDataFrame(geometry = line)
                gdf['label'] = label
            except ValueError:
                pass

        container.append(gdf)
    out_df = pd.concat(container)
    out_df.index = np.arange(len(out_df))
    return gpd.GeoDataFrame(out_df)

def geoDataFrame2pli(geodataframe, pli_file, geometry = 'Point'):
    """Converts a geopandas.GeoDataFrame to a pli file on disk.
        
    Extension conventions are:
            .pli   for polyline files
            .pliz  for polyline files with addional attributes
            .pol   for polyline files with identical first and last coordinate
    input:
        geoDataFrame  :    geopandas GeoDataFrame
        pliFile       :    path to the output file
        geometry      :    shapely geometry (Point, LineString, Polygon)    
    Returns the file path.
    """
    print 'Writing polyline data to:\n\t', pli_file
    with open(pli_file, 'w') as f:
        if geometry == 'Point':
            for label, df in geodataframe.groupby('label'):
                assert len(df.shape) == 2
                df = df.loc[:,['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']]
                f.write(label+'\n')
                f.write('%s %s\n' % df.shape)
                f.write(df.to_string(header=False, index=False))
                f.write('\n')
        elif geometry == 'LineString':
            pass
        elif geometry == 'Polygon':
            for index, pol, label in geodataframe.loc[:,['geometry', 'label']].itertuples():
                if pol.geom_type == 'MultiPolygon':
                    # TO DO: separate multipolygons and create new label
                    #        now the whole polygon is removed
                    pass
                else:
                    x,y = pol.exterior.coords.xy 
                    xy_coords = np.array([[xy[0], xy[1]] for xy in zip(x,y)])
                    f.write(label+'\n')
                    f.write('%s %s\n' % xy_coords.shape)
                    for xi, yi in xy_coords:
                        f.write('%.4f  %.4f\n' % (xi, yi))
    return pli_file

def pliz2shp(in_pliz_file, out_shp_file, out_type = 'Point'):
    """Convert a Delft3D-FM polygon file with z coordinates (.pliz) to a ESRI
    shapefile with separate columns for label type
    """
    if out_type == 'Point':
        points = pli_dict2geodataframe(read_pli(in_pliz_file), out_type = 'Point')
        labels = points.label.str.split(':', expand = True)
        labels.columns = ['labelNr', 'labelType']
        labeled_points = gpd.GeoDataFrame(pd.concat((points, labels), axis=1))       
        labeled_points.to_file(out_shp_file)
    elif out_type == 'LineString':
        lines = pli_dict2geodataframe(read_pli(in_pliz_file), out_type = 'LineString')
        labels = lines.label.str.split(':', expand = True)
        labels.columns = ['labelNr', 'labelType']
        labeledLines = gpd.GeoDataFrame(pd.concat((lines, labels), axis=1))       
        labeledLines.to_file(out_shp_file)