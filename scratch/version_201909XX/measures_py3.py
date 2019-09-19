#-import generic modules
import os
import string
import subprocess
import numpy as np
from collections import OrderedDict
import math
import pandas as pd
import geopandas as gpd
from scipy.spatial import Delaunay
from shapely.geometry import MultiLineString
from shapely.ops import cascaded_union, polygonize
import time

#-import spatial modules
import pcraster as pcr

#-import RiverScape modules
import mapIO
import pcrRecipes
#~ reload(mapIO)
#~ reload(pcrRecipes)

def assign_ecotopes(area, eco_string, legend_df):
    """
    Assign ecotope_nr to the area based on a lookup of 'eco_code'
    in the legend dataframe. Return a LegendMap claas
    """
    df = legend_df.set_index('ECO_CODE', drop=True)
    eco_nr = df.loc[eco_string, 'values']
    eco_map = pcr.ifthen(area, pcr.nominal(int(eco_nr)))
    ecotopes = LegendMap(eco_map, legend_df)
    return ecotopes

def alpha_shape(points, alpha):
    """
    Determine the alpha shape of 2D set of points.
    -------
    parameters:
    points: 2d column array with x and y coordinates
    alpha: float, threshold for minimum edge length
    
    Returns a shapely Polygon object.
    """
    tri = Delaunay(points)
    edges = set()
    edge_points = []
    def add_edge(i, j):
         """Add a line between the i-th and j-th points,
         if not in the list already"""
         if (i, j) in edges or (j, i) in edges:
            # already added
            return
         edges.add( (i, j) )
         edge_points.append(points[ [i, j] ])
         
    for ia, ib, ic in tri.vertices:
         add_edge(ia, ib)
         add_edge(ib, ic)
         add_edge(ic, ia)
         
    edges = set()
    edge_points = []
    # loop over triangles: ia, ib, ic = indices of triangle corners
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)
        # Here's the radius filter.
        if circum_r < alpha:
            add_edge(ia, ib)
            add_edge(ib, ic)
            add_edge(ic, ia)
    
    m = MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return cascaded_union(triangles)

def map2xy(pcr_map):
    """
    Extract the coordinates of pcr_map and return as a dataframe.
    ---------
    Parameters
        pcr_map: pcraster object
    
    Returns a pandas DataFrame with 'x' and 'y' as column labels.
    """
    bool_map = pcr.boolean(pcr_map)
    x_coor = pcr.pcr2numpy(pcr.xcoordinate(bool_map), -9999).flatten()
    y_coor = pcr.pcr2numpy(pcr.ycoordinate(bool_map), -9999).flatten()
    df = pd.DataFrame(data={'x':x_coor, 'y':y_coor})
    return df[df.x > -9999]

def alpha_shape_pcr(pcr_map, alpha_hull, sparse=None):
    """
    Extract the alpha hull of a PCRaster map.
    Alpha shape is slow when a dense raster is used. Sparse is used in 
    combination with pcr_map to speed up the function when pcr_map represents
    parts of the outline, e.g. dikes, and sparse the inside area, e.g. winbed.
    -----------
    Parameters:
        pcr_map: PCRaster object used to extract alpha shape
        alpha_hull: float, representing the minimum cell length
        clone_file: path to PCRaster clone file, used as rasterizing template
        sparse: PCRaster object used to sparsely fill in the inside of pcr_map
    
    Returns a PCRaster object.
    """
    xy_pcr = map2xy(pcr_map)
    points = xy_pcr.copy()
    if isinstance(sparse, pcr._pcraster.Field) == True:
        xy_sparse = map2xy(sparse)
        xy_sparse = xy_sparse.iloc[np.arange(0,len(xy_sparse), 25),:]
        points = pd.concat([xy_pcr, xy_sparse]).values
    alpha_pol = alpha_shape(points, alpha_hull)
    gdf = gpd.GeoDataFrame(pd.DataFrame(data={'geometry':[alpha_pol], 'a':1}))
    gdf.to_file('alpha_shape.shp')
    pcr.report(pcr_map, 'clone_temp.map')
    alpha_map = mapIO.vector2raster('ESRI Shapefile', 'clone_temp.map',
                                    'alpha_shape.shp', 'a')
    return alpha_map

def alpha_shape_lastools(inMap, cloneFile, length = 1000):
    """
    Compute the alpha shape of a boolean map using lasboundary.exe of lastools.
    ------------
    Parameters    
    inMap    : PCRaster object
    cloneFile: PCRaster clone map on disk for bounding box and resolution.
    length   : smoothness of the alpha shape, higher values give smoother outline
    """
    
    xmin, xmax, ymin, ymax = mapIO.getBoundingBox(cloneFile)
    cellLength = mapIO.getMapAttr(cloneFile)['cell_length']
    
    pointArray = mapIO.pcr2col([inMap], -9999)
    np.savetxt('tmp.txt', pointArray)
    cmd1 = 'lasboundary -i tmp.txt -o tmp.shp -concavity %s' % length
    subprocess.call(cmd1, shell=True)
    cmd2 = ('gdal_rasterize -burn 1 -te %s %s %s %s -tr %s %s '
                          ' tmp.shp tmp.tif') %\
           (xmin, ymin, xmax, ymax, cellLength, cellLength)
    subprocess.call(cmd2, shell=True)             
    subprocess.call('gdal_translate -of PCRaster -ot Float32 tmp.tif alphaMap.tmp')
    
    alphaMap = pcr.readmap('alphaMap.tmp')
    alphaMap = pcr.ifthen(alphaMap == 1.0, pcr.scalar(length))
    return alphaMap

def read_geom_maps(map_dir):
    """
    Read computational lulc maps into a class.
    """
    clone = pcr.readmap(os.path.join(map_dir, 'clone.map'))
    dem = pcr.readmap(os.path.join(map_dir, 'dem.map'))
    dist_to_groyne_field = pcr.readmap(os.path.join(map_dir,
                                                   'dist_to_groyne_field.map'))
    dist_to_main_channel = pcr.readmap(os.path.join(map_dir,
                                                   'dist_to_main_channel.map'))
    dist_to_main_dike = pcr.readmap(os.path.join(map_dir,
                                                   'dist_to_main_dike.map'))
    flpl_width = pcr.readmap(os.path.join(map_dir,
                                                   'flpl_width.map'))
    flpl_narrow = pcr.readmap(os.path.join(map_dir,
                                                 'flpl_narrow.map'))
    flpl_wide = pcr.readmap(os.path.join(map_dir, 'flpl_wide.map'))
    main_channel_width = pcr.readmap(os.path.join(map_dir,
                                                  'main_channel_width.map'))
    river_sides = pcr.readmap(os.path.join(map_dir, 'river_sides.map'))
    shore_line = pcr.readmap(os.path.join(map_dir, 'shore_line.map'))
    geom = RiverGeometry(clone, dem, dist_to_main_dike, dist_to_groyne_field, 
                         dist_to_main_channel, flpl_width,
                         flpl_narrow, flpl_wide,
                         main_channel_width, river_sides, shore_line)
    return geom

def read_lulc_maps(map_dir):
    """
    Read lulc maps into a class.
    """
    backwaters = pcr.readmap(os.path.join(map_dir, 'backwaters.map'))
    ecotopes = read_map_with_legend(os.path.join(map_dir, 'ecotopes.map'))
    floodplain = pcr.readmap(os.path.join(map_dir, 'floodplain.map'))
    groyne_field = pcr.readmap(os.path.join(map_dir, 'groyne_field.map'))
    main_channel = pcr.readmap(os.path.join(map_dir, 'main_channel.map'))
    trachytopes = pcr.readmap(os.path.join(map_dir, 'trachytopes.map'))
    sections = pcr.readmap(os.path.join(map_dir, 'sections.map'))
    winter_bed = pcr.readmap(os.path.join(map_dir, 'winter_bed.map'))
    real_estate_value = pcr.readmap(os.path.join(map_dir, 'cost_building_tot.map'))
    
    lulc = LandUseLandCover(backwaters, ecotopes, floodplain, groyne_field,
                 main_channel, trachytopes, sections, winter_bed,
                 real_estate_value)
    return lulc

def read_mesh_maps(map_dir):
    """
    Read computational mesh maps into a class.
    """
    fm_ID = pcr.readmap(os.path.join(map_dir, 'fm_ID.map'))
    m_coor = pcr.readmap(os.path.join(map_dir, 'm_coor.map'))
    n_coor = pcr.readmap(os.path.join(map_dir, 'n_coor.map'))
    grid_ID = pcr.readmap(os.path.join(map_dir, 'grid_ID.map'))
    mesh = Mesh(fm_ID, m_coor, n_coor, grid_ID)
    return mesh 

def read_axis_maps(map_dir):
    """
    Read river axis related maps and return an Axis class.
    """
    location = pcr.readmap(os.path.join(map_dir, 'river_axis.map'))
    radius = pcr.readmap(os.path.join(map_dir, 'axis_radius.map'))
    turning_direction = pcr.readmap(os.path.join(map_dir, 
                                                 'turning_direction.map'))
    rkm = pcr.readmap(os.path.join(map_dir, 'rkm_axis.map'))
    rkm_point = pcr.readmap(os.path.join(map_dir, 'rkm_point.map'))
    rkm_line = pcr.readmap(os.path.join(map_dir, 'rkm_line.map'))
    rkm_full = pcr.readmap(os.path.join(map_dir, 'rkm_full.map'))
    axis = RiverAxis(location, radius, turning_direction,
                     rkm, rkm_point, rkm_line, rkm_full)
    return axis

def read_dike_maps(map_dir, dike_type):
    """
    Read embankment related maps: main_dike, minemb, groynes.
    """
    location = pcr.readmap(os.path.join(map_dir, '%s_loc.map' % dike_type))
    length = pcr.readmap(os.path.join(map_dir, '%s_len.map' % dike_type))
    volume = pcr.readmap(os.path.join(map_dir, '%s_vol.map' % dike_type))
    height = pcr.readmap(os.path.join(map_dir, '%s_height.map' % dike_type))
    return RiverEmbankments(location, length, volume, height)

def read_hydro_maps(map_dir):
    """
    Read hydrodynamic attributes into a class.
    """
    # attributes at design descharge
    chezy = pcr.readmap(os.path.join(map_dir, 'chezy.map'))
    nikuradse = pcr.readmap(os.path.join(map_dir, 'nikuradse.map'))
    specific_q = pcr.readmap(os.path.join(map_dir, 'specific_q.map'))
    velocity = pcr.readmap(os.path.join(map_dir, 'velocity.map'))
    water_depth = pcr.readmap(os.path.join(map_dir, 'water_depth.map'))
    water_level = pcr.readmap(os.path.join(map_dir, 'water_level.map'))
    
    # water levels exceeded a number of days per year
    wl_exc2d = pcr.readmap(os.path.join(map_dir, 'wl_exc2d.map'))
    wl_exc20d = pcr.readmap(os.path.join(map_dir, 'wl_exc20d.map'))
    wl_exc50d = pcr.readmap(os.path.join(map_dir, 'wl_exc50d.map'))
    wl_exc100d = pcr.readmap(os.path.join(map_dir, 'wl_exc100d.map'))
    wl_exc150d = pcr.readmap(os.path.join(map_dir, 'wl_exc150d.map'))
    wl_exc363d = pcr.readmap(os.path.join(map_dir, 'wl_exc363d.map'))
    
    hydro = RiverHydro(chezy, nikuradse,
                       specific_q, velocity, water_depth, water_level,
                       wl_exc2d, wl_exc20d, wl_exc50d, wl_exc100d, wl_exc150d,
                       wl_exc363d)
    return hydro

def read_map_with_legend(pcr_file):
    """
    Read map and legend into LegendMap class for nominal or ordinal data.
    The legend needs 'key_label' pairs, separated by an underscore. For example
    '1_UM-1' links map values of 1 to 'UM-1'
    
    Returns a MapLegend class
    """
    # Read a pcraster legend into a data frame
    cmd = 'legend -w legend.tmp %s' % pcr_file
    subprocess.call(cmd, shell = True)
    df = pd.read_csv('legend.tmp', sep=' ')
    title = df.columns[1]
    data = {'values':df.iloc[:,0],
             title: df.iloc[:,1].str.split('_', expand=True).iloc[:,1]}
    legend = pd.DataFrame(data=data)
    
    pcr_map = pcr.readmap(pcr_file)
    return LegendMap(pcr_map, legend)

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


def percentile_slicing(pcr_map, percentile):
    """
    Map the spatial distribution where the map exceeds the score at percentile.
    ---------
    Parameters
        pcr_map: PCRaster object
        percentile: float between 0 and 100
    Returns a boolean map with the exceedance distribution:
        1 = values higher than score at percentile
        0 = values lower than score at percentile
    """
    score = map_score_at_percentile(pcr_map, percentile)
    return pcr_map > score

def map_score_at_percentile(pcr_map, percentile):
    """
    Determine the score at percentile of pcr_map's non missing values.
    """
    values = pcr.pcr2numpy(pcr_map, -9999).flatten()
    non_missing_values = values[values[:] > -9999]
    score_at_percentile = np.percentile(non_missing_values, percentile)    
    return score_at_percentile


def unitcell2unittrue(in_values, geom_type=None):
    """
    Convenience function to convert from 'unitcell' to 'unittrue' dependent on
    the geometry type.
    ---------
    parameters:
    in_values: float, PCRaster scalar map, numpy array
    geometry_type: string
                   options are 'length' or 'area'
    returns a value in unittrue
    """
    assert isinstance(geom_type, basestring)
    assert geom_type in ['length', 'area']
    
    cell_length = pcr.clone().cellSize()
    cell_area = cell_length**2
    if geom_type == 'length':
        return in_values * cell_length
    if geom_type == 'area':
        return in_values * cell_area
    
def unittrue2unitcell(in_values, geom_type=None):
    """
    Convenience function to convert from 'unittrue' to 'unitcell' dependent on
    the geometry type.
    ---------
    parameters:
    in_values: float, PCRaster scalar map, numpy array
    geometry_type: string
                   options are 'length' or 'area'
    returns a value in unitcell
    """
    assert isinstance(geom_type, basestring)
    assert geom_type in ['length', 'area']
    
    cell_length = pcr.clone().cellSize()
    cell_area = cell_length**2
    if geom_type == 'length':
        return in_values / cell_length
    if geom_type == 'area':
        return in_values / cell_area

def spread_values_from_points(values, points, mask=None):
    """
    Extrapolatie the 'values' at 'points' over the whole area or mask.
    
    Values are temporarily multiplied by 1000 and converted to ordinal values 
    for the extrapolation. 
    -----------
    Parameters:
        values: PCRaster map
        points: PCRaster map with the points for the spreading operation.
        mask:   boolean mask with area to limit the extrapolation to.
    
    Returns a scalar map with extrapolated values.
    """
    point_values = pcr.ifthen(pcr.boolean(points), pcr.scalar(values))
    point_values = pcr.nominal(pcr.cover((1e5 + 1000 * point_values), 0))
    spread_values = pcr.spreadzone(point_values, 0, 1)
    if isinstance(mask, pcr._pcraster.Field) == True:
        spread_values = pcr.ifthen(mask, spread_values)
    return (pcr.scalar(spread_values) - 1e5) / 1000

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

def write_measure(msr, msr_root_dir):
    """
    Write the measure's maps and settings to a new folder on disk.
    TO DO: make explicit directory paths after combination has been done.
    """
    settings = msr.settings
    # Create the directory
    #~ msr_dir_name = "_".join([settings['msr_type'], settings['ID'])
    msr_dir_name = str(settings['msr_type']) + "_" +  str((settings['ID']))
    msr_dir_path = os.path.join(msr_root_dir, msr_dir_name)
    pcrRecipes.make_dir(msr_dir_path)
    print(msr_dir_path)
    
    # Report the maps
    pcr.report(msr.area, os.path.join(msr_dir_path, 'area.map'))
    pcr.report(msr.dem, os.path.join(msr_dir_path, 'dem.map'))
    pcr.report(msr.trachytopes,
               os.path.join(msr_dir_path, 'trachytopes.map'))
    pcr.report(msr.groyne_height,
               os.path.join(msr_dir_path, 'groyne_height.map'))
    pcr.report(msr.minemb_height,
               os.path.join(msr_dir_path, 'minemb_height.map'))
    pcr.report(msr.main_dike_height,
               os.path.join(msr_dir_path, 'main_dike_height.map'))
    report_map_with_legend(msr.ecotopes, 
                           os.path.join(msr_dir_path, 'ecotopes.map'))
    with open(os.path.join(msr_dir_path, 'settings.txt'), 'w') as outfile:
        for k,v in settings.items():
            outfile.write('%s:%s\n' % (k,v))


class River(object):
    """
    Generic attributes of a river section. 
    Attribures are grouped in:
        * geom: geomtric aspects (height, distances)
        * dikes: main embankment, minor embankments, groynes, height difference
        * axis: river axis
        * hydro: hydrodynamic components as water level and flow velocity
        * lulc: land use and land cover
        * mesh: computational mesh
    """
    
    def __init__(self, name, river_axis,
                 main_dike, minemb, groynes, 
                 hydro, mesh, land_use_land_cover, geometry):
        self.name = name
        self.axis = river_axis
        self.main_dike = main_dike
        self.minemb = minemb
        self.groynes = groynes
        self.hydro = hydro
        self.mesh = mesh
        self.lulc = land_use_land_cover
        self.geom = geometry

        
class RiverAxis(object):
    """
    Attributes of the river axis.
    """
    def __init__(self, location, radius, turning_direction,
                 rkm, rkm_point, rkm_line, rkm_full):
        self.location = location
        self.radius = radius
        self.turning_direction = turning_direction
        self.rkm = rkm
        self.rkm_point = rkm_point
        self.rkm_line = rkm_line
        self.rkm_full = rkm_full


class RiverEmbankments(object):
    """
    Attributes of the river embankments.    
    """
    def __init__(self, location, length, volume, height):
        self.location = location
        self.length = length
        self.volume = volume
        self.height = height
    
    def plot(self):
        """
        Plot all PCRaster maps using aguila
        """
        ll = [pcr_map for var, pcr_map in self.__dict__.iteritems()]
        pcr.aguila(ll)


class RiverHydro(object):
    """
    Class with hydrodynamic properties of the river.
    """
    def __init__(self,
           chezy, nikuradse, specific_q, velocity, water_depth, water_level,
           wl_exc2d, wl_exc20d, wl_exc50d, wl_exc100d, wl_exc150d, wl_exc363d
           ):
         self.chezy = chezy
         self.nikuradse = nikuradse
         self.specific_q = specific_q
         self.velocity = velocity
         self.water_depth = water_depth
         self.water_level = water_level
         self.wl_exc2d = wl_exc2d
         self.wl_exc20d = wl_exc20d
         self.wl_exc50d = wl_exc50d
         self.wl_exc100d = wl_exc100d
         self.wl_exc150d = wl_exc150d
         self.wl_exc363d = wl_exc363d


class Mesh(object):
    """
    Class with computational mesh attributes.
    """
    def __init__(self, fm_ID, m_coor, n_coor, grid_ID):
        self.fm_ID = fm_ID
        self.m_coor = m_coor
        self.n_coor = n_coor
        self.grid_ID = grid_ID


class LandUseLandCover(object):
    """
    Class with land use and land cover attributes.
    """
    def __init__(self, backwaters, ecotopes, floodplain, groyne_field,
                 main_channel, trachytopes, sections, winter_bed,
                 real_estate_value):
        self.backwaters = backwaters
        self.ecotopes = ecotopes
        self.floodplain = floodplain
        self.groyne_field = groyne_field
        self.main_channel = main_channel
        self.trachytopes = trachytopes
        self.sections = sections
        self.winter_bed = winter_bed
        self.real_estate_value = real_estate_value


class RiverGeometry(object):
    """
    Class with geometry attributes.
    """
    def __init__(self, clone, dem, dist_to_main_dike, dist_to_groyne_field,
                 dist_to_main_channel, flpl_width, flpl_narrow,
                 flpl_wide, main_channel_width, river_side, shore_line):
        self.clone = clone
        self.dem = dem
        self.dist_to_main_dike = dist_to_main_dike
        self.dist_to_groyne_field = dist_to_groyne_field
        self.dist_to_main_channel = dist_to_main_channel
        self.flpl_width = flpl_width
        self.flpl_narrow = flpl_narrow
        self.flpl_wide = flpl_wide
        self.main_channel_width = main_channel_width
        self.river_side = river_side
        self.shore_line = shore_line


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

class Measure(object):
    """
    Contains spatial distributions of a RiverScape measure.
    """
    def __init__(self, settings,
                 area, dem, ecotopes, trachytopes,
                 groyne_height, minemb_height, main_dike_height):
        """
        Class containing the maps defining the measure:
             name, dem, area, ecotopes, trachytopes, and a DataFrame with pliz
             data.
        """
        self.settings = settings
        self.area = area
        self.dem = dem
        self.ecotopes = ecotopes
        self.trachytopes = trachytopes
        self.groyne_height = groyne_height
        self.minemb_height = minemb_height
        self.main_dike_height = main_dike_height
    
    def split(self, mask):
        """
        Split up the measure into separate measures for each area in "mask".
        Returns a list of measure objects.
        """
        pass
    
    
    def combine(self, other):
        """
        Combine two measures into one using attribute-specific methods.
        ---
        input:
            other: Measure object
            
        Attribute, methods:
            mrs_type: string concatenation with "|"
            ID: string concatenation with "|"
            area: pcr.cover(self, other)
            dem: pcr.cover(self, other). Minimum height in case of overlap
            ecotopes: pcr.cover(self, other). Minimum height ecotope is chosen
            trachytopes : pcr.cover(self, other). Minimum height trachytope
                          is chosen
            groyne_height: like dem
            minemb_height: like dem
            main_dike_height: like dem
        """
        
        def c_height(dem1, dem2, area):
            """
            Combine maps by taking the minimum in case of overlap.
            """
            dummy = pcr.scalar(area * 10000)
            return pcr.min(pcr.cover(dem1, dummy), pcr.cover(dem, dummy))
        
        def c_lulc(lulc_1, lulc_2, dem1, dem2):
            """
            Combine two land_use_land_cover maps. Lulc of minimnum height is
            used in case of overlap.            
            """
            lulc_overlap = pcr.ifthenelse(dem1 <= dem2, lulc_1, lulc_2)
            return pcr.cover(lulc_overlap, lulc_1, lulc_2)
        
        assert isinstance(other, Measure)
        msr_type = self.measure_type + "|" + other.measure_type
        ID = str(self.ID) + "|" + str(other.ID)
        area = pcr.cover(self.area, other.area)
        dem = c_height(self.dem, other.dem, area)
        groyne_height = c_height(self.groyne_height, other.groyne_height, area)
        minemb_height = c_height(self.minemb_height, other.minemb_height, area)
        main_dike_height = c_height(self.main_dike_height, 
                                          other.main_dike_height, area)
        ecotopes = c_lulc(self.ecotopes, other.ecotopes,
                          self.dem, other.dem)
        trachytopes = c_lulc(self.trachytopes, other.trachytopes,
                             self.dem, other.dem)

        new_msr = Measure(msr_type, ID, area, dem, ecotopes, trachytopes, 
                          groyne_height, minemb_height, main_dike_height)
        return new_msr
    
    def plot(self):
        pcr.aguila(self.area, self.dem, self.ecotopes.pcr_map, 
                   self.trachytopes, self.groyne_height, 
                   self.minemb_height, self.main_dike_height)
        area             = self.area             ; pcr.plot(self.area)
        dem              = self.dem              ; pcr.plot(self.dem)
        ecotopes         = self.ecotopes.pcr_map ; pcr.plot(ecotopes)
        trachytopes      = self.trachytopes      ; pcr.plot(self.trachytopes)
        groyne_height    = self.groyne_height    ; pcr.plot(self.groyne_height)
        minemb_height    = self.minemb_height    ; pcr.plot(self.minemb_height)
        main_dike_height = self.main_dike_height ; pcr.plot(self.main_dike_height)
                   
    
    def mask_out(self, out_mask):
        """Limit the measure to the spatial extent of 'area'"""
        eco_masked = LegendMap(pcr.ifthen(out_mask, self.ecotopes.pcr_map),
                               self.ecotopes.legend_df)
        new_msr = Measure(self.settings,
                          pcr.ifthen(out_mask, self.area),
                          pcr.ifthen(out_mask, self.dem),
                          eco_masked,
                          pcr.ifthen(out_mask, self.trachytopes),
                          pcr.ifthen(out_mask, self.groyne_height),
                          pcr.ifthen(out_mask, self.minemb_height),
                          pcr.ifthen(out_mask, self.main_dike_height))
        return new_msr


class RiverMeasures(object):
    """
    Class to generate river measures.
    """
    def __init__(self, river):
        self.r = river
        self.extent = clone_attributes()[0:4]
        self.cell_size = clone_attributes()[6]
        self.settings = {}
    
    def lowering_area(self, percentile, mask=None):
        """
        Designate areas with high potential for floodplain lowering. Areas 
        with high flow velocity and small water depths are prioritized.
        ----------
        Parameters
        percentile: 0-100, low value is most effecient locations
        mask: boolean PCRaster map, area of application at boolean True
        
        Returns a boolean map with the optimal area for flpl lowering. 
        """
        # Mask out water depth of aquatic regions
        depth_detail = self.r.hydro.water_level - self.r.geom.dem
        depth_detail = pcr.ifthen(self.r.lulc.main_channel == pcr.boolean(0),
                                  depth_detail)
        depth_detail = pcr.ifthen(self.r.lulc.backwaters == pcr.boolean(0),
                                  depth_detail)
        
        # Determine the priority map from depth and inversed flow velocity
        flpl_velocity = pcr.ifthen(self.r.lulc.floodplain,
                                   self.r.hydro.velocity)
        flpl_velocity_inv = pcr.mapmaximum(flpl_velocity) - flpl_velocity
        depth_velocity = depth_detail * flpl_velocity_inv
        
        # Apply a mask if present
        if isinstance(mask, pcr._pcraster.Field) == True:
            depth_velocity = pcr.ifthen(mask, depth_velocity)
        
        # Select regions with water depth times inv_velocity below percentile
        exceeded_area = percentile_slicing(depth_velocity, percentile)
        lowering_area = pcr.ifthen(exceeded_area == pcr.boolean(0),
                                   pcr.boolean(1))
        
        return lowering_area

    def lowering_dem(self, ref_water_level, area):
        """
        Terrain height at the loweringproject area.
        """
        ref_height = spread_values_from_points(ref_water_level,
                                               self.r.axis.location)
        return pcr.ifthen(area, ref_height)
    
    def lowering_measure(self, settings, mask=None, ID='dummy-ID'):
        """
        Create a Measure class for floodplain lowering.
        """
        self.settings = settings
        self.settings['msr_type'] = 'lowering'
        self.settings['ID'] = ID
        msr_settings = self.settings.copy()
        percentile = self.settings['lowering_percentage']
        trachytope_nr = self.settings['lowering_trachytope']
        area = self.lowering_area(percentile, mask=mask)
        dem = self.lowering_dem(self.r.hydro.wl_exc50d, area)
#        pcr.aguila(dem)
        dem = pcr.ifthen(dem < self.r.geom.dem, dem)
#        pcr.aguila(dem)
        area = pcr.ifthen(pcr.defined(dem), area)
        ecotopes = assign_ecotopes(area, 
                                   self.settings['lowering_ecotope'],
                                   self.r.lulc.ecotopes.legend_df)
        trachytopes = pcr.ifthen(area, pcr.ordinal(trachytope_nr))
        groyne_height = pcr.ifthen(area, pcr.scalar(-9999))
        minemb_height = pcr.ifthen(area, pcr.scalar(-9999))
        main_dike_height = pcr.scalar(mapIO.empty_map(area))

        measure = Measure(msr_settings, 
                      area, dem, ecotopes, trachytopes,
                      groyne_height, minemb_height, main_dike_height)
        return measure
        
    def reloc_alpha_area(self, alpha_hull, mask=None):
        """
        Determine the dike relocation areas using the alpha shape.
        ----------
        Parameters
        alpha_hull: float in m for alpha shape. 
                    Higher values give larger relocations
        mask: boolean PCRaster map, area of application at boolean True
        
        Returns a boolean map with the relocation area.
        """
        dikes = self.r.main_dike.location
        winbed = self.r.lulc.winter_bed
        if isinstance(mask, pcr._pcraster.Field) == True:
            dikes = pcr.ifthen(mask, dikes)
            winbed = pcr.ifthen(mask, winbed)
        concave_hull = alpha_shape_pcr(dikes, alpha_hull, sparse=winbed)
        extra_winter_bed = pcr.cover(self.r.lulc.winter_bed, pcr.boolean(0)) ^\
                           pcr.boolean(concave_hull)
        return pcr.ifthen(extra_winter_bed, extra_winter_bed)
    
    def reloc_dem(self, area):
        """
        Determine the DEM at the new floodplain area.
        """
        pre_dem = pcr.ifthen(area, self.r.geom.dem)
        clumps = pcr.clump(area)
        average_dem = pcr.areaaverage(pre_dem, clumps)
        outliers = percentile_slicing(pre_dem, 70)
        post_dem = pcr.ifthenelse(outliers, average_dem, pre_dem)
        return post_dem
        
    def reloc_alpha_measure(self, settings, mask=None, ID='dummy-ID'):
        """
        Create a Measure class for dike relocation using alpha shapes.
        """
        self.settings = settings
        self.settings['msr_type'] = 'relocation'
        self.settings['ID'] = ID
        msr_settings = self.settings.copy()
        alpha = self.settings['relocation_alpha']
        trachytope_nr = self.settings['relocation_trachytope']
        area = self.reloc_alpha_area(alpha, mask=mask)
        real_estate_smooth = pcr.windowaverage(self.r.lulc.real_estate_value, 250) 
        real_estate_bool = pcr.ifthenelse(real_estate_smooth > 3e5,
                                          pcr.boolean(1), pcr.boolean(0))
        area = pcr.ifthen(area ^ real_estate_bool, pcr.boolean(1))
        dem = self.reloc_dem(area)
        ecotopes = assign_ecotopes(area, 
                                   self.settings['relocation_ecotope'],
                                   self.r.lulc.ecotopes.legend_df)
        trachytopes = pcr.ifthen(area, pcr.ordinal(trachytope_nr))
        groyne_height = pcr.scalar(mapIO.empty_map(area))
        minemb_height = pcr.scalar(mapIO.empty_map(area))
        main_dike_height = pcr.scalar(mapIO.empty_map(area))

        measure = Measure(msr_settings, 
                      area, dem, ecotopes, trachytopes,
                      groyne_height, minemb_height, main_dike_height)
        return measure    
    
    def smoothing_area(self, percentile, mask=None):
        """
        Designate areas with high potential for vegetation roughness smoothing.
        ----------
        Parameters
        percentile: 0-100, low value is most effecient locations
        mask: boolean PCRaster map, area of application at boolean True
        
        Returns a boolean map with the optimal area for roughness smoothing. 
        """
        #-determine floodplain roughness and multiply with specific discharge
        perc = 100 - percentile
        flpl_roughness = pcr.ifthen(self.r.lulc.main_channel == pcr.boolean(0),
                                    self.r.hydro.nikuradse)
        flpl_roughness = pcr.ifthen(self.r.lulc.backwaters == pcr.boolean(0),
                                    flpl_roughness)
        ks_q = flpl_roughness * self.r.hydro.specific_q
        
        # Apply a mask if present
        if isinstance(mask, pcr._pcraster.Field) == True:
            ks_q = pcr.ifthen(mask, ks_q)
        
        rough_area = percentile_slicing(ks_q, perc)
        return pcr.ifthen(rough_area, rough_area)

    def smoothing_measure(self, settings, mask=None, ID='dummy-ID'):
        """
        Create a Measure class for floodplain lowering.
        """
        self.settings = settings
        self.settings['msr_type'] = 'smoothing'
        self.settings['ID'] = ID
        percentile = self.settings['smoothing_percentage']
        trachytope_nr = self.settings['smoothing_trachytope']
        area = self.smoothing_area(percentile, mask=mask)
        dem = pcr.scalar(mapIO.empty_map(area))
        ecotopes = assign_ecotopes(area, 
                                   self.settings['smoothing_ecotope'],
                                   self.r.lulc.ecotopes.legend_df)
        trachytopes = pcr.ifthen(area, pcr.ordinal(trachytope_nr))
        groyne_height = pcr.scalar(mapIO.empty_map(area))
        minor_emb_height = pcr.scalar(mapIO.empty_map(area))
        main_emb_height = pcr.scalar(mapIO.empty_map(area))

        measure = Measure(self.settings, 
                      area, dem, ecotopes, trachytopes,
                      groyne_height, minor_emb_height, main_emb_height)
        return measure
    
    def side_channel_friction(self):
        """
        Create the friction map for positioning of the side channel.
        
        High friction values are assigned to unsuitable areas for new channels:
            - distance to the major embankment: high friction close to the dike
            - distance to the main channel: high friction in the main channel
            - groyne field, diminishing further away from the channel
            - floodplain lakes: low friction at floodplain lakes
        Friction values are tweaked to prevent highly sinuous channels. 
        
        Returns a global friction map over the whole river section.
        """
        # Determine global friction
        dike_toe_friction = pcr.ifthen(self.r.geom.dist_to_main_dike< 200,
                                       pcr.scalar(15))
        channel_friction = pcr.ifthen(
                         self.r.lulc.main_channel | self.r.lulc.groyne_field,
                         pcr.scalar(25))
        dist_to_edge = pcr.min(self.r.geom.dist_to_main_dike,
                               self.r.geom.dist_to_main_channel)
        dist_to_edge = pcr.ifthen(self.r.lulc.winter_bed, dist_to_edge)
        edges_friction = pcr.scalar(12) -\
                  12 * dist_to_edge / (pcr.max(1,pcr.mapmaximum(dist_to_edge)))
        backwater_friction = pcr.ifthen(pcr.boolean(self.r.lulc.backwaters),
                                        edges_friction / 3)
        friction = pcr.cover(dike_toe_friction, channel_friction,
                             backwater_friction, edges_friction)
        friction = pcr.ifthen(self.r.lulc.winter_bed, friction)
        return friction

    def side_channel_positioning(self,  friction, section):
        """
        Position a single channel side channel on the floodplain section.

        friction = low friction for suitable locations of the channel
        section = boolean map with floodplain extent
        """
        
        #-determine start and end point of the channel
        section = pcr.cover(section, pcr.boolean(0))
        section_XL = pcr.ifthen(pcr.spreadmaxzone(section, 1,1,300),
                               pcr.boolean(1))
        section_XL = pcr.ifthen(self.r.lulc.winter_bed, section_XL)
        channel_mask   = pcr.ifthen(self.r.lulc.winter_bed == 1,
                                    pcr.boolean(1))
        n_coor_axis = pcr.ifthen(pcr.defined(self.r.axis.location),
                                 self.r.mesh.n_coor)
        start_point = pcr.ifthen(channel_mask, \
         pcr.cover(n_coor_axis == pcr.areaminimum(n_coor_axis, section_XL), 0))
        end_point = pcr.ifthen(channel_mask, \
         pcr.cover(n_coor_axis == pcr.areamaximum(n_coor_axis, section_XL), 0))
        
        #-determine location of the channel
        # determine channel as the shortest path over the friction field
        # do not allow routing over the main channel, except close to endpoints
        pcr.setglobaloption('lddin')
        end_points = start_point | end_point
        end_points_XL = pcr.boolean(
                        pcr.windowmaximum(pcr.ordinal(end_points), 700))
        mask = end_points_XL | pcr.cover(section, pcr.boolean(0))
        dist = pcr.ifthen(mask, pcr.spreadmax(start_point, 0, friction, 1e6))
        ldd = pcr.lddcreate(dist, 1e15,1e15,1e15,1e15)
        channel_center = pcr.path(ldd, end_point)
        
        #-mask out the channel center line at the upstream end to decouple from
        # the main channel
        channel_dist = pcr.ifthen(channel_center, dist)
        chan_dist_ave = pcr.areaaverage(channel_dist, section_XL)
        channel_upstream = pcr.ifthen(channel_dist <= chan_dist_ave,
                                      channel_center)
        chan_width= pcr.areaaverage(self.r.geom.main_channel_width, section_XL)
        channel_upstream = pcr.ifthen(
                                 self.r.geom.dist_to_main_channel > chan_width,
                                 channel_upstream)
        channel_downstream = pcr.ifthen(channel_dist > chan_dist_ave,
                                        channel_center)
        channel  = pcr.cover(channel_downstream, pcr.boolean(0)) |\
                   pcr.cover(channel_upstream, pcr.boolean(0))
        channel = channel == True
        
        db = False
        if db == True:
            pcr.report(ldd, 'ldd.map')
            pcr.report(channel_center, 'path.map')
            pcr.report(channel_dist, 'chanDist.map')
            pcr.report(channel_upstream, 'chanUp.map')
            pcr.report(channel, 'channel.map')
            pcr.report(section_XL, 'fpsXL.map')
            pcr.report(section, 'fps.map')
            pcr.report(dist, 'dist.map')
            pcr.report(ldd,  'ldd.map')
            pcr.report(start_point, 'sp.map')
            pcr.report(end_point, 'ep.map')
            pcr.report(friction, 'friction.map')
          
        return channel
    
    def channel_center_lines(self, friction, mask=None):
        """
        Determine the center lines of side channels within the mask area.
        
        Position side channels on all large and wide floodplain sections
        sections that currently do not have a side channel. Default friction 
        is based on friction function, but this can be an external PCRaster 
        map as well. 
        """
        #-select large wide floodplain sections only
        large_sections = pcr.ifthen(
                               pcr.areaarea(self.r.geom.flpl_wide) > 1e6,
                               self.r.geom.flpl_wide)
        
        #-select floodplain sections without side channels currently present
        # based on roughness code 105, but omitting groyne fields (langsdammen)
        existing_channels = self.r.lulc.floodplain &\
                            (self.r.lulc.trachytopes == 105)
        channel_cell_area = pcr.cover(
                            pcr.ifthen(existing_channels, pcr.cellarea()),
                            0)
        selected_sections = pcr.ifthen(
                        pcr.areatotal(channel_cell_area, large_sections) < 1e5,
                        large_sections)
        if isinstance(mask, pcr._pcraster.Field) == True:
            selected_sections = pcr.ifthen(mask, selected_sections)
        
        # Loop over selected sections to position side channels
        flpl_IDs  = list(np.unique(pcr.pcr2numpy(selected_sections, -9999))[1:])
        center_lines = pcr.nominal(mapIO.emptyMap(self.r.geom.dem))
        
        print(flpl_IDs)
        
        for ID in flpl_IDs[:]:
        #~ for ID in flpl_IDs[:][0:2]:
            print(ID)
            flpl_section = pcr.ifthen(pcr.scalar(self.r.geom.flpl_wide) == float(ID), pcr.boolean(1))
            channel = self.side_channel_positioning(friction, flpl_section)
            #~ pcr.aguila(channel)
            center_lines = pcr.cover(center_lines, pcr.ifthen(channel, pcr.nominal(float(ID))))
            
        return center_lines
    
    def side_channel_dem(self, center_lines):
        """
        Terrain height and bathymetry of the side channel project area.
        """
        ms = self.settings
        
        # Determine the side channel bathymetry using a trapezoidal
        # cross sectional shape relative to the reference height.
        ref_height = spread_values_from_points(self.r.hydro.wl_exc363d,
                                               self.r.axis.location)
        dist_to_center = pcr.spread(pcr.cover(center_lines, 0), 0,1) # in m
        
        #-determine the reference depth of the side channel, which is based
        # on the water levels that are exceeded 363 days/year
        chan_depth = ref_height - ms['channel_depth']
        chan_shore_depth = chan_depth + ms['channel_slope'] *\
                           (dist_to_center - ms['channel_width'] / 2)
        chan_dem = pcr.ifthenelse(dist_to_center <= ms['channel_width'] / 2,
                            pcr.scalar(chan_depth), 
                            chan_shore_depth)
        chan_dem = pcr.ifthen(chan_dem < self.r.geom.dem, chan_dem)
        return chan_dem
    
    def side_channel_area(self, dem):
        """Determine the area of the side channel areas based on the dem.
        ----------
        Parameters
        dem: PCRaster map with terrain height where digging is required. 
        
        Returns a boolean map with the side channel area.
        """
        return pcr.ifthen(pcr.defined(dem), pcr.boolean(1))
    
    def side_channel_ecotopes(self, area):
        """
        Assign ecotopes to the side channel area.
        """
        eco_string = self.settings['channel_ecotope']
        eco_legend = self.r.lulc.ecotopes.legend_df
        ecotopes = assign_ecotopes(area, eco_string, eco_legend)
        return ecotopes
    
    def side_channel_trachytopes(self, area):
        """
        Assign side channel trachytopes.
        """
        trachytope_nr = self.settings['channel_trachytope']
        return pcr.ifthen(area, pcr.ordinal(trachytope_nr))
    
    def side_channel_measure(self, settings, mask=None, ID='dummy-ID'):
        """
        Create a measure class with all required attributes.
        """
        self.settings = settings
        self.settings['msr_type'] = 'sidechannel'
        self.settings['ID'] = ID
        msr_settings = self.settings.copy()
        friction = self.side_channel_friction()
        center_lines = self.channel_center_lines(friction, mask=mask)
        dem = self.side_channel_dem(center_lines)
        area = self.side_channel_area(dem)
        ecotopes = self.side_channel_ecotopes(area)
        trachytopes = self.side_channel_trachytopes(area)
        groyne_height = pcr.scalar(mapIO.empty_map(area))
        minemb_height = pcr.ifthen(area, pcr.scalar(-9999))
        main_dike_height = pcr.scalar(mapIO.empty_map(area))

        measure = Measure(msr_settings,
                          area, dem, ecotopes, trachytopes,
                          groyne_height, minemb_height, main_dike_height)
        return measure
           
    def groyne_lowering_area(self,  mask=None):
        """
        Area where groyne lowering is applied.
        To do: prioritize based on hydrodynamics?
        """
        area = self.r.groynes.location
        if isinstance(mask, pcr._pcraster.Field) == True:
            area = pcr.ifthen(mask, area)
        return area
    
    def groyne_lowering_msr(self, settings, mask=None,  ID='dummy-ID'):
        """
        """
        self.settings = settings
        self.settings['msr_type'] = 'groynelowering'
        self.settings['ID'] = ID
        msr_settings = self.settings.copy()
        ref_water_level_key = self.settings['groyne_ref_level']
        ref_water_level = self.r.hydro.__dict__[ref_water_level_key]
        
        area = self.groyne_lowering_area(mask=mask)
        dem = pcr.scalar(mapIO.emptyMap(area))
        ecotopes = LegendMap(pcr.nominal(mapIO.emptyMap(area)),
                             self.r.lulc.ecotopes.legend_df)
        trachytopes = pcr.nominal(mapIO.emptyMap(area))
        groyne_height = self.lowering_dem(ref_water_level, area)
        minemb_height = pcr.scalar(mapIO.emptyMap(area))
        main_dike_height = pcr.scalar(mapIO.emptyMap(area))
        
        measure = Measure(msr_settings,
                      area, dem, ecotopes, trachytopes,
                      groyne_height, minemb_height, main_dike_height)
        return measure
    
    def minemb_lowering_area(self, mask):
        """"
        Area where minor embankment lowering is applied.
        To do:  prioritize based on hydrodynamics?
        """
        area = self.r.minemb.location
        if isinstance(mask, pcr._pcraster.Field) == True:
            area = pcr.ifthen(mask, area)
        return area

    def minemb_lowering_msr(self, settings, mask=None,  ID='dummy-ID'):
        """
        Create a Measure object representing minor embankment lowering.
        """
        self.settings = settings
        self.settings['msr_type'] = 'minemblowering'
        self.settings['ID'] = ID
        msr_settings = self.settings.copy()
        ref_water_level_key = self.settings['minemb_ref_level']
        ref_water_level = self.r.hydro.__dict__[ref_water_level_key]
        
        area = self.minemb_lowering_area(mask=mask)
        dem = pcr.scalar(mapIO.emptyMap(area))
        ecotopes = LegendMap(pcr.nominal(mapIO.emptyMap(area)),
                             self.r.lulc.ecotopes.legend_df)
        trachytopes = pcr.nominal(mapIO.emptyMap(area))
        groyne_height = pcr.scalar(mapIO.emptyMap(area))
        minemb_height = self.lowering_dem(ref_water_level, area)
        main_dike_height = pcr.scalar(mapIO.emptyMap(area))
        
        measure = Measure(msr_settings,
                      area, dem, ecotopes, trachytopes,
                      groyne_height, minemb_height, main_dike_height)
        return measure
    
    def main_dike_raising_msr(self, settings, mask=None, ID='dummy-ID'):
        """
        Create a Measure object representing dike raising.
        """
        self.settings = settings
        self.settings['msr_type'] = 'dikeraising'
        self.settings['ID'] = ID
        msr_settings = self.settings.copy()
        area = pcr.boolean(self.r.main_dike.location)
        if isinstance(mask, pcr._pcraster.Field) == True:
            area = pcr.ifthen(mask, area)
        dem = pcr.scalar(mapIO.emptyMap(area))
        ecotopes = LegendMap(pcr.nominal(mapIO.emptyMap(area)),
                             self.r.lulc.ecotopes.legend_df)
        trachytopes = pcr.nominal(mapIO.emptyMap(area))
        groyne_height = pcr.scalar(mapIO.emptyMap(area))
        minemb_height = pcr.scalar(mapIO.emptyMap(area))
        main_dike_height = self.r.main_dike.height + settings['main_dike_dh']
        main_dike_height = pcr.ifthen(mask, main_dike_height)
        
        measure = Measure(msr_settings,
                      area, dem, ecotopes, trachytopes,
                      groyne_height, minemb_height, main_dike_height)
        return measure        
    

#%% Main
if __name__ == "__main__":
        # Initialize the study area
    start_time = time.time()
    output_dir = 'D:/projecten/RiverScapeWaal2/output/waal_XL'
    input_dir = 'D:/projecten/RiverScapeWaal2/input'

    #~ restraints_dir = os.path.join(input_dir, 'restraints') - NOT NEEDED

    scratch_dir  = 'D:/projecten/RiverScapeWaal2/scratch'
    pcr.setglobaloption('unittrue')
    os.chdir(scratch_dir)
    
    #-define input
    settings_smooth = OrderedDict([
                        ('smoothing_percentage', 100),
                        ('smoothing_ecotope', 'UG-2'),
                        ('smoothing_trachytope', 1201),
                        
                        ('lowering_percentage', 100),
                        ('lowering_ecotope', 'UG-2'),
                        ('lowering_trachytope', 1201),
                        ('lowering_height', 'water_level_50d'),
                        
                        ('channel_width', 75),
                        ('channel_depth', 2.5),
                        ('channel_slope', 1./3.),
                        ('channel_ecotope', 'RnM'),
                        ('channel_trachytope', 105),
                        
                        ('relocation_alpha', 10000),
                        ('relocation_depth', 'AHN'),
                        ('relocation_ecotope', 'HG-2'),
                        ('relocation_trachytope', 1201),
                        
                        ('groyne_ref_level', 'wl_exc150d'),
                        ('minemb_ref_level', 'wl_exc50d'),
                        ('main_dike_dh', 0.50),
                        ])
    settings_natural = OrderedDict([
                        ('smoothing_percentage', 100),
                        ('smoothing_ecotope', 'UG-1'),
                        ('smoothing_trachytope', 1202),
                        
                        ('lowering_percentage', 100),
                        ('lowering_ecotope', 'UG-1'),
                        ('lowering_trachytope', 1202),
                        ('lowering_height', 'water_level_50d'),
                        
                        ('channel_width', 50),
                        ('channel_depth', 1),
                        ('channel_slope', 1./7.),
                        ('channel_ecotope', 'RnM'),
                        ('channel_trachytope', 105),
                        
                        ('relocation_alpha', 10000),
                        ('relocation_depth', 'AHN'),
                        ('relocation_ecotope', 'HG-1'),
                        ('relocation_trachytope', 1202),
                        
                        ('groyne_ref_level', 'wl_exc150d'),
                        ('minemb_ref_level', 'wl_exc50d'),
                        ('main_dike_dh', 0.50),
                        ])
        
    #-Waal XL
    current_dir = os.path.join(input_dir, 'reference_maps')
    pcr.setclone(os.path.join(current_dir, 'clone.map'))
    
    #- Import data
    main_dike = read_dike_maps(current_dir, 'main_dike')
    minemb = read_dike_maps(current_dir, 'minemb')
    groynes = read_dike_maps(current_dir, 'groyne')
    hydro = read_hydro_maps(current_dir)
    mesh = read_mesh_maps(current_dir)
    axis = read_axis_maps(current_dir)
    lulc = read_lulc_maps(current_dir)
    geom = read_geom_maps(current_dir)
        
    #- initiate the model
    waal = River('Waal', axis, main_dike, minemb, groynes, hydro, 
                 mesh, lulc, geom)
    waal_msr = RiverMeasures(waal)
    waal_msr.settings = settings_smooth
    
#    # Test measures
#    test_measures = True
#    if test_measures == True:
##        mask = waal.axis.rkm_full < 900
#        mask = pcr.boolean(1)
#        ID = 'everywhere'
#        lowering_msr = waal_msr.lowering_measure(settings, mask=mask, ID=ID)
#        groyne_low_msr = waal_msr.groyne_lowering_msr(settings, mask=mask, ID=ID)
#        minemb_low_msr = waal_msr.minemb_lowering_msr(settings, mask=mask, ID=ID)
#        main_dike_raise_msr = waal_msr.main_dike_raising_msr(settings, mask=mask, ID=ID)
#        chan_msr = waal_msr.side_channel_measure(settings, mask=mask, ID=ID)
#        smooth_msr = waal_msr.smoothing_measure(settings, mask=mask, ID=ID)
##        reloc_msr = waal_msr.reloc_alpha_measure(settings, mask=mask)
#        msr_list = [groyne_low_msr, minemb_low_msr,
#                    main_dike_raise_msr, lowering_msr, chan_msr, smooth_msr]
#        msr_root_dir = os.path.join(output_dir, 'measures_ensemble03/maps')
#        pcrRecipes.make_dir(msr_root_dir)
#        for msr in msr_list:
#            write_measure(msr, msr_root_dir)
    

