import os
import glob
import numpy as np
from scipy import optimize
from collections import OrderedDict
import datetime 
#-import spatial packages
import fiona
import pcraster as pcr
import shapely
from shapely.geometry import Polygon, Point
import geopandas as gpd
import pandas as pd
import netCDF4

#import RiverScape modules
import mapIO
import pcrRecipes


#%% Initialize the study area
# Test data extraction from baseline and WAQUA
root_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(root_dir, 'input')
output_dir = os.path.join(root_dir, 'output')
scratch_dir  = os.path.join(root_dir, 'scratch')
os.chdir(scratch_dir)
pcr.setglobaloption('unittrue')

#-Waal full extent ############################################################
river_extent = [120000, 201000, 420000, 436300] # whole Waal
xmin, xmax, ymin, ymax = river_extent
cell_length = 25
nr_cols = (xmax - xmin) / cell_length
nr_rows = (ymax - ymin) / cell_length
clone = mapIO.make_clone(nr_rows, nr_cols, cell_length, xmin, ymax)
clone_file = os.path.join(output_dir, output_dir, 'clone.map')
pcr.report(clone, clone_file)


#%% Extract restraints data
restraints_GDB = os.path.join(input_dir, 'restraints\\restraints.gdb')

# Note vector2raster werkt niet met Nr_Gerecht, omdat de ID-cijfers groter zijn
# dan 2.1 e9. Daardoor wisslen de laatste drie getallen van de cijfers.
# Alternatief is stringVector2raster
#owner_nr = mapIO.vector2raster('FileGDB', clone_file,
#                                             restraints_GDB, 'Nr_Gerecht',
#                                             layer = 'owners_waalXL')
owner_nr = mapIO.stringVector2raster('export_output.shp', 'Naam_Gerec', 
                                      clone_file, 'naam_gerec.map')
pcr.report(owner_nr, 'owner_nr.map')
ppp
veg_class = pcr.nominal(mapIO.vector2raster('FileGDB', clone_file,
                                             restraints_GDB, 'veg_class',
                                             layer = 'veglegger'))
flow_path = pcr.nominal(mapIO.vector2raster('FileGDB', clone_file,
                                             restraints_GDB, 'flow_path',
                                             layer = 'veglegger'))
hwat_free = mapIO.vector2raster('FileGDB', clone_file,
                                             restraints_GDB, 'SHAPE_Area',
                                             layer = 'hoogwatervrij_vlakken')
building_area = mapIO.vector2raster('FileGDB', clone_file,
                                    restraints_GDB, 'SHAPE_Area',
                                    layer = 'buildings_pol_cadastre')

directives = pcr.nominal(mapIO.vector2raster('FileGDB', clone_file,
                                             restraints_GDB, 'VHN_NEW',
                                             layer = 'natura2000'))
# 1 Birds Dir, 2: Habitat. Dir., 3: HabDir or BirdDir

waste_area = pcr.boolean(mapIO.vector2raster('FileGDB', clone_file,
                                             restraints_GDB, 'CATEGORIE',
                                             layer = 'stortplaatsen'))
owner_type = pcr.nominal(mapIO.vector2raster('FileGDB', clone_file,
                                             restraints_GDB, 'owner_type',
                                             layer = 'owners_waalXL'))
outline = mapIO.vector2raster('FileGDB', clone_file,
                               restraints_GDB, 'diss',
                               layer = 'outline')

pcr.report(veg_class, 'veg_class.map')
pcr.report(flow_path, 'flow_path.map')
pcr.report(hwat_free, 'hwat_free.map')
pcr.report(building_area, 'building_area.map')
pcr.report(directives, 'directives.map')
pcr.report(waste_area, 'waste_area.map')
pcr.report(owner_type, 'owner_type.map')
pcr.report(outline, 'outline.map')
#%% Reclass restraints based on lookup tables
import uuid
def lookup_scalar(pcr_map, column_data):
    lut_file = 'lut_reclass_%s.txt' % uuid.uuid4() 
        # needs a unique file name due to bug in pcr.lookupnominal
    np.savetxt(lut_file, column_data, fmt = '%.0f')
    recoded = pcr.lookupscalar(lut_file, pcr.scalar(pcr_map))
    os.remove(lut_file)
    return recoded

data = {'nr': [1,2,3,4,5,6,7],
        'label': ["Bos", "Gras en Akker", "Riet en Ruigte",
                  "Struweel", "Verhard oppervlak", "Water", 'Null'],
        'friction': [1,5,4,3,3,0,0]}
veg_lut = pd.DataFrame(data=data)

owner_types = ['state', 'waterboard', 'municipality', 'company', 'citizens',\
               'state_forest', 'province', 'prvc_nature', 'foundation', 
               'sand_clay', 'other']
owner_label = ['rws', 'wtb', 'mun', 'com', 'cit',
               'sbb', 'prv', 'prn', 'fdt',
               'sgc', 'oth']
data={'owner_type': [1, 2, 3, 4,   5, 6,   7, 8,   9,   10, 11],
      'types'    : owner_types,
      'label'   : owner_label,
      'friction'  : [1, 1, 1, 2, 1.5, 2, 1.3, 1.5, 2,  2.5,  1]}
owners_lut = pd.DataFrame(data=data)


# succession restrictions
veg_friction = lookup_scalar(veg_class, veg_lut[['nr', 'friction']])
flow_path_friction = pcr.ifthenelse(flow_path == 1, pcr.scalar(3), pcr.scalar(1))

# implementation restrictions
hwat_free_friction = 10 * pcr.scalar(pcr.boolean(hwat_free))
buildings_friction = 20 * pcr.scalar(pcr.boolean(building_area))
directives_friction = 5 * pcr.ifthenelse(pcr.scalar(directives) < 2.5, 
                                         pcr.scalar(1), pcr.scalar(2))
waste_friction = 20 * pcr.scalar(waste_area)
owner_friction = lookup_scalar(owner_type, owners_lut[['owner_type', 'friction']])

msr_friction = pcr.cover(hwat_free_friction, 0) +\
               pcr.cover(buildings_friction, 0) +\
               pcr.cover(waste_friction, 0) +\
               pcr.cover(directives_friction, 0) +\
               pcr.cover(owner_friction, 0)
msr_friction = pcr.ifthen(pcr.defined(outline), msr_friction)

pcr.report(veg_friction, 'veg_friction.map')
pcr.report(flow_path_friction, 'flow_path_friction.map')
pcr.report(buildings_friction, 'buildings_friction.map')
pcr.report(directives_friction, 'directives_friction.map')
pcr.report(waste_friction, 'waste_friction.map')
pcr.report(owner_friction, 'owner_friction.map')
pcr.report(msr_friction, 'msr_friction.map')
pcr.aguila(msr_friction)


#%%


































