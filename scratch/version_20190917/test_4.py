#!/usr/bin/env python
# coding: utf-8

# In[75]:


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


# In[76]:


import pcraster as pcr


# In[77]:


import mapIO


# In[78]:


import pcrRecipes


# In[79]:


import measures_py3 as msr


# In[80]:


start_time = time.time()


# In[81]:


output_dir = "/scratch/depfg/sutan101/tmp_menno/out/"
input_dir = "/scratch/depfg/hydrowld/river_scape/source/from_menno/riverscape/input/"
scratch_dir  = "/scratch/depfg/sutan101/tmp_menno/tmp/"


# In[82]:


pcr.setglobaloption('unittrue')


# In[83]:


pcrRecipes.make_dir(scratch_dir)
os.chdir(scratch_dir)


# In[84]:


#-define input
settings = OrderedDict([
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


# In[85]:


current_dir = os.path.join(input_dir, 'reference_maps')
pcr.setclone(os.path.join(current_dir, 'clone.map'))
print(current_dir)


# In[86]:


print(input_dir)
print(os.path.join(input_dir, 'reference_maps'))
print(current_dir)


# In[87]:


#- Import data
main_dike = msr.read_dike_maps(current_dir, 'main_dike')
minemb = msr.read_dike_maps(current_dir, 'minemb')
groynes = msr.read_dike_maps(current_dir, 'groyne')
hydro = msr.read_hydro_maps(current_dir)
mesh = msr.read_mesh_maps(current_dir)
axis = msr.read_axis_maps(current_dir)
lulc = msr.read_lulc_maps(current_dir)
geom = msr.read_geom_maps(current_dir)


# In[ ]:





# In[ ]:





# In[88]:


pcr.aguila(os.path.join(current_dir, 'clone.map'))


# In[89]:


pcr.aguila(geom.dem)


# In[90]:


#- initiate the model
waal = msr.River('Waal', axis, main_dike, minemb, groynes, hydro, 
             mesh, lulc, geom)
waal_msr = msr.RiverMeasures(waal)
waal_msr.settings = settings


# In[91]:


# Test measures
mask = pcr.boolean(1)
ID = 'everywhere'


# In[92]:


lowering_msr = waal_msr.lowering_measure(settings, mask=mask, ID=ID)


# In[93]:


lowering_msr.plot()


# In[94]:



os.system("killall aguila")


# In[95]:


groyne_low_msr = waal_msr.groyne_lowering_msr(settings, mask=mask, ID=ID)



# In[96]:


minemb_low_msr = waal_msr.minemb_lowering_msr(settings, mask=mask, ID=ID)


# In[97]:


main_dike_raise_msr = waal_msr.main_dike_raising_msr(settings, mask=mask, ID=ID)


# In[98]:


large_sections = pcr.ifthen(pcr.areaarea(waal.geom.flpl_wide) > 1e6, waal.geom.flpl_wide)

flpl_section = pcr.ifthen(large_sections == 1, pcr.nominal(1))
pcr.aguila(flpl_section) 


# In[99]:


test = list(np.unique(pcr.pcr2numpy(large_sections, -9999))[1:])
test[:]
for ID in test[:]:
    print(ID)


# In[100]:


chan_msr = waal_msr.side_channel_measure(settings, mask=mask, ID=ID)


# In[101]:


os.system("killall aguila")


# In[102]:


smooth_msr = waal_msr.smoothing_measure(settings, mask=mask, ID=ID)


# In[103]:


msr_list = [groyne_low_msr, minemb_low_msr,
            main_dike_raise_msr, lowering_msr, chan_msr, smooth_msr]


# In[104]:


msr_root_dir = os.path.join(output_dir, 'measures_ensemble03/maps')


# In[105]:


pcrRecipes.make_dir(msr_root_dir)


# In[106]:


from measures_py3 import *


# In[107]:


for measure in msr_list:
    write_measure(measure, msr_root_dir)


# In[ ]:




