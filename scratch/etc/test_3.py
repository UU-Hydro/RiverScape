#!/usr/bin/env python
# coding: utf-8

# In[140]:


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


# In[141]:


import pcraster as pcr


# In[142]:


import mapIO


# In[143]:


import pcrRecipes


# In[144]:


import measures_py3 as msr


# In[145]:


start_time = time.time()


# In[146]:


output_dir = "/scratch/depfg/sutan101/tmp_menno/out/"
input_dir = "/scratch/depfg/hydrowld/river_scape/source/from_menno/riverscape/input/"
scratch_dir  = "/scratch/depfg/sutan101/tmp_menno/tmp/"


# In[147]:


pcr.setglobaloption('unittrue')


# In[148]:


pcrRecipes.make_dir(scratch_dir)
os.chdir(scratch_dir)


# In[149]:


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


# In[150]:



# In[151]:


current_dir = os.path.join(input_dir, 'reference_maps')
pcr.setclone(os.path.join(current_dir, 'clone.map'))
print(current_dir)


# In[152]:


print(input_dir)
print(os.path.join(input_dir, 'reference_maps'))
print(current_dir)


# In[153]:


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





# In[154]:


pcr.aguila(os.path.join(current_dir, 'clone.map'))


# In[155]:


pcr.aguila(geom.dem)


# In[156]:


#- initiate the model
waal = msr.River('Waal', axis, main_dike, minemb, groynes, hydro, 
             mesh, lulc, geom)
waal_msr = msr.RiverMeasures(waal)
waal_msr.settings = settings


# In[157]:


# Test measures
mask = pcr.boolean(1)
ID = 'everywhere'


# In[158]:


lowering_msr = waal_msr.lowering_measure(settings, mask=mask, ID=ID)


# In[159]:


lowering_msr.plot()


# In[160]:



os.system("killall aguila")


# In[161]:


groyne_low_msr = waal_msr.groyne_lowering_msr(settings, mask=mask, ID=ID)



# In[162]:


minemb_low_msr = waal_msr.minemb_lowering_msr(settings, mask=mask, ID=ID)


# In[163]:


main_dike_raise_msr = waal_msr.main_dike_raising_msr(settings, mask=mask, ID=ID)


# In[164]:


large_sections = pcr.ifthen(pcr.areaarea(waal.geom.flpl_wide) > 1e6, waal.geom.flpl_wide)

flpl_section = pcr.ifthen(large_sections == 1, pcr.nominal(1))
pcr.aguila(flpl_section) 


# In[165]:


test = list(np.unique(pcr.pcr2numpy(large_sections, -9999))[1:])
test[:]
for ID in test[:]:
    print(ID)


# In[166]:


chan_msr = waal_msr.side_channel_measure(settings, mask=mask, ID=ID)


# In[ ]:


print(ID
     )


# In[ ]:




# In[ ]:




os.system("killall aguila")
