# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 17:40:28 2017

@author: straa005
"""

import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import scipy
import netCDF4
import string
import os
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
import mpld3
from collections import OrderedDict
sns.set_style("ticks")
import mapIO
import pcraster as pcr
import measures
import biosafe
import biosafeIO as bsIO
import evaluate_cost as ec


#%% Initial settings
root_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(root_dir, 'input')
ref_map_dir = os.path.join(input_dir, 'reference_maps')
bio_dir = os.path.join(input_dir, 'bio')
cost_dir = os.path.join(input_dir, 'cost')
restraints_dir = os.path.join(input_dir, 'restraints')

output_dir = os.path.join(root_dir, 'output/waal_XL')
ens_dir = os.path.join(output_dir, 'measures_ensemble04')
ens_map_dir = os.path.join(ens_dir, 'maps')
ens_FM_dir = os.path.join(ens_dir, 'hydro')
ens_overview_dir = os.path.join(ens_dir, 'overview')

scratch_dir = os.path.join(root_dir, 'scratch')
clone_file = os.path.join(ref_map_dir, 'clone.map')
pcr.setclone(clone_file)
pcr.setglobaloption('unittrue')
os.chdir(scratch_dir)

#%% Figure attributes per river km
rkm = pcr.readmap(os.path.join(ref_map_dir, 'rkm_full.map'))
sections = pcr.readmap(os.path.join(ref_map_dir, 'sections.map'))
flpl_sections = pcr.readmap(os.path.join(bio_dir, 'flpl_sections.map'))
river_sides = pcr.readmap(os.path.join(ref_map_dir, 'river_sides.map'))
trachytopes = pcr.readmap(os.path.join(ref_map_dir, 'trachytopes.map'))
c_building = pcr.readmap(os.path.join(cost_dir, 'cost_building_tot.map'))
c_land_acq = pcr.readmap(os.path.join(cost_dir, 'cost_land_acq.map'))
c_roads = pcr.readmap(os.path.join(cost_dir, 'cost_roads.map'))
c_minemb = pcr.readmap(os.path.join(cost_dir, 'cost_minemb_int_use.map'))
c_minemb = pcr.cover(c_minemb, 0)
c_smoothing = pcr.readmap(os.path.join(cost_dir, 'cost_smoothing.map'))
owner_type = pcr.readmap(os.path.join(restraints_dir, 'owner_type.map'))
owner_nr = pcr.readmap(os.path.join(restraints_dir, 'owner_nr.map'))
owner_point_nr = pcr.readmap(os.path.join(restraints_dir, 'owner_point_number.map'))
owner_point_nr = pcr.cover(owner_point_nr, 0)

MV = np.nan
data = {'rkm': pcr.pcr2numpy(rkm, MV).flatten(),
        'sections': pcr.pcr2numpy(sections, MV).flatten(),
        'flpl_sections': pcr.pcr2numpy(flpl_sections, MV).flatten(),
        'river_sides': pcr.pcr2numpy(river_sides, MV).flatten(),
        'trachytopes': pcr.pcr2numpy(trachytopes, MV).flatten(),
        'c_building': pcr.pcr2numpy(c_building, MV).flatten(),
        'c_land_acq': pcr.pcr2numpy(c_land_acq, MV).flatten(),
        'c_roads': pcr.pcr2numpy(c_roads, MV).flatten(),
        'c_minemb': pcr.pcr2numpy(c_minemb, MV).flatten(),
        'c_smoothing': pcr.pcr2numpy(c_smoothing, MV).flatten(),
        'area_ha': 0.0625,
        'area_cell': 1,
        'owner_nr': pcr.pcr2numpy(owner_nr, MV).flatten(),
        'owner_type': pcr.pcr2numpy(owner_type, MV).flatten(),
        'owner_point_nr': pcr.pcr2numpy(owner_point_nr, MV).flatten()}

df = pd.DataFrame(data=data).dropna()
df = df[df.sections > 1]

#%% fraction for Jan; RWS of terrestrial floodplain area
backwater = pcr.readmap(os.path.join(ref_map_dir, 'backwaters.map'))
RWS = owner_type == pcr.nominal(1)
non_main_ch = sections > pcr.scalar(1) #== pcr.scalar(3)
terrestrial = pcr.ifthen(backwater == pcr.boolean(0), pcr.boolean(1))
terrestrial = pcr.ifthen(non_main_ch, terrestrial)
terrestrial_RWS = pcr.ifthen(RWS, terrestrial)
#pcr.aguila(terrestrial, terrestrial_RWS)
#pcr.aguila(pcr.maptotal(pcr.scalar(terrestrial_RWS)) / pcr.maptotal(pcr.scalar(terrestrial)))

#%% Rearrange the data into datafromes with rkm on the index
sub_cols = ['rkm', 'river_sides', 'flpl_sections']
sec = df.loc[:, sub_cols].groupby('flpl_sections').mean()
sec = sec.reset_index()
sec = sec[sec.flpl_sections > 0].sort_values('rkm')
sec['rkm'] = sec.rkm.round().astype(int)
sec['side'] = sec.river_sides.replace({1:'l', 2:'r'})
sec['section'] = sec.apply(lambda x: str(int(x.rkm)) + '-' + x.side, axis=1)
sec = sec[sec.flpl_sections != 110]
sec = sec[sec.flpl_sections != 143]
sec.drop(['rkm', 'river_sides', 'side'], axis=1, inplace=True)
sec.to_csv('sections_left_right.csv')

#-total area
area  = df.loc[:,['flpl_sections','area_ha']].groupby(by='flpl_sections').sum()
sec_area = pd.merge(sec, area.reset_index(), on='flpl_sections')

#-costs
col_cost = ['flpl_sections',
            'c_land_acq', 'c_building', 'c_minemb','c_roads', 'c_smoothing']
sec_cost = df.loc[:,col_cost].groupby(by=['flpl_sections']).sum()
sec_cost = pd.merge(sec, sec_cost.reset_index(), on='flpl_sections')
sec_cost.set_index('section', inplace=True)
sec_cost.drop('flpl_sections', axis=1, inplace=True)
sec_cost = sec_cost * 1e-6 # in M euro

#-trachytopes
col_ttp = ['flpl_sections', 'trachytopes', 'area_ha']
sec_ttp = df.loc[:,col_ttp].groupby(by=['flpl_sections', 'trachytopes']).sum()
sec_ttp = sec_ttp.unstack(level=-1).drop(-2147483648)
sec_ttp.columns = sec_ttp.columns.droplevel(level=0)
large_trachytopes = sec_ttp.sum().sort_values().index.values[-10:][::-1]
sec_ttp = sec_ttp.loc[:, list(large_trachytopes)]
sec_ttp = pd.merge(sec, sec_ttp.reset_index(), on='flpl_sections')
sec_ttp = sec_ttp.set_index(['section']).sort_index()
sec_ttp = sec_ttp.drop(['flpl_sections'], axis=1).fillna(0)

#-nr of owners
col_own = ['flpl_sections', 'owner_nr', 'owner_type']
sec_own = df.loc[:,col_own].groupby(by=['flpl_sections', 'owner_type']).agg(['nunique'])
sec_own.columns = ['owner_nr']
sec_own = sec_own.unstack().fillna(0)
owner_labels = ['rws', 'wtb', 'mun', 'com', 'cit',  'sbb', 'prv', 'prn',
                'fdt', 'sgc', 'oth']
sec_own.columns = owner_labels
sec_own = sec_own.reset_index()
sec_own = pd.merge(sec, sec_own, on='flpl_sections')
sec_own = sec_own.set_index(['section']).sort_index()
sec_own.drop(['flpl_sections'], axis=1, inplace=True)
col_order = ['rws', 'sbb', 'prv', 'prn', 'mun', 'wtb',
             'sgc', 'com', 'fdt', 'oth', 'cit']
sec_own = sec_own.reindex_axis(col_order, axis=1)

#%%-nr of owners based on new map.
gdf = gpd.read_file(os.path.join(restraints_dir, 'owners_winbed_sections_diss.shp'))
gdf = gdf[gdf.FID_flplSe > 0]
gdf['flpl_sections'] = gdf.sections.astype('int')
gdf = pd.merge(gdf, sec, on='flpl_sections')

multi_areas = gdf.groupby(by=['flpl_sections', 'area_m2']).count().reset_index()
multi_areas = multi_areas.loc[:,['flpl_sections', 'area_m2', 'section']]
multi_areas.columns = ['flpl_sections', 'area_m2', 'shared_by']
gdf = pd.merge(gdf, multi_areas, on=['flpl_sections', 'area_m2'])
gdf['area_m2_corr'] = gdf.area_m2 / gdf.shared_by
gdf.to_file('owners_clean.shp')

#%%
grps = gdf.loc[:,['owner_type', 'section', 'area_m2_corr']].groupby(by=['section', 'owner_type'])

sec_own2 = grps.count().unstack().fillna(0)
sec_own2.columns = owner_labels
sec_own2 = sec_own2.reindex_axis(col_order, axis=1)

sec_area2 = grps.sum().unstack().fillna(0)
sec_area2.columns = owner_labels
sec_area2 = 0.0001 * sec_area2.reindex_axis(col_order, axis=1) # area in ha


#%%

#-owner area
col_area = ['flpl_sections', 'owner_type', 'area_ha']
sec_area = df.loc[:,col_area].groupby(by=['flpl_sections', 'owner_type']).sum()
sec_area = sec_area.unstack().fillna(0)
owner_labels = ['rws', 'wtb', 'mun', 'com', 'cit',  'sbb', 'prv', 'prn',
                'fdt', 'sgc', 'oth']
sec_area.columns = owner_labels
sec_area = sec_area.reset_index()

sec_area = pd.merge(sec, sec_area, on='flpl_sections')
sec_area = sec_area.set_index('section')
sec_area.drop(['flpl_sections'], axis=1, inplace=True)
col_order = ['rws', 'sbb', 'prv', 'prn', 'mun', 'wtb',
             'sgc', 'com', 'fdt', 'oth', 'cit']
sec_area = sec_area.reindex_axis(col_order, axis=1)

#-potential biodiversity
TFI_all = pd.read_csv(os.path.join(ens_overview_dir, 'TFI_all.csv'))
sec_TFI = TFI_all[TFI_all.msr == 'reference']
sec_TFI = pd.merge(sec, sec_TFI, left_on='flpl_sections', right_on='ID')
sec_TFI.set_index('section', inplace=True)
sec_TFI.drop(['flpl_sections', 'ID', 'msr'], inplace=True, axis=1)

#%% plot the overview per floodplain section

params = {'legend.fontsize': 7,
         'axes.labelsize': 8,
         'axes.titlesize': 8,
         'xtick.labelsize': 7,
         'ytick.labelsize': 7,
         'xtick.major.size': 2,
         'ytick.major.size': 2,
         'ytick.minor.size': 0,
         'ytick.major.pad': 2,
         'xtick.major.pad': 2,
         'axes.labelpad': 2.0,}
plt.rcParams.update(params)

fig, [ax1, ax2, ax3, ax4, ax5] = plt.subplots(ncols=1, nrows=5,
                                              sharex=True, figsize=(7.5, 10))
kind='bar'
lw = 0.1
width=1
cm = 'viridis'
    
#- Owner area
sec_area = sec_area2
try:
    sec_area.drop('oth', axis=1, inplace=True)
except ValueError:
    pass

owner_lut = {'rws': 'PWWM',
             'sbb': 'SFS',
             'prv': 'Province',
             'prn': 'Gelders Landschap',
             'mun': 'Municipality',
             'wtb': 'Water board',
             'sgc': 'Sand-gravel-clay',
             'com': 'Ltd companies',
             'fdt': 'Foundations',
             'oth': 'Other',
             'cit': 'Citizens'
             }
sec_area.rename(columns=owner_lut, inplace=True)
sec_area.plot(ax=ax1, kind=kind, linewidth=lw, width=width, stacked=True, cmap=cm)
ax1.legend(bbox_to_anchor=(1.26,-0.4), loc=1, labelspacing=-2.1)
ax1.set_ylabel('Area per owner type (ha)')
ax1.text(-0.1, 0.95,'(a)',transform=ax1.transAxes)
ax1.text(1.02, 0.25, 'Owner type',transform=ax1.transAxes, fontsize=8)

#- Owner number
sec_own2.plot(ax=ax2, kind=kind, linewidth=lw, width=width, stacked=True, cmap=cm, legend=False)
ax2.set_ylabel('Nr of owners\n per owner type (-)')
ax2.text(-0.1, 0.95, '(b)',transform=ax2.transAxes)
ax2.text(0.06, 0.9, 'Citizens  =  529', transform=ax2.transAxes, fontsize=8)
ax2.set_ylim(0,120)


#-roughness
ttp_lut = {1201: 'Production meadow',
           1202: 'Natural grassland',
           106:  'Lake',
           102:  'Groyne field',
           105:  'Side channel',
           1245: 'Softwood forest',
           1212: 'Dry herb. vegetation',
           1203: 'Rough grassland',
           114:  'Built-up, or paved',
           1250: 'Pioneer vegetation'
           }
sec_ttp.rename(columns=ttp_lut, inplace=True)
sec_ttp.plot(ax=ax3, kind=kind, linewidth=lw, width=width, stacked=True, cmap=cm)
ax3.legend(bbox_to_anchor=(1.26,0.2), labelspacing=-2.1)
ax3.set_ylabel('Area per roughness class')
ax3.text(-0.1, 0.95,'(c)',transform=ax3.transAxes)
ax3.text(1.02, 0.86, 'Roughness class',transform=ax3.transAxes, fontsize=8)

#- cost of land, buildings, minor embankments, roads and smoothing
cost_lut = {'c_smoothing': 'Roughness smoothing',
            'c_roads': 'Road removal',
            'c_minemb': 'Minor embankment rem.',
            'c_building': 'Building demolition',
            'c_land_acq': 'Land acquisition'
            }

sec_cost.rename(columns=cost_lut, inplace=True)
sec_cost.plot(ax=ax4, kind=kind, linewidth=lw, width=width, stacked=True, cmap=cm)
ax4.legend(bbox_to_anchor=(1.295,0.4), loc=1, labelspacing=-2.1)
ax4.set_ylabel('Cost (M euro)')
ax4.text(-0.1, 0.95,'(d)',transform=ax4.transAxes)
ax4.text(1.02, 0.67, 'Maximum costs',transform=ax4.transAxes, fontsize=8)

#-biodiversity
TFI_lut = {'HigherPlants': 'Higher Plants',
           'DragonDamselflies': 'Dragon- and damselflies'
           }
sec_TFI.rename(columns=TFI_lut, inplace=True)
sec_TFI.plot(ax=ax5, kind=kind, linewidth=lw, width=width, stacked=True, cmap=cm)
ax5.legend(bbox_to_anchor=(1.295,0.2), loc=1, labelspacing=-2.1)
ax5.set_ylabel('PotTax score (-)')
ax5.set_xticklabels(sec_TFI.index.values, fontsize=8)
ax5.text(-0.1, 0.95,'(e)',transform=ax5.transAxes)
ax5.text(1.02, 0.64, 'Taxonomic group',transform=ax5.transAxes, fontsize=8)
    
plt.subplots_adjust(right=0.8, hspace=0.07)   
plt.savefig('section_overview.png', dpi=300)

#%% number of owners per class
f1 = os.path.join(input_dir, 'restraints', 'owners_winbed.shp')
f2 = os.path.join(input_dir, 'restraints', 'owners_winbed_buf50.shp')
owners = gpd.read_file(f1)
owners_buf50 = gpd.read_file(f2)

#%%
print owners.shape
grps = owners.drop_duplicates(['Naam_Gerec', 'labels']).groupby('labels')
print grps.count()['owner_type']
grps = owners_buf50.drop_duplicates(['Naam_Gerec', 'labels']).groupby('labels')
print grps.count()['owner_type']

#%%
def maps_to_dataframe(maps, columns, MV = np.nan):
    """Convert a set of maps to flattened arrays as columns.
    
    Input: 
        maps: list of PCRaster maps
        columns: list of strings with column names
        MV: missing values, defaults to numpy nan.
    
    Returns a pandas dataframe with missing values in any column removed.
    """
    data = OrderedDict()
    for name, pcr_map in zip(columns, maps):
        data[name] = pcr.pcr2numpy(pcr_map, MV).flatten()
    
    return pd.DataFrame(data).dropna()


winbed = pcr.readmap(os.path.join(ref_map_dir, 'winter_bed.map'))
winbed = pcr.ifthen(winbed, winbed)
eco = mapIO.read_map_with_legend(os.path.join(ref_map_dir, 'ecotopes.map'))
eco_map = pcr.scalar(pcr.ifthen(winbed, eco.pcr_map))

eco_cells = maps_to_dataframe([eco_map, pcr.scalar(winbed)],
                               ['values', 'ha'])
eco_hectare = eco_cells.groupby('values').sum() * 625 / 10000
eco_hectare = eco_hectare.reset_index()
eco_stats = pd.merge(eco.legend_df, eco_hectare, on='values').sort_values('ECO_CODE')
lut = pd.read_csv(os.path.join(bio_dir, 'eco_conversion.csv'), index_col=0)

eco_simple = pd.merge(eco_stats, lut, on='values')
eco_simple = eco_simple.groupby('eco_simple').sum().reset_index()
eco_simple['perc'] = 100* eco_simple.ha / eco_simple.ha.sum()
#eco_simple.sort_values('eco_simple')
print eco_stats.head()
print lut.head()
print eco_simple
eco_simple.loc[:,['eco_simple', 'ha', 'perc']].to_csv('eco_simple.csv')




























