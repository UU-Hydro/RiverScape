# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:29:06 2017

@author: straa005
"""
import fiona
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import uuid
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
sns.set_style("ticks")
import mapIO
import pcraster as pcr
import measures
import biosafe
import biosafeIO as bsIO
import evaluate_cost as ec


def send_to_back(in_shp, out_shp):
    """Reorder the features in in_shp shapefile from front to back."""
    gdf = gpd.read_file(in_shp)
    gdf.sort_index(ascending=False).to_file(out_shp)

def empty_measure(eco_clone):
    """Create an empty river measure."""
    empty_map = pcr.scalar(mapIO.emptyMap(eco_clone.pcr_map))
    empty_eco = measures.LegendMap(pcr.nominal(empty_map), eco_clone.legend_df)
    settings = pd.DataFrame({1:['lowering', 'dummy']},
                            index = ['msr_type', 'ID'], )
    msr = measures.Measure(settings, 
                           empty_map, empty_map, empty_eco, empty_map,
                           empty_map, empty_map, empty_map)
    return msr
    
def read_maptable(msr_dir, dem_ref):
    dem_file = os.path.join(msr_dir, 'maaiveldhoogte_pl.shp')
    #send_to_back(ttp_file, 'ttp_tmp.shp')
    #send_to_back(dem_file, 'dem_tmp.shp')
    
    dem_gdf = gpd.read_file(dem_file)
    dem_rel = dem_gdf[dem_gdf.relative == 0]
    if len(dem_rel) > 0:
        dem_rel.to_file('dem_rel.shp')
        dem_rel_map = mapIO.vector2raster('ESRI Shapefile', clone_file,
                                          'dem_rel.shp', 'value')
    else:
        dem_rel_map = pcr.scalar(mapIO.emptyMap(dem_rel_map))
    
    dem_abs = dem_gdf[dem_gdf.relative == 1]
    if len(dem_abs) > 0:
        dem_abs.to_file('dem_abs.shp')
        dem_abs_map = mapIO.vector2raster('ESRI Shapefile', clone_file,
                                          'dem_abs.shp', 'value')
    else:
        dem_abs_map = pcr.scalar(mapIO.emptyMap(dem_rel_map))
    
    dem = pcr.cover(dem_abs_map, dem_rel_map + dem_ref)
    
    # create trachytopes for the measure
    ttp_file = os.path.join(msr_dir, 'landgebruik_pl.shp')
    ttp = mapIO.vector2raster('ESRI Shapefile', clone_file, ttp_file, 'code')
    return dem, ttp

def lookup_scalar(pcr_map, column_data):
    """
    """
    lut_file = 'lut_reclass_%s.txt' % uuid.uuid4() 
        # needs a unique file name due to bug in pcr.lookupnominal
    np.savetxt(lut_file, column_data, fmt = '%.0f')
    recoded = pcr.lookupscalar(lut_file, pcr.scalar(pcr_map))
    os.remove(lut_file)
    return recoded



#%% Initial settings
root_dir = os.path.dirname(os.getcwd())
input_dir = os.path.join(root_dir, 'input')
ref_map_dir = os.path.join(input_dir, 'reference_maps')
bio_dir = os.path.join(input_dir, 'bio')
cost_dir = os.path.join(input_dir, 'cost')
output_dir = os.path.join(root_dir, 'output/waal_XL')
ens_map_dir = os.path.join(output_dir, 'measures_ensemble04/maps')
ens_FM_dir = os.path.join(output_dir, 'hydro')
scratch_dir = os.path.join(root_dir, 'scratch')
clone_file = os.path.join(ref_map_dir, 'clone.map')
pcr.setclone(clone_file)
pcr.setglobaloption('unittrue')
os.chdir(scratch_dir)


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

#%% Initialize BIOSAFE
ndff_species = pd.read_pickle(os.path.join(bio_dir, 'ndff_sub_BS_13.pkl'))
flpl_sections = pcr.readmap(os.path.join(bio_dir, 'flpl_sections.map'))
ecotopes = measures.read_map_with_legend(os.path.join(bio_dir, 'ecotopes.map'))
legalWeights, linksLaw, linksEco = bsIO.from_csv(bio_dir)
speciesPresence = pd.DataFrame(np.random.randint(2, size=len(linksLaw)),\
                    columns=['speciesPresence'], \
                    index=linksLaw.index)
ecotopeArea = pd.DataFrame(np.ones(82) * 1e5,\
                           columns = ['area_m2'],\
                           index = linksEco.columns.values[0:-1])

bs = biosafe.biosafe(legalWeights, linksLaw, linksEco, speciesPresence, ecotopeArea)
excel_file = os.path.join(bio_dir, 'BIOSAFE_20150629.xlsx')
lut1 = pd.read_excel(excel_file, sheetname = 'lut_RWES').fillna(method='ffill')
    # this lookup table has:
    #       ecotope codes of BIOSAFE in the first column: oldEcotope
    #       aggregated/translated ectotopes in the second column: newEcotope
linksEco1 = biosafe.aggregateEcotopes(linksEco, lut1)
bs.linksEco = linksEco1
TEI = bs.TEI()
TEI.to_csv('TEI.csv')
print TEI.loc['Butterflies', 'OG-1':'UB-3']

#%% calculate all costs
# Read input maps
dike_length = pcr.readmap(os.path.join(cost_dir, 'dike_length.map'))
dike_raise_sections = pcr.readmap(os.path.join(cost_dir,
                                               'dike_raise_sections.map'))
pollution_zones  = pcr.readmap(os.path.join(cost_dir,
                                            'pollution_zones.map'))
smoothing_cost_classes = pcr.readmap(os.path.join(cost_dir,
                                                 'smoothing_cost_classes.map'))
dem = pcr.readmap(os.path.join(ref_map_dir, 'dem.map'))

# Read input distributions
road_distr = ec.read_distribution(cost_dir, 'roads')
smoothing_distr = ec.read_distribution(cost_dir, 'smoothing')
building_distr = ec.read_distribution(cost_dir, 'building_tot')
dike_raise50_distr = ec.read_distribution(cost_dir, 'dike_raise50')
dike_raise100_distr = ec.read_distribution(cost_dir, 'dike_raise100')
dike_reloc_distr = ec.read_distribution(cost_dir, 'dike_reloc')
land_acq_distr = ec.read_distribution(cost_dir, 'land_acq')
minemb_ext_use_distr = ec.read_distribution(cost_dir, 'minemb_ext_use')
minemb_int_use_distr = ec.read_distribution(cost_dir, 'minemb_int_use')
minemb_polluted_distr = ec.read_distribution(cost_dir, 'minemb_polluted')

groyne = measures.read_dike_maps(ref_map_dir, 'groyne')
minemb = measures.read_dike_maps(ref_map_dir, 'minemb')
main_dike = measures.read_dike_maps(ref_map_dir, 'main_dike')

# Set earthwork values
cost_input_ew = pd.read_csv(os.path.join(cost_dir, 'cost_ew.csv'), index_col=0)
cost_correction = pd.read_csv(os.path.join(cost_dir, 'cost_correction.csv'),
                              index_col=0, comment='#')

# Instantiate cost classes
# Smoothing
c_sm = ec.CostSmoothing('dummy', smoothing_distr)

# Earth work
c_ew = ec.CostEarthwork('dummy',
             minemb_ext_use_distr, minemb_int_use_distr, minemb_polluted_distr,
             dem, minemb, groyne,
             pollution_zones,
             cost_input_ew)

# Land preparation
c_lp = ec.CostPreparation('dummy', 
                       land_acq_distr, road_distr, smoothing_distr,
                       building_distr, smoothing_cost_classes)

### Dike raising
c_draise = ec.CostDikeRaising('dummy',
                           dike_raise50_distr, dike_raise100_distr,
                           dike_length)

# Evaluate all measures on cost       
cost_types = [c_sm, c_ew, c_lp, c_draise]
cost_all_msrs, std_all_msrs = ec.assess_effects(cost_types, ens_map_dir,
                                                cost_correction)
cost_all_msrs.to_csv('cost_all.csv')
std_all_msrs.to_csv('std_all.csv')

#%% Plot costs
costs = cost_all_msrs.copy()
costs.drop(['flpl_low_extstorage', 'minemb_extstorage','raise_50cm'],
           axis=1, inplace=True)

costs.drop(['dikeraising_lrg_smooth'], inplace=True)


idx0 = costs.index.values
idx1 = [s.replace('_evr_', '_AS_') for s in idx0]
idx2 = [s.replace('_lrg_', '_LS_') for s in idx1]
idx_lut = {k:v for k,v in zip(idx0, idx2)}
costs.rename(index=idx_lut, inplace=True)

cols = ['land_acq', 'building_rem', 'flpl_low_localuse',
        'raise_100cm', 'flpl_low_polluted', 
        'minemb_polluted', 'groyne_low',
        'road_rem', 'minemb_localuse', 'smoothing', 'forest_rem']

new_cols = ['land acquisition', 'building removal', 'earthworks flpl lowering',
            'dike raising', 'earthworks polluted soil',
            'earthworks minemb polluted soil', 'groyne lowering', 
            'road removal', 'earthworks minemb', 'smoothing', 'forest removal']
cols_lut = {k:v for k,v in zip (cols, new_cols)}
costs.rename(columns=cols_lut, inplace=True)

fig, ax = plt.subplots(1,1, figsize=(3,4))
costs = costs[costs.sum().sort_values(ascending=False).index.values]
idx = costs.sum(axis=1).sort_values(ascending=False).index.values
costs = costs.loc[idx]
colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c',
          '#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99']              
(1e-6 * costs).plot.bar(ax=ax,stacked=True, width=0.8, legend='reverse', color=colors)
ax.set_ylabel('Implementation costs (MEuro)')
ax.set_xlabel('Measure')
ax.set(ylim=(-25, 1800))
plt.subplots_adjust(left=0.16, bottom=0.39, top=0.98, right=0.98)
plt.savefig('costs.png', dpi=300)


#%% Evaluate measures on biodiversity
def assess_bio_effects(bio_mod, eco_ref, sections, ndff_species, maps_dir):
    """
    Determine all effects for all measures.
    """
    bs_ref = biosafe.spatialBiosafe(bio_mod, eco_ref, sections, ndff_species,
                                    params=['FI', 'TFI'], toFiles=None)
    FI_ref, TFI_ref = bs_ref.spatial()
    TFI_ref.drop(['xcoor', 'ycoor'], axis=1, inplace=True)
    TFI_ref['msr'] = 'reference'
    FI_ref.drop(['xcoor', 'ycoor'], axis=1, inplace=True)
    FI_ref['msr'] = 'reference'
    FI_list, TFI_list = [FI_ref], [TFI_ref]
    measure_names = [f for f in os.listdir(maps_dir)
                       if os.path.isdir(os.path.join(maps_dir,f))]
    for msr_name in measure_names[:]:
        print msr_name
        measure_pathname = os.path.join(maps_dir, msr_name)
        assert os.path.isdir(measure_pathname) == True
        msr = measures.read_measure(measure_pathname)
        eco_map = pcr.cover(msr.ecotopes.pcr_map, ecotopes.pcr_map)
        msr_eco = measures.LegendMap(eco_map, ecotopes.legend_df)
        bs_msr = biosafe.spatialBiosafe(bio_mod, msr_eco, sections, 
                                        ndff_species, params=['FI','TFI'],
                                        toFiles=None)
        FI_msr, TFI_msr = bs_msr.spatial()                                
        TFI_msr.drop(['xcoor', 'ycoor'], axis=1, inplace=True)
        TFI_msr['msr'] = msr_name
        FI_msr.drop(['xcoor', 'ycoor'], axis=1, inplace=True)
        FI_msr['msr'] = msr_name
        FI_list.append(FI_msr)
        TFI_list.append(TFI_msr)
    
    FI_all = pd.concat(FI_list)    
    TFI_all = pd.concat(TFI_list)
    return FI_all, TFI_all

# test
bs_ref = biosafe.spatialBiosafe(bs, ecotopes, flpl_sections, ndff_species,
                                    params = ['FI', 'TFI'],
                                    toFiles = None)
FI_ref, TFI_ref = bs_ref.spatial()

#msr = measures.read_measure(os.path.join(ens_map_dir, 'sidechannel_everywhere'))
#msr_eco = measures.LegendMap(pcr.cover(msr.ecotopes.pcr_map, ecotopes.pcr_map),
#                             ecotopes.legend_df)
#
#bs_msr = biosafe.spatialBiosafe(bs, msr_eco, flpl_sections, ndff_species,
#                                params = ['FI', 'TFI'], toFiles=None)
#FI_msr, TFI_msr = bs_msr.spatial()

# application
FI_all, TFI_all = assess_bio_effects(bs, ecotopes, flpl_sections, ndff_species,
                                     ens_map_dir)
FI_all.to_csv('FI_all.csv')
TFI_all.to_csv('TFI_all.csv')

#%% Summarize all effects
# Biodiversity
FI_all = pd.read_csv('FI_all.csv', index_col=0)
TFI_all = pd.read_csv('TFI_all.csv', index_col=0)
summary_FI = FI_all.groupby(by='msr').mean().sort_values('FI')
summary_TFI = TFI_all.groupby(by='msr').mean()

# Cost
cost_all_msrs = pd.read_csv('cost_all.csv', index_col=0)
std_all_msrs = pd.read_csv('std_all.csv', index_col=0)
cost_all_msrs.index.name = 'msr'
cols = ['flpl_low_extstorage', 'minemb_extstorage', 'raise_50cm']
cost_summary = cost_all_msrs.drop(cols, axis=1).sum(axis=1)

# Hydro
hydro_stats = pd.read_csv('hydro_stats.csv', index_col = (0,1))
dikeraise_km = pd.read_csv('dikeraise_km.csv', index_col=0)
hydro_stats_reaches = hydro_stats.copy().T
hydro_stats_reaches.columns = [hydro + '_' + reach 
                               for hydro, reach in hydro_stats_reaches.columns.values]
hydro_stats_reaches.index.name = 'msr'
hydro_stats_reaches.columns.name = None
hydro_stats.rename(index={'reference_FM_dQdh':'reference'}, inplace=True)
hydro_stats_all = hydro_stats.mean(level=0).T
hydro_stats_all.index.name = 'msr'
hydro_stats_all.columns.name = None
hydro_stats_all.rename(index={'reference_FM_dQdh':'reference'}, inplace=True)
hydro_stats_all = pd.concat([hydro_stats_all, dikeraise_km], axis=1)
print hydro_stats_all


#%% calculate number of owners involved in the measure.
restraints_dir = os.path.join(input_dir, 'restraints')

owner_nr = pcr.readmap(os.path.join(restraints_dir, 'owner_nr.map'))  
  # Rasterized from 'Nr_Gerechtigde' from cadastral map, removes multiple owners
owners_per_cell = pcr.readmap(os.path.join(restraints_dir, 'owner_point_number.map'))
  # number of owners per cell, from dissolve >> point >> point2raster (count)

# mask out areas from owner_nr with multiple ownners
owner_total = pcr.areatotal(owners_per_cell, owner_nr)
import gdal, ogr
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

#area_map = os.path.join(ens_map_dir, 'sidechannel_lrg_smooth\\area.map')
#area_shp = 'area.shp'
#pcr2Shapefile(area_map, area_shp, 'test')
#gdf = gpd.read_file(area_shp)
#gdf = gdf[gdf.test == 1]
#gdf.to_file('gdf.shp')
#
#owners_clean = gpd.read_file('owners_clean.shp')
#owners_msr = gpd.overlay(gdf, owners_clean, how='intersection')
#owners_msr.to_file('owners_msr.shp')
#%% owners from vector data.
owners_clean = gpd.read_file('owners_clean.shp')

measure_names = [f for f in os.listdir(ens_map_dir)
                       if os.path.isdir(os.path.join(ens_map_dir,f))]

overlay = False
if overlay == True:
    for msr_name in measure_names[:]:
        msr_map = os.path.join(ens_map_dir, msr_name, 'area.map')
        print msr_map
        pcr2Shapefile(msr_map, msr_name + '.shp', 'msr')
        msr_gdf = gpd.read_file(msr_name + '.shp')
        msr_gdf2 = msr_gdf[msr_gdf['msr'] == 1]
        msr_gdf2.to_file(msr_name + '_clean.shp')
        owners_msr = gpd.overlay(msr_gdf2, owners_clean, how='intersection')
        owners_msr.to_file(msr_name + '_owners.shp')

ll = []
for msr_name in measure_names:
    owners_msr = gpd.read_file(msr_name + '_owners.shp')
    nr_stakeholders_vector = owners_msr.drop_duplicates(['Naam_Gerec', 'shared_by']).shared_by.sum()
    if 'lrg' in msr_name: 
        nr_stakeholders_vector = 2
    if 'groyne' in msr_name:
        nr_stakeholders_vector = 1
    ll.append(nr_stakeholders_vector)
stakeholder_series = pd.Series(data = np.array(ll), index=measure_names)
#measure_names = [f for f in os.listdir(ens_map_dir)
#                       if os.path.isdir(os.path.join(ens_map_dir,f))]
#ll=[]
#for msr_name in measure_names[:]:
#    print msr_name
#    msr_area = pcr.readmap(os.path.join(ens_map_dir, msr_name, 'area.map'))
#    stakeholders_single = pcr.ifthen(msr_area, owner_nr)
#    nr_stakeholders_single = len(np.unique(pcr.pcr2numpy(stakeholders_single, -9999)))
#    ll.append(nr_stakeholders_single)
#    print '\t ', nr_stakeholders
#stakeholder_series = pd.Series(data = np.array(ll), index=measure_names)
print stakeholder_series
#%% hydro reaches plot
palette = sns.color_palette('nipy_spectral', 16)
sns.palplot(palette)
sns.set_palette(palette)

sub = hydro_stats_reaches.iloc[:,[0,2,3,5,6,8]]
cols = ['dikeraising_evr_smooth', 'dikeraising_lrg_smooth']
sub.drop(cols, inplace=True)
sub.columns = [s[4:] for s in sub.columns]
sub=sub.loc[['reference_FM_dQdh','groynelowering_evr_smooth', 'groynelowering_lrg_smooth',
       'lowering_evr_natural', 'lowering_evr_smooth',
       'lowering_lrg_natural', 'lowering_lrg_smooth',
       'minemblowering_evr_smooth', 'minemblowering_lrg_smooth',
        'sidechannel_evr_natural',
       'sidechannel_evr_smooth', 'sidechannel_lrg_natural',
       'sidechannel_lrg_smooth', 'smoothing_evr_natural',
       'smoothing_evr_smooth', 'smoothing_lrg_natural',
       'smoothing_lrg_smooth'],:]

col_lut = {'Qref_low': 'Q16_dh0.0, lower',
           'Qref_upp': 'Q16_dh0.0, upper',
           'dQ_low': 'Q18_dh0.0, lower',
           'dQ_upp': 'Q18_dh0.0, upper',
           'dQdh_low': 'Q18_dh1.8, lower',
           'dQdh_upp': 'Q18_dh1.8, upper'}
sub.rename(index=idx_lut, inplace=True)
sub.rename(columns = col_lut, inplace=True)

fig, ax = plt.subplots(figsize=(8,3))
sub.T.plot.bar(ax=ax, width=.8, linewidth=0.25, rot=0)
plt.legend(ncol=3, fontsize=6.6, loc=4)
plt.subplots_adjust(bottom=0.2, top=0.99, left=0.1, right=0.99)
plt.xlabel('\nHydrodynamic scenario, river reach')
plt.ylabel('Change in water level compared \nwith reference (Q16_dh0.0) (m)')
plt.ylim(-1.35, 1.1)
plt.savefig('reaches.png', dpi=300)

#%% plot biodiversity effects

#-prepare the data
msrs =['lowering_evr_natural', 'lowering_evr_smooth',
       'lowering_lrg_natural', 'lowering_lrg_smooth',
       'reference', 'sidechannel_evr_natural', 'sidechannel_evr_smooth',
       'sidechannel_lrg_natural', 'sidechannel_lrg_smooth',
       'smoothing_evr_natural', 'smoothing_evr_smooth',
       'smoothing_lrg_natural', 'smoothing_lrg_smooth'
       ]

sec = pd.read_csv('sections_left_right.csv', index_col=0)
TFI_all = pd.read_csv('TFI_all.csv', index_col=0)
FI_all = pd.read_csv('FI_all.csv', index_col=0)
FI_eco = FI_all[FI_all.msr.isin(msrs)]

FI_eco = FI_eco.set_index('msr', append=True)
FI_eco = FI_eco.unstack()
FI_eco.columns = FI_eco.columns.droplevel(level=0)
dFI_eco = FI_eco.subtract(FI_eco.loc[:,'reference'], axis='index')

dFI_eco['flpl_sections'] = dFI_eco.index.values
dFI_eco = pd.merge(dFI_eco, sec, on='flpl_sections')
dFI_eco.drop(['flpl_sections', 'reference'], axis=1, inplace=True)
dFI_eco.set_index('section', inplace=True)
dFI_eco.sort_index(inplace=True)

dFI_eco.rename(columns=idx_lut, inplace=True)

#-plot the data
fig, [ax1, ax2, ax3] = plt.subplots(3,1, sharex=True, figsize=(7,4))
kind='bar'
lw = 0.1
width=.8
cm = 'viridis'

# ax1: smoothing, use for legend building
smooth_cols = ['smoothing_AS_smooth', 'smoothing_LS_smooth',
               'smoothing_AS_natural', 'smoothing_LS_natural']

dFI_smooth = dFI_eco[smooth_cols]
cols_smooth = [ii[10:] for ii in smooth_cols]
dFI_smooth.columns = cols_smooth
dFI_smooth.plot(ax=ax1, width=width, kind=kind,
                          cmap=cm, linewidth=lw, rot=90)

ax1.legend(bbox_to_anchor=(0.85, 1.27), loc=1, labelspacing=-2, fontsize=8, ncol=4)
ax1.set_ylabel('delta PotAll (-)')
ax1.set_ylim(-120,120)
ax1.text(-0.1, 0.9, '(a)', transform=ax1.transAxes)


# ax2: side channels
side_cols = ['sidechannel_AS_smooth', 'sidechannel_LS_smooth',
             'sidechannel_AS_natural', 'sidechannel_LS_natural']
dFI_eco[side_cols].plot(ax=ax2, width=width, kind=kind,
                          cmap=cm, linewidth=lw, rot=90, legend=None)
#ax2.legend(bbox_to_anchor=(1.295,0.2), loc=1, labelspacing=-2, fontsize=7)
ax2.set(ylim=(-4, 13))
ax2.set_ylabel('delta PotAll (-)')
ax1.text(-0.1, 0.9, '(b)', transform=ax2.transAxes)


#ax3: floodplain lowering
low_cols = ['lowering_AS_smooth', 'lowering_LS_smooth',
            'lowering_AS_natural', 'lowering_LS_natural']
dFI_eco[low_cols].plot(ax=ax3, width=width, kind=kind,
                          cmap=cm, linewidth=lw, rot=90, legend=None)
#ax3.legend(bbox_to_anchor=(1.26,0.2), loc=1, labelspacing=-2, fontsize=7)
ax3.set_ylabel('delta PotAll (-)')
ax3.set_ylim(-120,120)
ax3.text(-0.1, 0.9, '(c)', transform=ax3.transAxes)

plt.subplots_adjust(left=0.1, bottom=0.11, top=0.95, right=0.98, hspace=0.1)
plt.savefig('dPotAll_eco.png', dpi=300)


#%% compile stats overview
m_order = ['dikeraising_evr_smooth', 'dikeraising_lrg_smooth',
           'groynelowering_evr_smooth', 'groynelowering_lrg_smooth',
           'minemblowering_evr_smooth', 'minemblowering_lrg_smooth',
           'smoothing_evr_natural', 'smoothing_evr_smooth',
           'smoothing_lrg_natural', 'smoothing_lrg_smooth',
           'sidechannel_evr_natural', 'sidechannel_evr_smooth',
           'sidechannel_lrg_natural', 'sidechannel_lrg_smooth',
           'lowering_evr_natural', 'lowering_evr_smooth',
           'lowering_lrg_natural', 'lowering_lrg_smooth',
           'reference']
stats = hydro_stats_all.copy()
labels = pd.Series(data=stats.index.values, index=stats.index).str.split('_', expand=True)
labels.columns = ['msr', 'area', 'eco']
stats['labels'] = stats.index.values
stats['cost_sum'] = cost_summary
stats['nr_stakeholders'] = stakeholder_series
stats.rename(index={'reference_FM_dQdh':'reference'}, inplace=True)
stats = pd.concat([stats, summary_FI, labels], axis=1).fillna(0)
stats.loc['dikeraising_evr_smooth', 'dwl_Qref'] = -1
stats = stats.loc[m_order]

palette =   [(0.89059593116535862, 0.10449827132271793, 0.11108035462744099),
             (0.98320646005518297, 0.5980161709820524, 0.59423301088459368),
             
             (0.99760092286502611, 0.99489427150464516, 0.5965244373854468),
             (0.69411766529083252, 0.3490196168422699, 0.15686275064945221),
                          
             (0.99175701702342312, 0.74648213716698619, 0.43401768935077328),
             (0.99990772780250103, 0.50099192647372981, 0.0051211073118098693),
             
             (0.78329874347238004, 0.68724338552531095, 0.8336793640080622),
             (0.78329874347238004, 0.68724338552531095, 0.8336793640080622),
             (0.42485198495434734, 0.2511495584950722, 0.60386007743723258),
             (0.42485198495434734, 0.2511495584950722, 0.60386007743723258),
             
             (0.68899655751153521, 0.8681737867056154, 0.54376011946622071),
             (0.68899655751153521, 0.8681737867056154, 0.54376011946622071),
             (0.21171857311445125, 0.63326415104024547, 0.1812226118410335),                          
             (0.21171857311445125, 0.63326415104024547, 0.1812226118410335),
             
             (0.65098041296005249, 0.80784314870834351, 0.89019608497619629),
             (0.65098041296005249, 0.80784314870834351, 0.89019608497619629),
             (0.12572087695201239, 0.47323337360924367, 0.707327968232772),
             (0.12572087695201239, 0.47323337360924367, 0.707327968232772),             
             ]
markers = ['o', 'o', 'o', 'o', 'o', 'o', 
           's', '>', 's', '>', 
           's', '>', 's', '>', 
           's', '>', 's', '>', 
           'D']

markers = ['o', 'o', 'o', 'o', 'o', 'o', 
           's', '>', 's', '>', 
           's', '>', 's', '>', 
           's', '>', 's', '>', 
           'D']


import scipy

def pareto_points(two_col_df):
    """Extract the optimal points in the lower left corner in 2D.
    input:
        two_col_df: DataFrame with two columns.
    """
    print two_col_df.head()
    points = two_col_df.values
    hull = scipy.spatial.ConvexHull(points)
    vertices = np.hstack((hull.vertices, hull.vertices[0]))
    ll = []
    for ii in range(len(vertices[0:-1])):
        p1 = points[vertices[ii]]
        p2 = points[vertices[ii+1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if (dx >= 0) & (dy <= 0):
            ll.append(p1)
            ll.append(p2)
    optimal_df = pd.DataFrame(data=np.vstack(ll), columns=two_col_df.columns)
    optimal_df.drop_duplicates(inplace=True)
    optimal_df.sort_values(optimal_df.columns[0], inplace=True)
    return optimal_df


stats.rename(index=idx_lut, inplace=True)
stats['labels'] = stats.index.values
print stats
g = sns.pairplot(stats, 
                 hue='labels', 
                 vars = ['dwl_Qref', 'nr_stakeholders', 'FI', 'cost_sum'],
                 palette = palette, 
                 markers=markers,
                 plot_kws=dict(s=40, edgecolor="w", linewidth=0.1))
g.axes[2,0].invert_yaxis()
g.axes[3,2].invert_xaxis()

#plot pareto points
kwargs = {'linewidth':11, 'alpha':0.15, 'color':'k', 'zorder':0}
pp = pareto_points(stats[['dwl_Qref', 'nr_stakeholders']])
g.axes[1,0].plot(pp.iloc[:,0].values, pp.iloc[:,1].values, **kwargs)


pp = pareto_points(pd.concat([stats[['dwl_Qref']], -stats[['FI']]], axis=1))
g.axes[2,0].plot(pp.iloc[:,0].values, -pp.iloc[:,1].values, **kwargs)

pp = pareto_points(stats[['dwl_Qref', 'cost_sum']])
g.axes[3,0].plot(pp.iloc[:,0].values, pp.iloc[:,1].values, **kwargs)

pp = pareto_points(pd.concat([stats[['nr_stakeholders']], -stats[['FI']]], axis=1))
g.axes[2,1].plot(pp.iloc[:,0].values, -pp.iloc[:,1].values, **kwargs)

try: # single point gets selected.
    pp = pareto_points(stats[['nr_stakeholders', 'cost_sum']].drop('reference'))
    g.axes[3,1].plot(pp.iloc[:,0].values, pp.iloc[:,1].values, **kwargs)
except ValueError:
    pass

pp = pareto_points(pd.concat([-stats[['FI']], stats[['cost_sum']]], axis=1))
g.axes[3,2].plot(-pp.iloc[:,0].values, pp.iloc[:,1].values, **kwargs)


g.axes[3,0].set_xlabel('water level lowering (m)')
g.axes[3,1].set_xlabel('nr of stakeholders (-)')
g.axes[3,2].set_xlabel('PotAll (-)')

g.axes[3,0].set_ylabel('Implementation costs (euro)')
g.axes[2,0].set_ylabel('PotAll (-)')
g.axes[1,0].set_ylabel('nr of stakeholders (-)')

g.axes[1,0].set_ylim(-150, 1300)
g.axes[3,1].set_xlim(-150, 1300)
g.axes[3,0].set_ylim(-2.5e8, 2e9)



x,y = -0.21, 0.95
g.axes[1,0].text(x,y, '(a)', transform=g.axes[1,0].transAxes)
g.axes[2,0].text(x,y, '(b)', transform=g.axes[2,0].transAxes)
g.axes[3,0].text(x,y, '(c)', transform=g.axes[3,0].transAxes)
g.axes[2,1].text(x,y, '(d)', transform=g.axes[2,1].transAxes)
g.axes[3,1].text(x,y, '(e)', transform=g.axes[3,1].transAxes)
g.axes[3,2].text(x,y, '(f)', transform=g.axes[3,2].transAxes)

plt.savefig('scplma_gov.png', dpi=600)
plt.savefig('scplma_gov.svg')
#import mpld3
#mpld3.save_html(plt.gcf(), 'scpl_ma.html')
stats.to_csv('stats_scatterplot.csv')

















