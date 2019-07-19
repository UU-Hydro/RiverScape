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
ens_map_dir = os.path.join(output_dir, 'measures_ensemble01/maps')
ens_FM_dir = os.path.join(output_dir, 'hydro')
scratch_dir = os.path.join(root_dir, 'scratch')
clone_file = os.path.join(ref_map_dir, 'clone.map')
pcr.setclone(clone_file)
pcr.setglobaloption('unittrue')
os.chdir(scratch_dir)

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

#%% Initialize cost module
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

#%% Instantiate cost classes
### Smoothing
c_sm = ec.CostSmoothing('dummy', smoothing_distr)

### Earth work
c_ew = ec.CostEarthwork('dummy',
             minemb_ext_use_distr, minemb_int_use_distr, minemb_polluted_distr,
             dem, minemb, groyne,
             pollution_zones,
             cost_input_ew)

### Land preparation
c_lp = ec.CostPreparation('dummy', 
                       land_acq_distr, road_distr, smoothing_distr,
                       building_distr, smoothing_cost_classes)

### Dike relocation
c_dreloc = ec.CostDikeRelocation('dummy', dike_reloc_distr, dike_length)

### Dike raising
c_draise = ec.CostDikeRaising('dummy',
                           dike_raise50_distr, dike_raise100_distr,
                           dike_length)

#%% Evaluate all measures on cost       
cost_types = [c_sm, c_ew, c_lp, c_dreloc, c_draise]
cost_all_msrs, std_all_msrs = ec.assess_effects(cost_types, ens_map_dir,
                                                cost_correction)
cost_all_msrs.to_csv('cost_all.csv')
std_all_msrs.to_csv('std_all.csv')
print cost_all_msrs.iloc[:,0:3]
#print std_all_msrs

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
dwl = pd.read_csv('dwl.csv', index_col=0)
dwl_stats = dwl.describe()
dwl_lower_max = dwl_stats.loc['min',:]
dwl_higher_max = dwl_stats.loc['max',:]

# changes in surface area along the profile
dwl_lower_area = 1 * dwl.clip(upper=0).sum()
dwl_higher_area = 1 * dwl.clip(lower=0).sum()

data = OrderedDict([('dwl_lower_max', dwl_lower_max),
                    ('dwl_higher_max', dwl_higher_max),
                    ('dwl_lower_area', dwl_lower_area),
                    ('dwl_higher_area', dwl_higher_area),
                    ('rkm_dwl_lower_max', dwl.idxmin()),
                    ('rkm_dwl_higher_max', dwl.idxmax() )
                    ])
hydro_stats = pd.DataFrame(data=data)
hydro_stats.index.name = 'msr'

stats = pd.concat([hydro_stats.loc[:,['dwl_lower_area', 'dwl_lower_max']], summary_FI], axis=1)
stats['cost_sum'] = cost_summary
stats.loc['dikeraising_everywhere', 'dwl_lower_area'] = -1000 * 94000 # sq m.
stats.loc['dikeraising_wb', 'dwl_lower_area'] = -1000 * 94000 # sq m.
stats.loc['dikeraising_everywhere', 'dwl_lower_max'] = -0.5 # m.
stats.loc['dikeraising_wb', 'dwl_lower_max'] = -0.5 # m.
stats.drop(['reference', 'reference_FM', 'lowering_everywhere'], inplace=True)
stats['msr_type'] = [name.split('_')[0] for name in stats.index.values]
stats['everywhere'] = ['everywhere' in i for i in stats.index.values]


hue_order = stats.msr_type.unique()
sns.pairplot(stats, hue='everywhere', vars=['dwl_lower_max', 'cost_sum', 'FI'],
             palette = 'colorblind', 
             plot_kws=dict(s=70, edgecolor="w", linewidth=1))
plt.savefig('scplma_gov.png', dpi=400)

#cheap = stats[stats.cost_sum < 5e6]
#sns.pairplot(cheap, hue='msr_type', vars=list(stats.columns.values[0:4]),
#             palette = 'colorblind', hue_order=hue_order,
#             plot_kws=dict(s=70, edgecolor="w", linewidth=1))
#plt.savefig('scplma_gov_cheap.png')
#%%



































