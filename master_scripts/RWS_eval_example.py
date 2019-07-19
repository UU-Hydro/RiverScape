# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 16:29:06 2017

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

def read_wl_ensemble(msr_fm_root, msr_names, var_names=[], xsec_names=[]):
    """
    Read the water levels at the rivers axis for an ensemble of measures. 
    """
    data = {}
    for msr_name in msr_names:
        f = os.path.join(msr_fm_root, msr_name, 'DFM_OUTPUT_msr/msr_his.nc.')
        if os.path.isfile(f) == True:
            wl = read_his_nc(f, station_vars=['waterlevel'])[0][0].iloc[-1,:]
            data[msr_name] = wl
        else:
            pass
    return pd.DataFrame(data=data)

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
ens_dir = os.path.join(output_dir, 'measures_ensemble03')
ens_map_dir = os.path.join(ens_dir, 'maps')
ens_FM_dir = os.path.join(ens_dir, 'hydro')
ens_overview_dir = os.path.join(ens_dir, 'overview')

scratch_dir = os.path.join(root_dir, 'scratch')
clone_file = os.path.join(ref_map_dir, 'clone.map')
pcr.setclone(clone_file)
pcr.setglobaloption('unittrue')
os.chdir(scratch_dir)
ppp
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
cost_all_msrs.to_csv(os.path.join(ens_overview_dir, 'cost_all.csv'))
std_all_msrs.to_csv(os.path.join(ens_overview_dir, 'std_all.csv'))

#%% Evaluate all measures on biodiversity
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
FI_all.to_csv(os.path.join(ens_overview_dir, 'FI_all.csv'))
TFI_all.to_csv(os.path.join(ens_overview_dir, 'TFI_all.csv'))

#%% Evaluate all measures on water level differences
# extract water level lowering at the river axis for all measures
msr_names = next(os.walk(ens_FM_dir))[1]
wl_ens = read_wl_ensemble(ens_FM_dir, msr_names, var_names=['waterlevel'])
dwl = wl_ens.subtract(wl_ens['reference_FM'], axis=0).iloc[29:,:]
dwl['stat_names'] = dwl.index.values
dwl['rkm'] = dwl.stat_names.str.split('_', expand=True)[0].astype(float)
dwl.index = dwl.rkm
dwl = dwl.drop(['stat_names', 'rkm'], axis=1)
dwl.to_csv(os.path.join(ens_overview_dir, 'dwl.csv'))

# Extract key statistics on water level changes
dwl_stats = dwl.describe()
dwl_lower_max = dwl_stats.loc['min',:]
dwl_higher_max = dwl_stats.loc['max',:]

# changes in surface area along the profile
   #plus backwater effect outside of the model domain
dwl_lower_area = 1000 * dwl.clip(upper=0).sum() - dwl.loc[868,:] * 0.5 * 20000
dwl_higher_area = 1000 * dwl.clip(lower=0).sum()

data = OrderedDict([('dwl_lower_max', dwl_lower_max),
                    ('dwl_higher_max', dwl_higher_max),
                    ('dwl_lower_area', dwl_lower_area),
                    ('rkm_dwl_lower_max', dwl.idxmin()),
                    ('rkm_dwl_higher_max', dwl.idxmax() )
                    ])
hydro_stats = pd.DataFrame(data=data)
hydro_stats.index.name = 'msr'
hydro_stats.to_csv(os.path.join(ens_overview_dir, 'hydro_stats.csv'))

#%% Summarize all effects
# Biodiversity
FI_all = pd.read_csv(os.path.join(ens_overview_dir, 'FI_all.csv'), index_col=0)
FI_all.set_index('msr', append=True, inplace=True)
FI_unstack = FI_all.unstack('msr')
FI_ref = FI_unstack.loc[:,('FI', 'reference')]
dFI = -FI_unstack.subtract(FI_ref, axis=0, level=-1)
dFI_all = dFI.stack().reset_index('msr')
dFI_all.columns = ['msr', 'dFI']

TFI_all = pd.read_csv(os.path.join(ens_overview_dir, 'TFI_all.csv'),
                      index_col=0)
groups = TFI_all.groupby('msr')
TFI_ref_numeric = TFI_all[TFI_all.msr == 'reference'].drop('msr', axis=1)
TFI_list = []
for group_name in groups.groups.keys():
    group_df = groups.get_group(group_name)
    dTFI_msr = -(group_df.drop('msr', axis=1) - TFI_ref_numeric)
    dTFI_msr['msr'] = group_name
    TFI_list.append(dTFI_msr)
dTFI_all = pd.concat(TFI_list, axis=0)

summary_dFI = dFI_all.groupby(by='msr').mean().sort_values('dFI')
summary_dTFI = dTFI_all.groupby(by='msr').mean()

# Cost
cost_all = pd.read_csv(os.path.join(ens_overview_dir, 'cost_all.csv'), index_col=0)
std_all = pd.read_csv(os.path.join(ens_overview_dir, 'std_all.csv'), index_col=0)
cost_all.index.name = 'msr'
cols = ['flpl_low_extstorage', 'minemb_extstorage', 'raise_50cm']
cost_summary = cost_all.drop(cols, axis=1).sum(axis=1)
cost_summary = cost_summary * 1e-6 # in M euro

# Hydro
hydro_stats = pd.read_csv(os.path.join(ens_overview_dir, 'hydro_stats.csv'), index_col=0)

# Combine the evaluations in a single dataframe
frames = [hydro_stats.loc[:,['dwl_lower_area', 'dwl_lower_max']],
          summary_dFI, summary_dTFI]
stats = pd.concat(frames, axis=1)
stats['cost_sum'] = cost_summary

del_list = ['dikeraising_cit_smooth', 'dikeraising_com_smooth',
            'dikeraising_fdt_smooth', 'dikeraising_mun_smooth',
            'dikeraising_prn_smooth', 'dikeraising_prv_smooth',
            'dikeraising_rws_smooth', 'dikeraising_sbb_smooth',
            'dikeraising_sgc_smooth', 'reference_FM', 'reference']
stats.drop(del_list, inplace=True)
stats['msr_type'] = [name.split('_')[0] for name in stats.index.values]
stats['msr_loc'] = [name.split('_')[1] for name in stats.index.values]
stats['eco_type'] = [name.split('_')[-1] for name in stats.index.values]
stats.loc['dikeraising_evr_smooth', 'dwl_lower_max'] = -0.5
stats.loc['dikeraising_wtb_smooth', 'dwl_lower_max'] = -0.5
stats.loc['dikeraising_evr_smooth', 'dwl_lower_area'] = -0.5 * 94000
stats.loc['dikeraising_wtb_smooth', 'dwl_lower_area'] = -0.5 * 94000
stats.to_csv(os.path.join(ens_overview_dir, 'stats.csv'))

#%% plot the output
# overall
def plot_eco_effect(x,y, **kwargs):
    points = plt.scatter(x,y, s=50, linewidth=0, edgecolors='k', **kwargs)
    tooltip = mpld3.plugins.PointLabelTooltip(points, labels=list(x.index.values))
    mpld3.plugins.connect(plt.gcf(), tooltip)
    msr_type = x.index[0].split('_')[0]
    if msr_type in ['lowering', 'sidechannel']:
        x_idx = x.reset_index()['index'].str.split('_', expand=True)
        y_idx = y.reset_index()['index'].str.split('_', expand=True)
        x_natural = x[(x_idx.iloc[:,2] == 'natural').values]
        x_smooth = x[(x_idx.iloc[:,2] == 'smooth').values]
        y_natural = y[(y_idx.iloc[:,2] == 'natural').values]
        y_smooth = y[(y_idx.iloc[:,2] == 'smooth').values]
        plt.plot(x_natural, y_natural, 'o',
                 ms=4, markerfacecolor='None', mec = 'g', mew=4, zorder=0)
        plt.plot(x_smooth, y_smooth, 'o',
                 ms=4, markerfacecolor='None', mec = 'b', mew=4, zorder=0)
        for nat in x_natural.index.values:
            smo = nat.replace('natural', 'smooth')
            x_coor = [x_natural[nat], x_smooth[smo]]
            y_coor = [y_natural[nat], y_smooth[smo]]
            plt.plot(x_coor, y_coor, '-', c='k', lw=0.5, zorder=-1)
    return x,y

def pareto_points(two_col_df):
    """Extract the optimal points in the lower left corner in 2D.
    input:
        two_col_df: DataFrame with two columns of x and y axes.
    """
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
    
    #- test
    debug=False
    if debug == True:
        fig, ax = plt.subplots()    
        two_col_df.plot.scatter('cost_sum', 'dwl_lower_max', ax=ax)
        hull_df = two_col_df.iloc[vertices,:]
        ax.plot(hull_df.cost_sum, hull_df.dwl_lower_max, c='r')
        optimal_df.plot.scatter('cost_sum', 'dwl_lower_max', ax=ax, c='r', s=50)
    return optimal_df


#params = {'legend.fontsize': 12,
#         'axes.labelsize': 12,
#         'axes.titlesize': 12,
#         'xtick.labelsize': 12,
#         'ytick.labelsize': 12,
#         'xtick.major.size': 2,
#         'ytick.major.size': 2,
#         'ytick.minor.size': 0,
#         'ytick.major.pad': 2,
#         'xtick.major.pad': 2,
#         'axes.labelpad': 2.0,}
#plt.rcParams.update(params)
sns.set_style('ticks')
sns.set_context("paper", font_scale=1)
locs = ['cit', 'com', 'evr', 'fdt', 'mun', 'prn', 'prv', 'rws', 'sbb', 'sgc', 'wtb']
locs = ['cit', 'com', 'fdt', 'mun', 'prn', 'prv', 'rws', 'sbb', 'sgc', 'wtb']
plot_vars = ['cost_sum', 'dFI', 'dwl_lower_max', 'dwl_lower_area']
#plot_vars = ['Birds', 'Butterflies', 'DragonDamselflies', 'Fish',
#             'Herpetofauna', 'HigherPlants', 'Mammals']

stats_sub = stats[stats.msr_loc.isin(locs)]
hue_order = stats.msr_type.unique()
markers = ['o', 's', 'd', 'D', 'H', 'h' ]
g = sns.PairGrid(stats_sub, hue='msr_type', hue_kws={'marker': markers},
                 vars=plot_vars, palette = 'colorblind', size=3)
g.map_lower(plot_eco_effect)
#g.map_lower(plt.scatter, s=40)
g.map_diag(plt.hist, bins=20)
g.add_legend()


#pp10 = pareto_points(stats_sub.loc[:,['cost_sum', 'dFI']])
#pp10.plot('cost_sum', 'dFI', ax=g.axes[1,0], label='opt')
#pp20 = pareto_points(stats_sub.loc[:,['cost_sum', 'dwl_lower_max']])
#pp20.plot('cost_sum', 'dwl_lower_max', ax=g.axes[2,0], label='opt')
#pp21 = pareto_points(stats_sub.loc[:,['dFI', 'dwl_lower_max']])
#pp21.plot('dFI', 'dwl_lower_max', ax=g.axes[2,1], label='opt')

mpld3.save_html(plt.gcf(), 'matrix_subLocs_all.html')
plt.savefig('matrix_subLocs_BIOSAFE.png', dpi=150)

## BIOSAFE
#dTFI_stats = dTFI_all.groupby(by='msr').mean()
#dTFI_stats.index.name = 'msr'
#dTFI_stats['msr_type'] = [name.split('_')[0] for name in dTFI_stats.index.values]
#sns.pairplot(dTFI_stats, hue='msr_type',
#             palette = 'colorblind', 
#             plot_kws=dict(s=70, edgecolor="w", linewidth=1))
#plt.savefig('scplma_BIOSAFE.png', dpi=150)

#%% test area

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

def plot_pareto(x, y, **kwargs):
    """Helper function for seaborn FacetGrid."""
    print x
    print y
    if len(x) >= 3:
        print ll
        
    else:
        pass
    ppp

plt.figure(figsize=(10,6))
df = stats.loc[:,['cost_sum', 'dwl_lower_area', 'msr_type', 'msr_loc']]
df = df[df.dwl_lower_area > -20000]
fg = sns.FacetGrid(df, hue='msr_loc', col='msr_type', col_wrap=2)
fg.map(plt.scatter, 'cost_sum', 'dwl_lower_area', s=90)
#fg = sns.FacetGrid(df, col='msr_type', col_wrap=2)
#fg.map(plt.scatter, 'cost_sum', 'dwl_lower_area', s=20)

fg.map(plt.hist, 'cost_sum')
#fg.map(plot_pareto, 'cost_sum', 'dwl_lower_area')
fg.add_legend()
plt.savefig('fg.png', dpi=300)

#%%
fig, ax = plt.subplots() 
for msr_type in df.msr_type.unique():
    print msr_type
    if msr_type == 'dikeraising':
        pass
    else:
        df_msr = df[df.msr_type == msr_type]
        pp= pareto_points(df_msr.loc[:,['cost_sum', 'dwl_lower_area']])
        pp.plot.line('cost_sum', 'dwl_lower_area', ax=ax,)
        df_msr.plot.scatter('cost_sum', 'dwl_lower_area', ax=ax, label=msr_type)
#ax.set_xlim(0,200)
ax.set_ylim(-20000, 5000)
ax.set_xscale('log')





































