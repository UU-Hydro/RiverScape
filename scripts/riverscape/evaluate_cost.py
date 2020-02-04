#import fiona
import os
import numpy as np
import pandas as pd
import pcraster as pcr

from .measures import Measure, read_measure, read_dike_maps
from . import ipynb_utils


#%% helper functions
def read_check_data(path_to_map, clone_file):
    """Check if a map exists and contains data.
    input:
    path_to_map: path to a PCRaster .map file"""
    try:
        is_file = os.path.isfile(path_to_map)
        pcr_map = pcr.readmap(path_to_map)
        unique_values = np.unique(pcr.pcr2numpy(pcr_map, -9999))
        unique_values = unique_values[unique_values > -9999]
        has_data = unique_values > 0
    except RuntimeError:
        print('WARNING: File {} does not exist. Empty map used instead'.format(path_to_map))
        clone = pcr.readmap(clone_file)
        pcr_map = pcr.ifthen(pcr.scalar(clone) == -9999, pcr.scalar(-9999))
        is_file = False
        has_data = False
    return is_file, has_data, pcr_map

def area_total_value(values, area_class):
    """Calculate the total value over the area class.

    values: map with values
    area_class: project area of the measure

    Returns a float with the total value over the area class.
    """
    area_total = pcr.areatotal(pcr.scalar(values), pcr.nominal(area_class))
    total_value, _ = pcr.cellvalue(pcr.mapmaximum(area_total), 1, 1)
    if total_value <= -3.40282346638e+38: # return 0 if only missing values
        total_value = 0
    return total_value


#%% create cost classes
class CostTypes(object):
    """Generic attributes for all cost types.
    Cost types are:
        * Vegetation removal
        * Earthwork on minor embankments and terrain height
        * Acquisition of land and buildings including demolition
        * Raising embankment by 50 or 100 cm
        * Relocating embankment"""

    def __init__(self):
        self.name = 'TerribleLongDefaultNameWithoutAnyUseAndToBeRemoved'


class CostPreparation(CostTypes):
    """
    Class for cost calculation for preparation of the project area.
    """
    def __init__(self,
                 land_acq_distr, roads_distr, smoothing_distr, buildings_distr,
                 smoothing_classes):
        CostTypes.__init__(self)
        self.land_acq_distr = land_acq_distr
        self.roads_distr = roads_distr
        self.smoothing_distr = smoothing_distr
        self.buildings_distr = buildings_distr
        self.smoothing_classes = smoothing_classes

    def land_acquisition(self, area):
        """Calculate the cost and st.dev of land acquisition."""
        cost = area_total_value(self.land_acq_distr.mean, area)
        std = area_total_value(self.land_acq_distr.stddev, area)
        return cost, std

    def road_removal(self, area):
        """Calculate costs and st.dev of road removal."""
        cost = area_total_value(self.roads_distr.mean, area)
        std = area_total_value(self.roads_distr.stddev, area)
        return cost, std

    def forest_removal(self, area):
        """Calculate costs and st.dev of forest removal."""
        forested_area = self.smoothing_classes == 3
        removal_area = pcr.ifthen((pcr.defined(area) & forested_area),
                                  pcr.boolean(1))
        cost = area_total_value(self.smoothing_distr.mean, removal_area)
        std = area_total_value(self.smoothing_distr.stddev, removal_area)
        return cost, std

    def buildings(self, area):
        """Calculate cost and st.dev of building acquisition and demolition."""
        cost = area_total_value(self.buildings_distr.mean, area)
        std = area_total_value(self.buildings_distr.stddev, area)
        return cost, std

    def overview(self, msr):
        """Give an overview of the cost and st.dev in separate DataFrames.
        The row index is the name of the measure, the columns the cost type."""
        msr_type = msr.settings.loc['msr_type',1]
        name =  msr_type + '_' + msr.settings.loc['ID']

        if msr_type in ['lowering', 'sidechannel', 'relocation']:
            land_acq_values = self.land_acquisition(msr.area)
            road_values = self.road_removal(msr.area)
            forest_values = self.forest_removal(msr.area)
            building_values = self.buildings(msr.area)
        else:
            land_acq_values = [0,0]
            road_values = [0,0]
            forest_values = [0,0]
            building_values = [0,0]

        cost_data = {'land_acq': land_acq_values[0],
                     'road_rem': road_values[0],
                     'forest_rem': forest_values[0],
                     'building_rem': building_values[0]}
        cost_df = pd.DataFrame(data=cost_data, index=[name])

        std_data = {'land_acq': land_acq_values[1],
                     'road_rem': road_values[1],
                     'forest_rem': forest_values[1],
                     'building_rem': building_values[1]}
        std_df = pd.DataFrame(data=std_data, index=[name])
        return cost_df, std_df


class CostEarthwork(CostTypes):
    """Cost for earthwork on the DEM.
    Minor embankments are completely removed and the DEM is lowered, as is the
    case for floodplain lowering and side channel recreation. Cost depends on
    the pollution of the soil. """

    def __init__(self, \
               minemb_ext_use, minemb_int_use, minemb_polluted,\
               dem, minemb, groyne,
               pollution_zones, cost_input):
        CostTypes.__init__(self)
        self.minemb_ext_use = minemb_ext_use        # distribution in euro/cell
        self.minemb_int_use = minemb_int_use
        self.minemb_polluted = minemb_polluted

        self.ref_dem = dem                                # in m +OD
        self.minemb = minemb
        self.groyne = groyne

        self.pollution_zones = pollution_zones            # ordinal 1-6
        self.cost_input = cost_input

    def dem_lowering(self, msr, area_polluted):
        """Calculate the cost and standard deviation of lowering the DEM."""

        if msr.settings.loc['msr_type',1] in ['sidechannel', 'lowering']:
            # Determine the total volumetric difference over the measure area
            # Assume that polluted area is < 0.5 m deep, see doc and Middelkoop.
            dh_tot = self.ref_dem - msr.dem
            dh_polluted = pcr.ifthen(area_polluted, pcr.min(0.5, dh_tot))
#            pcr.aguila(dh_polluted)
            dV_tot = dh_tot * pcr.cellarea()
            dV_polluted = dh_polluted * pcr.cellarea()
            volume_tot = area_total_value(dV_tot, msr.area)
            volume_polluted = area_total_value(dV_polluted, area_polluted)
        else:
            volume_tot = 0
            volume_polluted = 0
        # Return cost and standard deviation as a dataframe
        cost_ew = self.cost_input.iloc[0:3,0:2]
        cost_init = volume_tot * cost_ew.drop('flpl_low_polluted')
        cost_polluted = volume_polluted * cost_ew.loc['flpl_low_polluted',:]
        cost_out = cost_init.copy()
        cost_out = cost_out.append(cost_polluted)
        return cost_out

    def minemb_lowering(self, msr, area_polluted):
        """Calculate the cost and std. dev. of removing minor embankment.
        The pollution threshold defines which classes of self.pollution_zones
        are classified as polluted. Here additional costs are calculated."""

        if msr.settings.loc['msr_type',1] in ['sidechannel', 'lowering']:
            volume_tot = area_total_value(self.minemb.volume, msr.area)
            volume_polluted = area_total_value(self.minemb.volume, area_polluted)
        elif msr.settings.loc['msr_type',1] == 'minemblowering':
            delta_height = self.minemb.height - msr.minemb_height
            delta_height = pcr.max(0, delta_height)
            delta_volume = 7 * delta_height * self.minemb.length
            volume_tot = area_total_value(delta_volume, msr.area)
            volume_polluted = area_total_value(delta_volume, area_polluted)
        else:
            volume_tot = 0
            volume_polluted = 0

        cost_data = self.cost_input.iloc[6:,:2]
        cost_clean = volume_tot * cost_data.drop('minemb_polluted')
        cost_polluted = volume_polluted * cost_data.loc['minemb_polluted']
        cost_out = cost_clean.append(cost_polluted)
        return cost_out

    def groyne_lowering(self, msr):
        """Cost for earthworks for lowering of the groynes based on the
        length of the existing groynes.
        """
        if msr.settings.loc['msr_type',1] == 'groynelowering':
            groyne_lowering_cost = 650   # euro/m
            groyne_lowering_std = 170    # euro/m
            cost_per_cell = groyne_lowering_cost * self.groyne.length
            std_per_cell = groyne_lowering_std * self.groyne.length
            cost_total = area_total_value(cost_per_cell, msr.area)
            std_total = area_total_value(std_per_cell, msr.area)
        else:
            cost_total = 0
            std_total = 0
        out = pd.DataFrame(data=[[cost_total, std_total]],
                           index=[['groyne_low']],
                           columns=['cost', 'stddev'])
        return out

    def overview(self, msr, pollution_threshold=2):
        """Give the overview of costs and st.dev for dem and minor embankments.
        """
        msr_type = msr.settings.loc['msr_type',1]
        name =  msr_type + '_' + msr.settings.loc['ID']

        # Separate between clean and polluted areas
        area_clean = pcr.ifthen(self.pollution_zones >= pollution_threshold,\
                                pcr.nominal(1))
        area_polluted = pcr.ifthen(self.pollution_zones < pollution_threshold,\
                                   pcr.nominal(1))

        area_clean = pcr.defined(msr.area) & pcr.defined(area_clean)
        area_clean = pcr.ifthen(area_clean,  pcr.boolean(1))
        area_polluted = pcr.defined(msr.area) &\
                                    pcr.defined(area_polluted)
        area_polluted = pcr.ifthen(area_polluted, pcr.boolean(1))

        # Calculate costs and stddev for all earthwork types.
        flpl_low_values = self.dem_lowering(msr, area_polluted)
        minemb_low_values = self.minemb_lowering(msr, area_polluted)
        groyne_lowering_values = self.groyne_lowering(msr)
        cost_ew = pd.concat([flpl_low_values, groyne_lowering_values,
                             minemb_low_values])

        cost_df = cost_ew.iloc[:,0:1].T
        cost_df.index = [name]
        std_df = cost_ew.iloc[:,1:2].T
        std_df.index = [name]
        return cost_df, std_df


class CostSmoothing(CostTypes):
    """Cost for lowering the floodplain roughness, i.e. removal
    of existing vegetation."""
    def __init__(self, smoothing_distr):
        CostTypes.__init__(self)

        # cost distribution input
        self.smoothing_distr = smoothing_distr

    def smoothing_value(self, msr_area):
        """Sum up the cost and std of vegetation removal over the measure area.
        Returns a float."""
        cost = area_total_value(self.smoothing_distr.mean, msr_area)
        std = area_total_value(self.smoothing_distr.stddev, msr_area)

        return cost, std

    def overview(self, msr):
        """Give an overview of the cost and st.dev of smoothing."""
        if msr.settings.loc['msr_type',1] == 'smoothing':
            cost, std = self.smoothing_value(msr.area)
        else:
            cost, std = 0, 0
        name = msr.settings.loc['msr_type',1] + '_' + msr.settings.loc['ID']
        cost_df = pd.DataFrame(data={'smoothing': cost}, index=[name])
        std_df = pd.DataFrame(data={'smoothing': std}, index=[name])

        return cost_df, std_df

class CostDikeRelocation(CostTypes):
    """Calculate the cost of embankment relocation over the measure area."""
    def __init__(self,\
                 dike_reloc_distr, dike_length):
        CostTypes.__init__(self)
        self.dike_reloc_distr = dike_reloc_distr
        self.dike_length = dike_length

    def relocation(self, reloc_area):
        """Sum up the cost and st.dev of dike relocation."""
        reloc_area = pcr.cover(pcr.boolean(reloc_area), pcr.boolean(0))
        buffer_m = 1.5 * pcr.clone().cellSize() # buffer 1 cell in 8 directions
        reloc_buffered = pcr.spreadmaxzone(reloc_area, 0, 1, buffer_m)
        reloc_length = pcr.ifthen(reloc_buffered, self.dike_length)
        cost_spatial = 0.001 * reloc_length * self.dike_reloc_distr.mean
        std_spatial = 0.001 * reloc_length * self.dike_reloc_distr.stddev

        area = pcr.ifthen(reloc_area, pcr.boolean(1))
        cost = area_total_value(cost_spatial, area)
        std = area_total_value(std_spatial, area)

        return cost, std

    def overview(self, msr):
        """Give an overview of cost and standard deviation of dike relocation.
        """
        if msr.settings.loc['msr_type',1] == 'relocation':
            cost, std = self.relocation(msr.area)
        else:
            cost, std = 0, 0
        name = msr.settings.loc['msr_type',1] + '_' + msr.settings.loc['ID']
        cost_df = pd.DataFrame(data={'relocation': cost}, index=[name])
        std_df = pd.DataFrame(data={'relocation': std}, index=[name])
        return cost_df, std_df

class CostDikeRaising(CostTypes):
    """Calculate the cost of raising the embankment over 50 and 100 cm."""
    def __init__(self,
                 dike_raise50_distr, dike_raise100_distr,
                 dike_length):
        #self.name = name
        self.dike_raise50_distr = dike_raise50_distr    # units: Euro/km
        self.dike_raise100_distr = dike_raise100_distr
        self.dike_length = dike_length

    def raising(self, raise_area):
        """Sum up the cost and its st.dev for raising embankments."""
        raise_length = pcr.ifthen(raise_area, self.dike_length)
        cost_spatial_50cm = 0.001 * raise_length * self.dike_raise50_distr.mean
        cost_spatial_100cm = 0.001 * raise_length * self.dike_raise100_distr.mean
        std_spatial_50cm = 0.001 * raise_length * self.dike_raise50_distr.stddev
        std_spatial_100cm = 0.001 * raise_length * self.dike_raise100_distr.stddev

        cost_list =  [area_total_value(cost_spatial_50cm, raise_area),
                      area_total_value(cost_spatial_100cm, raise_area)]
        std_list =  [area_total_value(std_spatial_50cm, raise_area),
                      area_total_value(std_spatial_100cm, raise_area)]
        return cost_list, std_list

    def overview(self, msr):
        """Give an overview of the cost and standard deviation of dike raising.
        """
        msr_type = msr.settings.loc['msr_type',1]
        name = msr_type + '_' + msr.settings.loc['ID']

        if msr_type == 'dikeraising':
            cost, std = self.raising(msr.area)
        else:
            cost, std = [0, 0], [0, 0]

        cost_df = pd.DataFrame(data={'raise_50cm': cost[0],
                                     'raise_100cm': cost[1]},
                               index=[name])
        std_df = pd.DataFrame(data={'raise_50cm': std[0],
                                    'raise_100cm': std[1]},
                              index=[name])
        return cost_df, std_df


class CostDistribution(object):
    """Generic class for the spatial distribution of cost attributes.
    Attributes are the mean and standard deviation of the cost/sttdev per cell.
    """
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def plot(self, std_dev=None):
        #pcr.aguila(self.mean, self.stddev)

        if std_dev is None:
          return ipynb_utils.plot(self.mean, 'Mean values')
        else:
          return (ipynb_utils.plot(self.mean, 'Mean values') + ipynb_utils.plot(self.stddev, 'Standard deviation')).cols(1)


def read_distribution(input_dir, cost_name):
    """Read pcr_maps with mean and stddev of 'cost_name' into a distribution.

    Input:
    input_dir: path to the directory with the PCRaster maps
    cost_name: name of the cost attribute

    Naming convention:
    mean: 'cost_%s.map' % cost_name
    stddev: 'std_%s.map' % cost_name

    Returns a CostDistribution class
    """
    mean = pcr.readmap(os.path.join(input_dir, 'cost_%s.map' % cost_name))
    stddev = pcr.readmap(os.path.join(input_dir, 'std_%s.map' % cost_name))
    return CostDistribution(mean, stddev)

def assess_effects(cost_types, maps_dir, cost_correction):
    """
    Determine all effects for all measures.
    """
    measure_names = [f for f in os.listdir(maps_dir)
                       if os.path.isdir(os.path.join(maps_dir,f))]

    cost_list = []
    std_list = []
    for msr_name in measure_names[:]:
        measure_pathname = os.path.join(maps_dir, msr_name)
        measure = read_measure(measure_pathname)
        cost_all_types, std_all_types = assess_costs(cost_types, measure)
        cost_list.append(cost_all_types)
        std_list.append(std_all_types)

    cost_all_msrs = pd.concat(cost_list)
    std_all_msrs = pd.concat(std_list)
    cost_all_msrs.index.name = 'id'
    std_all_msrs.index.name = 'id'
    return cost_all_msrs, std_all_msrs

def assess_costs(cost_types, measure):
    "Calculate all the cost types for a single measure."""
    cost_list = []
    std_list = []
    for cost_type in cost_types:
        cost_df, std_df = cost_type.overview(measure)
        cost_list.append(cost_df)
        std_list.append(std_df)
    cost_df = pd.concat(cost_list, axis=1)
    std_df = pd.concat(std_list, axis=1)
    return cost_df, std_df







if __name__ == '__main__':
    #%% integrated assessment
    # Directory settings
    # Waal XL


    # /home/schmi109/development/projects/RiverScape/scripts/measures_ensemble/

    root_dir = os.path.dirname(os.getcwd())
    ref_map_dir = os.path.join(root_dir, r'input_files/input/reference_maps')
    msr_map_dir = os.path.join(root_dir, r'scripts/measures_ensemble/maps')
    input_dir = os.path.join(root_dir, 'input_files/input')
    cost_dir = os.path.join(root_dir, r'input_files/input/cost')
    scratch_dir = os.path.join(root_dir, 'scripts/scratch')
    pcr.setglobaloption('unittrue')
    os.chdir(scratch_dir)

    # Read input maps
    pcr.setclone(os.path.join(ref_map_dir, 'clone.map'))
    depthToSand = pcr.readmap(os.path.join(cost_dir, 'depthToSand.map'))
    dike_length = pcr.readmap(os.path.join(cost_dir, 'dike_length.map'))
    dike_raise_sections = pcr.readmap(os.path.join(cost_dir,
                                                   'dike_raise_sections.map'))
    pollution_zones  = pcr.readmap(os.path.join(cost_dir,
                                                'pollution_zones.map'))
    smoothing_cost_classes = pcr.readmap(os.path.join(cost_dir,
                                                     'smoothing_cost_classes.map'))
    dem = pcr.readmap(os.path.join(ref_map_dir, 'dem.map'))

    # Read input distributions
    road_distr = read_distribution(cost_dir, 'roads')
    smoothing_distr = read_distribution(cost_dir, 'smoothing')
    building_distr = read_distribution(cost_dir, 'building_tot')
    dike_raise50_distr = read_distribution(cost_dir, 'dike_raise50')
    dike_raise100_distr = read_distribution(cost_dir, 'dike_raise100')
    dike_reloc_distr = read_distribution(cost_dir, 'dike_reloc')
    land_acq_distr = read_distribution(cost_dir, 'land_acq')
    minemb_ext_use_distr = read_distribution(cost_dir, 'minemb_ext_use')
    minemb_int_use_distr = read_distribution(cost_dir, 'minemb_int_use')
    minemb_polluted_distr = read_distribution(cost_dir, 'minemb_polluted')

    groyne = read_dike_maps(ref_map_dir, 'groyne')
    minemb = read_dike_maps(ref_map_dir, 'minemb')
    main_dike = read_dike_maps(ref_map_dir, 'main_dike')

    # Set earthwork values
    cost_input_ew = pd.read_csv(os.path.join(cost_dir, 'cost_ew.csv'), index_col=0)
    cost_correction = pd.read_csv(os.path.join(cost_dir, 'cost_correction.csv'),
                                  index_col=0, comment='#')

    ### Smoothing
    c_sm = CostSmoothing('dummy', smoothing_distr)
    sm_cost, sm_std = c_sm.overview(msr)
    if printing == True:
        print(sm_cost)
        print(sm_std)

    ### Earth work
    c_ew = CostEarthwork(label,
                 minemb_ext_use_distr, minemb_int_use_distr, minemb_polluted_distr,
                 dem, minemb, groyne,
                 pollution_zones,
                 cost_input_ew)
    ew_cost, ew_std = c_ew.overview(msr, pollution_threshold=2)
    if printing == True:
        print(ew_cost)
        print(ew_std)

    ### Land preparation
    c_lp = CostPreparation(label,
                           land_acq_distr, road_distr, smoothing_distr,
                           building_distr, smoothing_cost_classes)

    lp_cost, lp_std = c_lp.overview(msr)
    if printing == True:
        print(lp_cost)
        print(lp_std)

    ### Dike relocation
    c_dreloc = CostDikeRelocation(label, dike_reloc_distr, dike_length)
    dr_cost, dr_std = c_dreloc.overview(msr)
    if printing == True:
        print(dr_cost)
        print(dr_std)

    ### Dike raising
    c_draise = CostDikeRaising(label,
                               dike_raise50_distr, dike_raise100_distr,
                               dike_length)
    draise_cost, draise_std = c_draise.overview(msr)
    if printing == True:
        print(draise_cost)
        print(draise_std)

    #%% evaluate all measures on cost
    cost_types = [c_sm, c_ew, c_lp, c_dreloc, c_draise]
    cost_all_msrs, std_all_msrs = assess_effects(cost_types, msr_map_dir, cost_correction)
    cost_all_msrs.to_csv('tmp_cost_all.csv')
    std_all_msrs.to_csv('tmp_std_all.csv')
    print(cost_all_msrs)
    print(std_all_msrs)


    #%% Visualization example
    #####import matplotlib.pyplot as plt
    #####side_channel = cost_all_msrs.loc['sidechannel_custom_label',:]
    #####side_channel.plot.bar(rot=90)
    #####plt.subplots_adjust(bottom=0.35)
    #####plt.ylabel('Euro')
    #####plt.savefig('cost_sidechannel_example.png', dpi=300)
    #####side_channel.to_csv('cost_sidechannel_example.csv')
