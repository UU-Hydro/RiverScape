# Evaluation of implementation costs


## Overview

Costs of particular interventions can be mainly grouped into two types:
non-recurring costs (e.g. construction works) accruing at measure implementation, and recurring costs (e.g. yearly maintenance) as follow-up costs.
This notebook provides the cost evaluation of a set of measures.




## How to start


### Setting up the environment


As before, first import the RiverScape and additionally required Python modules to run this notebook.



``` code
# Standard and scientific modules
import numpy
import pandas
import os

# Visualisation
import geoviews
geoviews.extension('bokeh')

# Modelling
import pcraster
import riverscape
```





### Specify the input data location

The evaluation of affected landowners is performed on a set of measures.
You can use the default example data measures to continue this notebook:


``` code
measure_dir = riverscape.example_data_path()
```


Cost evaluation is performed on a set of measures that you defined in the intervention planning notebook.
Previously, you stored a set of measures to disk.
Now choose the directory holding the measures that you want to evaluate, select the ```maps``` folder:


``` code
# measure_dir = riverscape.select_directory()
```





## Stepwise calculation of the costs

To rate whether a measure is financially feasible or not, it is relevant to have an estimate on the potential investment costs.
These costs usually include costs for earthwork, treatment of polluted soil, dike raising, groyne lowering,
and acquisition or demolition cost of buildings and land.
All these different cost types use various input sources as basis for an cost estimate.


## Spatial datasets of cost items

The cost calculation of interventions also depends on various input datasets,
such as location and types of infrastructure and buildings, or pollution zones.
The spatial distribution and different types of costs contribute to the overall costs of a measure.

You you need to specify the paths to the required input directories:

``` code
root_dir = os.path.dirname(os.getcwd())
ref_map_dir = os.path.join(root_dir, 'input_files', 'input', 'reference_maps')
input_dir = os.path.join(root_dir, 'input_files', 'input')
cost_dir = os.path.join(root_dir, 'input_files', 'input', 'cost')
```

With the given paths to the input directories you can read all the datasets required for the cost calculation.
Costs for the measures are thereby calculated using different types of source data.

One type of sources that you will firstly read are spatial datasets showing the spatial distribution of certain costs.
You can just read them from disk and assigned them to variables that you can further use in this notebook:


``` code
# Define the area mask
pcraster.setclone(os.path.join(ref_map_dir, 'clone.map'))

# Read input maps
depthToSand = pcraster.readmap(os.path.join(cost_dir, 'depthToSand.map'))
dike_length = pcraster.readmap(os.path.join(cost_dir, 'dike_length.map'))
dike_raise_sections = pcraster.readmap(os.path.join(cost_dir, 'dike_raise_sections.map'))
pollution_zones  = pcraster.readmap(os.path.join(cost_dir, 'pollution_zones.map'))
smoothing_cost_classes = pcraster.readmap(os.path.join(cost_dir, 'smoothing_cost_classes.map'))
dem = pcraster.readmap(os.path.join(ref_map_dir, 'dem.map'))

groyne = riverscape.read_dike_maps(ref_map_dir, 'groyne')
minemb = riverscape.read_dike_maps(ref_map_dir, 'minemb')
main_dike = riverscape.read_dike_maps(ref_map_dir, 'main_dike')
```







You can visualize these datasets and browse the costs per cell.
Maps can be plotted with by passing the variables to the map plotting function of the riverscape module, e.g.

``` python
riverscape.plot(pollution_zones)
```

``` code
```

Some spatial datasets are also provided in pairs holding two maps, the mean values and the standard deviation values of particular cost items.
You can read them as input distributions and assign them to objects by:


``` code
# Read input distributions
road_distr = riverscape.read_distribution(cost_dir, 'roads')
smoothing_distr = riverscape.read_distribution(cost_dir, 'smoothing')
building_distr = riverscape.read_distribution(cost_dir, 'building_tot')
dike_raise50_distr = riverscape.read_distribution(cost_dir, 'dike_raise50')
dike_raise100_distr = riverscape.read_distribution(cost_dir, 'dike_raise100')
dike_reloc_distr = riverscape.read_distribution(cost_dir, 'dike_reloc')
land_acq_distr = riverscape.read_distribution(cost_dir, 'land_acq')
minemb_ext_use_distr = riverscape.read_distribution(cost_dir, 'minemb_ext_use')
minemb_int_use_distr = riverscape.read_distribution(cost_dir, 'minemb_int_use')
minemb_polluted_distr = riverscape.read_distribution(cost_dir, 'minemb_polluted')
```


You can plot these, e.g. for the roads, with the plot method of the individual objects.
Most likely your main interest is only on the mean values, in that case just use plot directly.
In case you are interested in the values showing the standard deviation of a cost item, indicate that by passing an additional argument:

``` python
# Show the mean values only
road_distr.plot()

# Use this to additionally plot the standard deviation as well
road_distr.plot(std_dev=True)

```

``` code
```


## Categorical description of cost items


Another specification of costs is given by categories, such as the costs for earth work or removal of roads and bridges.
In case you are interested in the implementation price per unit you can load and display the full table:

``` code
# Reading the input file
costs = pandas.read_csv(os.path.join(cost_dir, 'implementation_costs.csv'))

# Formatting output
costs = costs.replace(numpy.nan, '', regex=True)
dfStyler = costs.style.set_properties(**{'text-align': 'left'})
dfStyler.hide_index()
dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
```




You also need to read costs for earthwork:


``` code
cost_input_ew = pandas.read_csv(os.path.join(cost_dir, 'cost_ew.csv'), index_col=0)
cost_correction = pandas.read_csv(os.path.join(cost_dir, 'cost_correction.csv'), index_col=0, comment='#')
```



## Specifying the costs per cost item



Each cost calculation is implemented in a separate Python class, you first need to import them from the RiverScape module:


``` code
from riverscape import CostSmoothing, CostEarthwork, CostPreparation, CostDikeRelocation, CostDikeRaising
```

The maps holding the spatially distributed cost values are passed per type to the corresponding classes and furtheron used to calculate the total costs.

### Smoothing costs

These costs comprise the removal of existing vegetation to lower the floodplain roughness.

``` code
c_sm = CostSmoothing(smoothing_distr)
```


### Earthwork costs

The costs for earthwork consist for example of excavation costs for floodplain lowering and side channel recreation, the removal of polluted soil and type of pollution, and costs for the lowering of the groynes.


``` code
c_ew = CostEarthwork(minemb_ext_use_distr, minemb_int_use_distr, minemb_polluted_distr,
    dem, minemb, groyne, pollution_zones, cost_input_ew)
```

### Land preparation

Several costs may accrue ahead of an implementation of a measure.
Examples are costs for the acquisition of land and houses, the modification of traffic infrastructure, the removal of forrested areas, or the demolition of buildings.

``` code
c_lp = CostPreparation(land_acq_distr, road_distr, smoothing_distr, building_distr, smoothing_cost_classes)
```

### Dike relocation

These costs occur in case embankments need to be relocated and comprise on the area affected and length of the dike to be relocated.

``` code
c_dreloc = CostDikeRelocation(dike_reloc_distr, dike_length)
```


### Dike raising

These type indicate the costs when raising the embankment 50 or 100 centimetres.

``` code
c_draise = CostDikeRaising(dike_raise50_distr, dike_raise100_distr, dike_length)
```


## Calculation of total costs


With the required input data prepared it is now possible to calculate the costs for the measures.
For each cost per cost type, the cost maps are overlayed with the area of a measure and aggregated:

``` code
cost_types = [c_sm, c_ew, c_lp, c_dreloc, c_draise]
cost_all_msrs, std_all_msrs = riverscape.assess_effects(cost_types, measure_dir, cost_correction)
```


After calculation, you can display the calculated costs in tabular form:


``` code
display(cost_all_msrs)
```

In addition, you can visualise the results for the selected measures:


``` code
riverscape.plot_costs(cost_all_msrs)
```

## Saving the output table

In case you want to keep the results you can store them to disk.
The default location is the measure directory, change it if necessary.


``` code
filename = 'cost_all.csv'
path = os.path.join(measure_dir, filename)

cost_all_msrs.to_csv(path)
```

