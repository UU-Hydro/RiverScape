# Cost evaluation


This notebook provides cost evaluation of a set of measures.
First import the RiverScape and additional required Python modules to run this notebook.


``` code
import pandas
import numpy
from riverscape import *

# Visualisation
import geoviews
geoviews.extension('bokeh')
```

Cost evaluation is performed on a set of measures that you defined in the intervention planning notebook.
Previously, you stored a set of measures to disk.
Now choose the directory holding the measures that you want to evaluate, select the ```maps``` folder:


``` code
msr_map_dir  = select_directory()
```

After that you need to specify the paths to the required input directories:

``` code
root_dir = os.path.dirname(os.getcwd())
ref_map_dir = os.path.join(root_dir, 'input_files', 'input', 'reference_maps')
input_dir = os.path.join(root_dir, 'input_files', 'input')
cost_dir = os.path.join(root_dir, 'input_files', 'input', 'cost')
```

With the given paths you can read the input datasets required for the cost calculation:

``` code
# Read input maps
pcr.setclone(os.path.join(ref_map_dir, 'clone.map'))
depthToSand = pcr.readmap(os.path.join(cost_dir, 'depthToSand.map'))
dike_length = pcr.readmap(os.path.join(cost_dir, 'dike_length.map'))
dike_raise_sections = pcr.readmap(os.path.join(cost_dir, 'dike_raise_sections.map'))
pollution_zones  = pcr.readmap(os.path.join(cost_dir, 'pollution_zones.map'))
smoothing_cost_classes = pcr.readmap(os.path.join(cost_dir, 'smoothing_cost_classes.map'))
dem = pcr.readmap(os.path.join(ref_map_dir, 'dem.map'))

groyne = read_dike_maps(ref_map_dir, 'groyne')
minemb = read_dike_maps(ref_map_dir, 'minemb')
main_dike = read_dike_maps(ref_map_dir, 'main_dike')

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

```



Costs for the measures are calculated using different sources.
One type of sources are spatial datasets showing the spatial distribution of certain costs, you just read the from disk.

You can visualize these datasets and browse the costs per cell.
Maps can be plotted with e.g.

``` python
plot(pollution_zones)
```

``` code
```

Datasets read as input distributions hold two maps, the mean values and the standard deviation values.
You can plot these, e.g. for the roads, with

``` python
road_distr.plot()
```

``` code
```

Another specification of costs is given by categories, such as the costs for earth work or removal of roads and bridges.
In case you are interested in the implementation price per unit you can load and display the full table:

``` code
costs = pandas.read_csv(os.path.join(cost_dir, 'implementation_costs.csv'))
costs = costs.replace(numpy.nan, '', regex=True)
dfStyler = costs.style.set_properties(**{'text-align': 'left'})
dfStyler.hide_index()
dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
```




You also need to read costs for earthwork:


``` code
cost_input_ew = pd.read_csv(os.path.join(cost_dir, 'cost_ew.csv'), index_col=0)
cost_correction = pd.read_csv(os.path.join(cost_dir, 'cost_correction.csv'), index_col=0, comment='#')
```

Each cost calculation is implemented in a separate class, you first need to instantiate them:

``` code
c_sm = CostSmoothing('dummylabelCostSmoothing', smoothing_distr)

### Earth work
c_ew = CostEarthwork('dummylabelCostEarthwork', minemb_ext_use_distr, minemb_int_use_distr, minemb_polluted_distr, dem, minemb, groyne, pollution_zones, cost_input_ew)

### Land preparation
c_lp = CostPreparation('dummylabelCostPreparation', land_acq_distr, road_distr, smoothing_distr, building_distr, smoothing_cost_classes)

### Dike relocation
c_dreloc = CostDikeRelocation('dummylabelCostDikeRelocation', dike_reloc_distr, dike_length)

### Dike raising
c_draise = CostDikeRaising('dummylabelCostDikeRaising', dike_raise50_distr, dike_raise100_distr, dike_length)
```
With the required input data prepared it is now possible to calculate the costs for the measures:

``` code
cost_types = [c_sm, c_ew, c_lp, c_dreloc, c_draise]
cost_all_msrs, std_all_msrs = assess_effects(cost_types, msr_map_dir, cost_correction)
```


After calculation, you can display the calculated costs in tabular form:


``` code
display(cost_all_msrs)
```

In addition, you can visualise the results for the selected measures:


``` code
plot_costs(cost_all_msrs)
```

In case you want to keep the results you can store them to disk.


``` code
filename = 'cost_all.csv'
cost_all_msrs.to_csv(filename)
```







