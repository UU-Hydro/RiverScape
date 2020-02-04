# Evaluating pre-defined measures

## Overview

This notebook gives an additional insight how RiverScape model and scenario outputs can be evaluated.
This demonstration will also include water level changes resulting from a hydrodynamic model [(Delft3D Flexible Mesh)](https://oss.deltares.nl/web/delft3dfm), whose execution is not part of this set of notebooks due to significant computational requirements.
In this notebook we therefore use a set outputs from a preprocessed model.
The measures used in that model run were parameterised on two principles: nature restoration and maximising flood conveyance capacity.
For a detailed explanation of the measures used here we refer to the publication 'Towards multi-objective optimization of large-scale fluvial landscaping measures' by [Straatsma et al. (2019)](https://doi.org/10.5194/nhess-19-1167-2019).




## How to start

### Extract the example data


The required dataset is provided as compressed file *example_measures_waal.zip* located in the *output* directory.
If you have not done so far, extract the contents of the file into the output directory.



### Setting up the environment


As before, import the RiverScape and additional required Python modules to run this notebook:


``` code
# Standard and scientific modules
import numpy
import os
import pandas

# Visualisation
import geoviews
geoviews.extension('bokeh')


# Modelling
import riverscape
```


### Specify the input data location


In case you extracted the example data in the output folder you can just use the following to determine the input path:

``` code
measure_dir = os.path.join('..', 'outputs', 'example_measures_waal', 'maps')
```


Otherwise, uncomment and select the *maps* subdirectory of the example data directory:


``` code
# measure_dir = select_directory()
```

You can inspect the path to the input directory by printing it:

``` code
print(measure_dir)
```





## Evaluation of water level changes

One output that is not computed in this set of notebooks by a model run is the hydrodynamic model due to its computational requirements.
Still, to give an impression about the resulting potential water level changes you can plot the result from a pre-calculated model run.
All measures were evaluated in combination with the hydrodynamic model Delft3D Flexible Mesh.

Executing the following cell will show plots
of level water level changes along the river (in km) per measure, here exemplary for the Q16 design discharge (10165 m3s-1).


``` code
riverscape.plot_water_level_change()
```


## Exploring a set of measures <!--stimation of optimal solutions-->

A broad set of model parameters results in a large parameter space, and measures parameterised differently may result in a large spread of potential outcomes.
To give an better insight into the variation of outcomes we can plot certain variables against each other, for example, the number of involved landowners against the implementation costs.
This way it is possible to indicate the trade-offs of different landscaping measures.

Executing the following cell will create a scatterplot matrix of the major intervention planning criteria.
In the scatterplot matrix, the lower left corners of the subplots indicate an utopian situation, to the upper right the situation gets dystopian.
The Pareto fronts indicate optimal combinations of measure configurations.
PotAll refers to the potential biodiversity score,
the PotAll axes were reversed to synchronise utopia to the lower left.
Rectangles indicate measures of type natural, squares of type smooth.
The reference scenario is indicated by the red diamond.

Hovering the symbols will indicate the name of the underlying measure.




``` code
riverscape.plot_scatter()
```


## Exploring your set of measures

You can also compare effects of your measures with other measures by adding your measure (i.e. position in the state space) to the scatterplot matrix.
Specify your inputs by using values from the previous notebooks or use default values.



``` code
# Measure name
label = 'my_measure'

# Water level lowering
delta_water = -0.20

# Involved stakeholders
nr_stakeholders = 800

# Estimated costs
implementation_costs = 120000000

# Biodiversity score
potAll = 107.3
```
and plot the scatterplot matrix again with your data added. Your measure will show up as golden hexagonal marker.

``` code
riverscape.plot_scatter(label, delta_water, nr_stakeholders, implementation_costs, potAll)
```


In the individual scatterplots, the hovering tool of the plot markers will show the name of a measure.
It is also possible to visualise the spatial datasets underlying a particular measure.
Execute the next code cell and you can select the measure you want to display from the dropdown menu.
Please note that updating the plots may take a moment.

``` code
riverscape.plot_measures(measure_dir)
```


