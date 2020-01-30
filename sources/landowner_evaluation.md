# Evaluating the number of affected landowners

## Overview

Decisions for adaptation measures require an overview of stakeholders involved,
an information relevant e.g. for compensation or expropriation of landowners.
The number of landowners can therefore be seen as proxy for governance complexity.
This notebook assesses the number and types of land owners involved in a measure extent.



## How to start


### Setting up the environment

First import the RiverScape and additional required Python modules to run this notebook.


``` code
# Standard modules
import os

# Visualisation
import geoviews
geoviews.extension('bokeh')

# Modelling
import riverscape
```


### Specify the input data location

The evaluation of affected landowners is performed on a set of measures.
You can use the default example data measures to continue this notebook:

``` code
measure_dir = os.path.join(riverscape.example_data_path(), 'sidechannel_evr_natural')
```

<!-- that you defined in the intervention planning notebook. -->

In case you previously stored a set of measures to disk you can use those.
Uncomment and execute the following line and choose the directory holding your measures.
Select a subdirectory of the *maps* folder, such as ``lowering_everywhere``:

<!-- choose a directory holding the measures that you want to evaluate. -->

``` code
# measure_dir = riverscape.select_directory()
```

## Landowners affected by measures

Intended measures in particular areas may affect various stakeholders, such as citizens, companies or governmental institutions.
You can visualise and inspect the areas affected by measures and the corresponding variety of ownerships:

<!--
Example is shown in the figure,
on the left potential areas of a measure are shown,
on the right an impression of the variety of ownerships.-->

``` code
riverscape.plot_landowners(measure_dir)
```


The ownership is furthermore spread over individual owners within the area.
these owners need to participate in the decision process or considerer for imminent expropriation.
It is therefore relevant to know the number of affected stakeholders to estimate the governance complexity of a particular measure.




The following operation returns that number, calculated by an overlay of the areas affected by a measure and the landowner map.
Note that the numbers fairly give an indication about the amount of involved stakeholders.


``` code
affected = riverscape.involved_stakeholders(measure_dir)
```


After formatting you will see the resulting table:



``` code
dfStyler = affected.style.set_properties(**{'text-align': 'left'})
dfStyler.hide_index()
dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])
```



## Saving the output table

Execute the next cell in case you want to store the result.
You can adapt the filename and location, default is the selected directory of measures.


``` code
filename = 'involved_owners.csv'
path = os.path.join(measure_dir, filename)

affected.to_csv(path, index=False)
```
