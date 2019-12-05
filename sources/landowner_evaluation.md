# Landowner evaluation


This notebook assesses the number and types of land owners involved in the measure extent.
First import the RiverScape and additional required Python modules to run this notebook.


``` code
import pandas
import numpy
from riverscape import *

# Visualisation
import geoviews
geoviews.extension('bokeh')
```

Evaluation is performed on a set of measures that you defined in the intervention planning notebook.
Previously, you stored a set of measures to disk.
Now choose the directory holding the measures that you want to evaluate, select the ```maps``` folder:


``` code
msr_map_dir  = select_directory()
```







