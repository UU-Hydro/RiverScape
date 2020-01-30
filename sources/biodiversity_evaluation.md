Introduction
============

This notebook facilitates using the RiverScape Python package to evaluate the effects of RiverScape measures on biodivesity using the BIOSAFE model. If this is your first time running the notebook, we advise you to read its reference paper (Straatsma et al., 2017, https://advances.sciencemag.org/content/3/11/e1602762), including its supplementary material (http://advances.sciencemag.org/cgi/content/full/3/11/e1602762/DC1).




```python
# to display all output from a cell
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# see: https://ipython.readthedocs.io/en/stable/config/options/terminal.html#configtrait-InteractiveShell.ast_node_interactivity
```

Requirement (Python modules/packages)
-------------------------------------

To run this notebook, please import the following Python modules.


```python
# import standard modules
import os
import shutil
import sys
import string
import subprocess
import time
import math
import pprint as pp

# import required modules/packages (which may require installation)
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns

import matplotlib.pyplot as plt

import pcraster as pcr
```

Please also make sure that this notebook file is in the same folder as the RiverScape Python module files that must be loaded. You can then import the RiverScape Python module by running the following cell.


```python
from riverscape import *
import geoviews
geoviews.extension('bokeh')
```

Input folder
========================

The input folder is given in the following location.


```python
# set the current directory as the root directory
try:
    if root_dir == None: root_dir = os.getcwd()
except:
    root_dir = os.getcwd()
    
# by default, the input folder is given in the following location that is relative to the root directory (the current directory)
github_input_dir_source = "../input_files/input/bio/"
input_dir_source = os.path.join(root_dir, github_input_dir_source)
```

You may also want to set the input folder, interactively, by running the following cell. 


```python
# go to the above suggested 'input_dir_source'
os.chdir(input_dir_source); input_dir_source = os.getcwd()

# open an interactive window
get_input_dir_source = ""
get_input_dir_source = select_directory()
# - if cancel or not used, we use the above suggested 'input_dir_source'
if get_input_dir_source == "" or get_input_dir_source == "()": get_input_dir_source = input_dir_source

input_dir_source = get_input_dir_source
```


```python
msg = "The input folder is set on   : {}".format(input_dir_source)
print(msg)

# return to the root folder for further processes. 
os.chdir(root_dir)
```

Output (scratch) folder
========================

Please set your output folder. Please adjust if necessary. 


```python
output_folder = "/scratch-shared/edwinhs/riverscape/biosafe_development/test_biosafe_5/"
```

You can also set the output folder, interactively, by running the following cell. 


```python
# open an interactive window
get_output_folder = ""
get_output_folder = select_directory()
# - if cancel or not used, we use the above defined 'output_folder'
if get_output_folder == () or get_output_folder == "": get_output_folder = output_folder

output_folder = get_output_folder
```


```python
msg = "The output folder is set on   : {}".format(output_folder)
print(msg)
```

The files in the input folder will be copied to the output folder.  


```python
if os.path.exists(output_folder): shutil.rmtree(output_folder)
shutil.copytree(input_dir_source, output_folder)

# use the output folder as the working directory
os.chdir(output_folder)
```

Reading input files
========================

In this notebook, the Spreadsheet Application For Evaluation of BIOdiversity (BIOSAFE) model (REFERENCES) are applied to calculate biodiversity indices of protected and endangered species that are characteristic of the fluvial environment.


```python
# the original excel file
# excelFile = os.path.join(output_folder, 'BIOSAFE_20150629.xlsx')
```


The first input to the spreadsheet is a matrix **legalWeights** (Legislative Weights) of user-specified weights for 16 different policy and legislative documents. The 16 legal documents considered are the most recent versions of: Dutch Red List species that are based on the criteria of the International Union for Conservation of Nature (IUCN), Dutch Flora and Fauna Act, EU Habitat Directive, EU Birds Directive, Bern Convention, and Bonn Convention. We assumed equal weights, attributing equal value to the protection by each of the legal and policy documents. 

Please execute the following cells to load and visualize the **legalWeights** used in this study. 


```python
legalWeights = pd.read_csv(os.path.join(output_folder, 'legalWeights.csv'), index_col = 0)
```


```python
bsIO.show_full_data_frame(legalWeights)
```

The second input is **linksLaw** (Legal Links), which is a relational matrix that links species(s) along the rows to the relevant policy and legislative documents (**legalWeights**) along the columns. If a species is protected by or mentioned in a specific law, the corresponding cell in the matrix is given a value of one, else the cell is zero.

Please execute the following cells to load and visualize the **linksLaw** used in this study. 


```python
linksLaw = pd.read_csv(os.path.join(output_folder, 'linksLaw.csv'), index_col = 0)
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(linksLaw)

#TODO: Formating 0 and 1 without decimal.
```

The third input is **linksEco** (Ecotope Links), which is the relational matrix between species and ecotopes. The matrix is based on expert ecological judgement of species and their hydromorphological and vegetation habitat needs in different life stages. 

Please execute the following cells to load and visualize the **linksEco** used in this study.


```python
linksEco = pd.read_csv(os.path.join(output_folder, 'linksEco.csv'), index_col = 0)
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(linksEco)
```

The column titles in linksEco represent the ecotope classes used in this study. In addition to them, we introduce their merged/main classes based on the following relationship matrix **lut** that can be loaded and visualized by executing the following notebook cells.     


```python
excelFile = os.path.join(output_folder, 'BIOSAFE_20150629.xlsx')
lut = pd.read_excel(excelFile, sheet_name = 'lut_RWES').fillna(method='ffill')
```


```python
bsIO.show_full_data_frame(lut)
```

In the matrix **lut**, the column oldEcotope represents the ecotope classes given along the top rows of linksEco, while the classes in the column newEcotope are their merged ecotope classes. Based on **lut** (relation between oldEcotope and newEcotope) and by running the following notebook cells, we can derive and visualize the Ecotope Links **linksNewEcoAgg**, consisting oldEcotope and newEcotope. 


```python
linksNewEcoAgg = biosafe.aggregateEcotopes(linksEco, lut)
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(linksNewEcoAgg)
```


```python
print(linksEco.shape)
print(linksNewEcoAgg.shape)
```

Understanding/testing BIOSAFE in a simple single instance (non-spatial mode)
========================

To understand the BIOSAFE model, we introduce the following exercise running the BIOSAFE model in a simple single instance.   

For this exercise, we randomly draw the presence of each species by running the following cells. The assumption drawn is given in the table, particularly in the column **speciesPresence**, with the values 0 and 1 indicating absent and present.  


```python
speciesPresence = pd.DataFrame(np.random.randint(2, size=len(linksLaw)),\
                    columns=['speciesPresence'], \
                    index=linksLaw.index)
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(speciesPresence)
```


```python
# TODO: Menno may want to reduce a number of species. 
```

For this exercise, we simplify all areas of ecotope classes (**ecotopeArea**) equal to 1e5 m2. This is done by running the following cell. 


```python
ecotopeArea = pd.DataFrame(np.ones(len(linksNewEcoAgg.columns)-1) * 1e5,\
                           columns = ['area_m2'],\
                           index = linksNewEcoAgg.columns.values[0:-1])
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(ecotopeArea)
```

Running the BIOSAFE in a simple single instance (non-spatial mode)
========================

Given the inputs **legalWeights**, **linksLaw** and **linksNewEcoAgg**, as well as the assumptions **speciesPresence** and **ecotopeArea** defined above, the BIOSAFE model is initiated by running the following cell.   


```python
bs = biosafe.biosafe(legalWeights, linksLaw, linksNewEcoAgg, speciesPresence, ecotopeArea)
```

Then, we can calculate all BIOSAFE scores (potential and actual ones).


```python
SScoresSpecies = bs.SScoresSpecies()
summarySScores = bs.taxGroupSums()
SEcoPot = bs.SEcoPot()
SEcoAct = bs.SEcoAct()
PTE = bs.PTE()
ATE = bs.ATE()
TES = bs.TES()
TEI = bs.TEI()
ATEI = bs.ATEI()
TFI = bs.TFI()
FI = bs.FI()
ATFI = bs.ATFI()
AFI = bs.AFI()
FIS = bs.FIS()
TFIS = bs.TFIS()
TFHS = bs.TFHS()
FTEI = bs.FTEI()
```

For each species, the calculated potential and actual species scores, Spotential and Sactual (PSS and ASS in the paper), are given as follows. 


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(SScoresSpecies)
```

For each taxonomic group, the potential and actual taxonomic group scores, PTB and ATB, can be calculated as follows. In addition, the taxonomic group biodiversity, TBS, saturation can also be calculated. 


```python
bsIO.show_full_data_frame(summarySScores)
```

For each ecotope class, we can also compute the potentil and actual species specific S scores for each ecotope. Their values for each species and ecotope areas are given in the following two cells, the first one shows their potential scores, SEcoPot, while the latter contains their actual scores, SEcoAct. 


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(SEcoPot)
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(SEcoAct)
```

For each taxonomic group and for each ecotope class, the sum of SEcoPot can be calculated as PTE, potential taxonomic group ecotope constant. 


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(PTE)
```

For each taxonomic group and for each ecotope class, the sum of SEcoAct can be calculated as ATE, actual taxonomic group ecotope constant.  


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
bsIO.show_full_data_frame(ATE)
```

Based on ATE and PTE, we can calculate TES, taxonomic group ecotope saturation index, for each taxonomic group and ecotope area.


```python
bsIO.show_full_data_frame(TES)
```

Based on PTB and PTE, we can calculate TEI, taxonomic group ecotope importance constant, for each taxonomic group and ecotope area.


```python
bsIO.show_full_data_frame(TEI)
```

Based on TEI and TES, we can calculate ATEI (ATE in the paper), actual taxonomic group ecotope constant, for each taxonomic group and ecotope area.


```python
bsIO.show_full_data_frame(ATEI)
```

Based on the ecotopeArea assumed above and TEI, the potential taxonomic group flood plain importance score, TFI (PotTax in the paper), can be calculated as follows.  


```python
bsIO.show_full_data_frame(TFI)
```

The sum of TFI is reported as FI (PotAll), the potential floodplain importance score.


```python
bsIO.show_full_data_frame(FI)
```

Based on the ecotopeArea assumed above and ATEI, the actual taxonomic group flood plain importance score, ATFI (ActTax in the paper), can be calculated as follows.  


```python
bsIO.show_full_data_frame(ATFI)
```

The sum of ATFI is reported as AFI (ActAll), the actual floodplain importance score.


```python
bsIO.show_full_data_frame(AFI)
```

Based on AFI and FI, the floodplain importance saturation, FIS (SatAll in the paper), can be calculated as follows. 


```python
bsIO.show_full_data_frame(FIS)
```

For each taxonomic group, the taxonomic group floodplain importance saturation, TFIS (SatTax in the paper), can be calculated based on ATFI and FIS. TFIS describes the fraction/saturation of actual over potential biodiversity. 


```python
bsIO.show_full_data_frame(TFIS)
```

Based on TEI and ecotopeArea, we can calculate TFHS (HabDiv in the paper), the taxonomic group floodplain habitat saturation, which describes the fraction of suitable ecotopes, weighed by law, for each taxonomic group. 


```python
bsIO.show_full_data_frame(TFHS)
```


```python
# bsIO.show_full_data_frame(FTEI)
# FTEI is not found in the paper.
# FTEI: Floodplain Taxonomic group Ecotope Importance
```

Running the BIOSAFE in a spatial mode
========================

In the following cells, we will demonstrate the BIOSAFE model in a spatial model. 

We assume the following inputs of floodplain section map, **flpl_sections**, and ecotope map, **ecotopes**. 


```python
flpl_sections_f = os.path.join(output_folder, 'flpl_sections.map')
pcr.setclone(flpl_sections_f)
flpl_sections = pcr.readmap(flpl_sections_f)
```


```python
plot(flpl_sections)
```


```python
ecotopes = biosafe.read_map_with_legend(os.path.join(output_folder, 'ecotopes.map'))
```


```python
plot(ecotopes.pcr_map)
#TODO: Improving ecotope plotting. 
```

The species presence and characteristics are given in the following database ndff_species. 


```python
ndff_species = pd.read_pickle(os.path.join(output_folder, 'ndff_sub_BS_13.pkl'))
```


```python
# Note: This cell is for visualizing data only and may be heavy. 
# You may want to skip it or clear its output after executing it.
# ndff_species.shape
# bsIO.show_full_data_frame(ndff_species)
```

Given the inputs defined above, we can initiate the BIOSAFE model in a spatial model by running the following cell. Note that the inputs of speciesPresence and ecotopoArea are calculated based on ndff_species and flpl_sections (i.e. the aforementioned speciesPresence and ecotopoArea values are overwritten). For this example, we calculate the parameters FI and TFI. 


```python
bs_spatial = biosafe.spatialBiosafe(bs, ecotopes, flpl_sections, ndff_species,
                            params = ['FI', 'TFI'],
                            toFiles = None)
FI, TFI = bs_spatial.spatial()
#TODO: Remove the print statement '<class 'pcraster._pcraster.Field>'
```

The calculated FI values for all floodplain IDs are given as follows. 


```python
bsIO.show_full_data_frame(FI)
```

The calculated TFI values are given as follows (for each taxonomic group and floodplain section). 


```python
bsIO.show_full_data_frame(TFI)
```

Assess biodiversity scores before and after measure
========================

The following cells are for an exercise to access such a floodplain measure to biodiversity.  

We assume such a measure is implemented in the following ecotope map **msr_eco**. 


```python
msr_eco = biosafe.read_map_with_legend(os.path.join(output_folder, 'ecotopes_msr.map'))
```


```python
plot(msr_eco.pcr_map)
#TODO: Improving ecotope plotting. 
```

Note that this measure is only implemented in some parts of the model area (not for all floodplain IDs). Its reference/existing condition (before the measure), which is taken from them map **ecotopes** map, is given as **ref_eco**. 


```python
msr_area = pcr.defined(msr_eco.pcr_map)
sections = pcr.ifthen(msr_area, pcr.nominal(1))
ref_eco = biosafe.LegendMap(pcr.ifthen(msr_area, ecotopes.pcr_map), msr_eco.legend_df)
```


```python
plot(sections)
```


```python
plot(ref_eco.pcr_map)
#TODO: Improving ecotope plotting. 
```

The BIOSAFE model for the reference/existing condition is calculated as follows. For this exercise, we just calculate the scores FI and TFI. 


```python
bs_ref = biosafe.spatialBiosafe(bs, ref_eco, sections, ndff_species,
                        params = ['FI', 'TFI'], toFiles=None)
FI_ref, TFI_ref = bs_ref.spatial()
```

The BIOSAFE model for the measure condition is calculated as follows.


```python
bs_msr = biosafe.spatialBiosafe(bs, msr_eco, sections, ndff_species,
                        params = ['FI', 'TFI'], toFiles=None)
FI_msr, TFI_msr = bs_msr.spatial()
```

Then, we can compare and plot the TFI scores, before and after the measure.


```python
#%% Visualization
%matplotlib notebook 

sns.set_style("ticks")
# TFI_ref.drop(['xcoor', 'ycoor'], axis=1).plot.bar()
# TFI_msr.drop(['xcoor', 'ycoor'], axis=1).plot.bar()

comparison = pd.concat([TFI_ref, TFI_msr]).drop(['xcoor', 'ycoor'], axis=1)
comparison.index = ['Reference', 'Measures']
comparison.columns = [u'Birds', u'Butterflies', u'Dragon- and damselflies',
                    u'Fish', u'Herpetofauna', u'Higher plants', u'Mammals']
comparison.columns.name = 'Taxonomic group'

comparison.plot.bar(rot=0)


plt.savefig('comparison_biosafe.png', dpi=300)
comparison.to_csv('comparison_biosafe.csv')
```

We can also compare and plot the FI scores, before and after the measure.


```python
#%% Visualization
%matplotlib notebook 
 
sns.set_style("ticks")
# FI_ref.drop(['xcoor', 'ycoor'], axis=1).plot.bar()
# FI_msr.drop(['xcoor', 'ycoor'], axis=1).plot.bar()
 
comparison = pd.concat([FI_ref, FI_msr]).drop(['xcoor', 'ycoor'], axis=1)
comparison.index = ['Reference', 'Measures']
comparison.plot.bar(rot=0)

plt.savefig('comparison_biosafe_FI.png', dpi=300)
comparison.to_csv('comparison_biosafe_FI.csv')
```


```python

```
