# -*- coding: utf-8 -*-
import fiona
import geopandas as gpd

import pandas as pd
import numpy as np
import os
import shutil

import biosafeIO as bsIO
import pcraster as pcr
import mapIO
import pcrRecipes as pcrr
import pickle
import subprocess

from biosafe_py3 import *
        
#%% initiate biosafe 

# Directory settings
root_dir = os.path.dirname(os.getcwd())
# - for testing on eejit
root_dir = "/scratch/depfg/sutan101/test_biosafe_2/"

input_dir  = os.path.join(root_dir, 'inputData')
# - for testing on eejit, use input files from the following folder  
input_dir_source = "/scratch/depfg/hydrowld/river_scape/source/from_menno/riverscape/input/bio/"
if os.path.exists(input_dir): shutil.rmtree(input_dir)
shutil.copytree(input_dir_source, input_dir)

scratch_dir = os.path.join(root_dir, 'scratch')
if os.path.exists(scratch_dir): shutil.rmtree(scratch_dir)
os.makedirs(scratch_dir)
os.chdir(scratch_dir)

# Input data
excelFile = os.path.join(input_dir, 'BIOSAFE_20150629.xlsx')
#bsIO.xlsxToCsv(excelFile, scratch_dir)
legalWeights, linksLaw, linksEco = bsIO.from_csv(input_dir)
speciesPresence = pd.DataFrame(np.random.randint(2, size=len(linksLaw)),\
                    columns=['speciesPresence'], \
                    index=linksLaw.index)
ecotopeArea = pd.DataFrame(np.ones(82) * 1e5,\
                           columns = ['area_m2'],\
                           index = linksEco.columns.values[0:-1])

ndff_species = pd.read_pickle(os.path.join(input_dir, 'ndff_sub_BS_13.pkl'))
flpl_sections_f = os.path.join(input_dir, 'flpl_sections.map')
pcr.setclone(flpl_sections_f)
flpl_sections = pcr.readmap(flpl_sections_f)
ecotopes = read_map_with_legend(os.path.join(input_dir, 'ecotopes.map'))

#%% test a single instance of biosafe
legalWeights, linksLaw, linksEco = bsIO.from_csv(input_dir)
bs = biosafe(legalWeights, linksLaw, linksEco, speciesPresence, ecotopeArea)
lut1 = pd.read_excel(excelFile, sheet_name = 'lut_RWES').fillna(method='ffill')
    # this lookup table has:
    #       ecotope codes of BIOSAFE in the first column: oldEcotope
    #       aggregated/translated ectotopes in the second column: newEcotope
linksEco1 = aggregateEcotopes(linksEco, lut1)
bs.linksEco = linksEco1
ecotopeArea1 = pd.DataFrame(np.ones(len(linksEco1.columns)-1) * 1e5,\
                           columns = ['area_m2'],\
                           index = linksEco1.columns.values[0:-1])
bs.ecotopeArea = ecotopeArea1

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

#bsIO.output2csv(bs, os.path.join(scratch_dir, 'test'))
#bsIO.output2xlsx(bs, 'output1.xlsx')

#%% test biosafe in spatial mode
bsModel = biosafe(legalWeights, linksLaw, linksEco, speciesPresence, ecotopeArea)
bsModel.linksEco = linksEco1
bs_spatial = spatialBiosafe(bsModel, ecotopes, flpl_sections, ndff_species,
                            params = ['FI', 'TFI'],
                            toFiles = None)
FI, TFI = bs_spatial.spatial()

#%% Example for Deltares with a measure
msr_eco = read_map_with_legend(os.path.join(input_dir, 'ecotopes_msr.map'))
msr_area = pcr.defined(msr_eco.pcr_map)
ref_eco = LegendMap(pcr.ifthen(msr_area, ecotopes.pcr_map), msr_eco.legend_df)
sections = pcr.ifthen(msr_area, pcr.nominal(1))

bs_ref = spatialBiosafe(bsModel, ref_eco, sections, ndff_species,
                        params = ['FI', 'TFI'], toFiles=None)
FI_ref, TFI_ref = bs_ref.spatial()

bs_msr = spatialBiosafe(bsModel, msr_eco, sections, ndff_species,
                        params = ['FI', 'TFI'], toFiles=None)
FI_msr, TFI_msr = bs_msr.spatial()


#%% Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("ticks")
TFI_ref.drop(['xcoor', 'ycoor'], axis=1).plot.bar()
TFI_msr.drop(['xcoor', 'ycoor'], axis=1).plot.bar()

comparison = pd.concat([TFI_ref, TFI_msr]).drop(['xcoor', 'ycoor'], axis=1)
comparison.index = ['Reference', 'Measures']
comparison.columns = [u'Birds', u'Butterflies', u'Dragon- and damselflies',
                    u'Fish', u'Herpetofauna', u'Higher plants', u'Mammals']
comparison.columns.name = 'Taxonomic group'
comparison.plot.bar(rot=0)
plt.savefig('comparison_biosafe.png', dpi=300)
comparison.to_csv('comparison_biosafe.csv')

























