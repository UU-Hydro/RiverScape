# -*- coding: utf-8 -*-
import fiona
import geopandas as gpd

import pandas as pd
import numpy as np
import os
import shutil
import pickle
import subprocess

import pcraster as pcr

from . import biosafeIO as bsIO
from . import pcrRecipes as pcrr
from . import mapIO

version = '0.1'

def zonalSumArea(nominalMap, areaClass):
    """Memory efficient method to sum up the surface area of the different
        classes in the nominal map. Separate by the regions in areaClass

        input:
            nominalMap: nominal map, e.g. ecotope map
            areaClass: regions to compute surface areas over
    """
    #-create a pointMap of the output locations, one for each areaClass
    outputPointMap = pcrr.pointPerClass(areaClass)

    #-iniate output DataFrame
    dfInit = pcrr.getCellValues(outputPointMap, mapList = [areaClass], columns = ['areaClass'])

    #-loop over the classes in nominalMap and compute the summed area per areaClass
    IDs = np.unique(pcr.pcr2numpy(nominalMap, -9999))[1:]
    dfList = []
    for ID in IDs[:]:
        pcrID = pcr.nominal(float(ID))
        pcr.setglobaloption('unittrue')
        IDArea = pcr.ifthen(nominalMap == pcrID, pcr.cellarea())
        sectionSum = pcr.areatotal(IDArea, areaClass)
        df = pcrr.getCellValues(outputPointMap, [sectionSum], [ID])
            # df columns = rowIdx, colIdx, ID
        df = df.drop(['rowIdx', 'colIdx'],  axis=1)
        dfList.append(df)

    dfOut = dfInit.join(dfList)
    #return dfInit, df, dfOut, dfList
    return dfOut

def ndffSpeciesPresence(biosafeSpecies, ndffSpecies, sectionID):
    """Test which biosafe species are present in a specific floodplain section
    and period of time.

    Input
        biosafeSpecies:     df with scientific species name as index and column
        ndffSpecies:        df with name = 'SOORT_WET'
                                    section = 'sections'
        sectionID:          int, relating to floodplain section

    Returns
    -------
    df with 'speciesPresence' column
    """

    presentNDFF = ndffSpecies.query('sections == %s' % (sectionID))
    df = biosafeSpecies.copy()
    df['speciesPresence'] = df.Scientific.isin(presentNDFF.SOORT_WET)+0
    return df.drop('Scientific', axis=1)

def cleanNDFF(dbfFile, biosafeSpecies):
    """Clean up NDFF data to enable usage in spatialTFI
    dbfFile is a dbase file connected to a point shapefile with a sections
    attribute for floodplain linkage.
    """
    df = bsIO.dbf2df(dbfFile)
    df['inBs'] = df.SOORT_WET.isin(biosafeSpecies.Scientific)
    df = df[df.inBs == True]
    df = df[df.PROTOCOL != '17.002 Braakbalonderzoek']
    df = df[df.AREA_M2 != 25000000]
    df['dateMiddle'] = df.apply(lambda x:
                        x.DATM_START + ((x.DATM_STOP - x.DATM_START)/2),axis=1)
    df.drop_duplicates(subset = ['SOORT_WET', 'dateMiddle', 'sections'], inplace=True)
    df['year'] = df.apply(lambda x:
                        x.dateMiddle.year, axis=1)
    df['ecoMapCycle'] = pd.cut(df.year, \
                                bins = [1995, 2001, 2006, 2010, 2015],\
                                labels = [1,2,3,4])
    df.drop_duplicates(subset = ['SOORT_WET', 'ecoMapCycle', 'sections'], inplace=True)
    return df

def aggregateEcotopes(linksEco, lookupTable):
    """Add additional ecotope columns to the linkage of species and ecotopes
    based on the lookup table
    """
    #- filter the lookup table for ecotopes already present in the links df
    bsEcotopes = linksEco.columns.values
    lut = lookupTable.copy()
    lut['inBS'] = lut.oldEcotope.isin(bsEcotopes)
    lut = lut[lut.inBS == True]

    #-loop over the ecotopes to be added and sum up species-ecotope links
    linksEcoNew = linksEco.copy().iloc[:,0:-1]
    newEcotopes = np.unique(lut.newEcotope.values)
    for newEcotope in newEcotopes:
        oldEcotopes = lut[lut.newEcotope == newEcotope]
        ecoLinksNew = linksEco.loc[:,oldEcotopes.oldEcotope.values]
        linksEcoNew[str(newEcotope)] = ecoLinksNew.sum(axis=1).clip(upper=1)
    linksEcoNew['taxonomicGroup'] = linksEco.taxonomicGroup
    return linksEcoNew

def sumUpEcotopeAreaRaster(ecotopes, sections):
    """
    """
    print('\nComputing surface area sum of ecotopes from raster')
    #-prepare ecotope dataframe
    ecoMap = ecotopes.pcr_map
    ecoDf = ecotopes.legend_df.copy()
    ecoDf.set_index('values', inplace = True)

    #- create dataframe of ecotope surface areas per section
    ecotopeSurfaceAreas = zonalSumArea(ecoMap, sections)
    ecotopeSurfaceAreas.set_index('areaClass', inplace = True)
    ecotopeSurfaceAreas.drop(['rowIdx', 'colIdx'], axis = 1, inplace = True)
    ecotopeSurfaceAreas = ecoDf.join(ecotopeSurfaceAreas.drop_duplicates().T)
    ecotopeSurfaceAreas.set_index('ECO_CODE',inplace = True)

    return ecotopeSurfaceAreas.sort_index(axis=0).sort_index(axis=1)

def read_map_with_legend(pcr_file):
    """
    Read map and legend into LegendMap class for nominal or ordinal data.
    The legend needs 'key_label' pairs, separated by an underscore. For example
    '1_UM-1' links map values of 1 to 'UM-1'

    Returns a MapLegend class
    """
    # Read a pcraster legend into a data frame
    cmd = 'legend --nothing -w legend.tmp %s' % pcr_file
    subprocess.call(cmd, shell = True)
    df = pd.read_csv('legend.tmp', sep=' ')
    title = df.columns[1]
    data = {'values':df.iloc[:,0],
             title: df.iloc[:,1].str.split('_', expand=True).iloc[:,1]}
    legend = pd.DataFrame(data=data)

    pcr_map = pcr.readmap(pcr_file)
    return LegendMap(pcr_map, legend)

class spatialBiosafe(object):

    def __init__(self, bsModel, ecotopes, sections, ndffSpecies,
                 params = [], toFiles = ''):
        self.bsModel = bsModel
        self.ecotopes = ecotopes
        self.sections = sections
        self.ndffSpecies = ndffSpecies
        self.params = params

        self.summedAreas = sumUpEcotopeAreaRaster(self.ecotopes, self.sections).fillna(0)
        self.summedAreas.index.name=None
        self.toFiles = toFiles

    def getParam(self, parameter = 'FI', ID = 9999):
        assert parameter in ['FI', 'TFI', 'AFI', 'ATFI', 'TFHS', 'TFIS', 'FIS', 'FTEI']
        if parameter == 'FI':
            df = self.bsModel.FI()
            df.columns.name = parameter
        elif parameter == 'TFI':
            df = self.bsModel.TFI()
            df.columns.name = parameter
        elif parameter == 'AFI':
            df = self.bsModel.AFI()
            df.columns.name = parameter
        elif parameter == 'ATFI':
            df = self.bsModel.ATFI()
            df.columns.name = parameter
        elif parameter == 'TFHS':
            df = self.bsModel.TFHS()
            df.columns.name = parameter
        elif parameter == 'TFIS':
            df = self.bsModel.TFIS()
            df.columns.name = parameter
        elif parameter == 'FIS':
            df = self.bsModel.FIS()
            df.columns.name = parameter
        if parameter == 'FTEI':
            df = self.bsModel.FTEI()
            df['ID'] = ID
            df = df.T

        return df

    def sectionScores(self, sectionID):
        #-initialize ecotope surface areas
        ecoAreas = self.summedAreas.loc[:,[sectionID]]
        ecoAreas.columns = ['area_m2']
        dfEcoAreas = pd.DataFrame(ecoAreas)

        # select present species in NDFF database
        biosafeSpecies = pd.DataFrame(self.bsModel.linksLaw, columns=['Scientific'])
        sectionPresentSpecies = ndffSpeciesPresence(biosafeSpecies, self.ndffSpecies, sectionID)

        #-initialize the biosafe model for a specific flpl section
        self.bsModel.ecotopeArea = dfEcoAreas
        self.bsModel.speciesPresence = sectionPresentSpecies

        #-compile biosafe scores for the section
        sectionParameters = []
        for param in self.params:
            bsParam = self.getParam(parameter = param, ID = sectionID)
            sectionParameters.append(bsParam.rename(columns = {param: sectionID}))

        #-write to file if required
        if self.toFiles == 'xlsx':
            bsIO.output2xlsx(self.bsModel, 'biosafeOutput_section_%s.xlsx' % sectionID)
        elif self.toFiles == 'csv':
            bsIO.output2csv(self.bsModel, 'biosafeOutput_section_%s' % sectionID)
        elif self.toFiles == '':
            pass
        return sectionParameters

    def spatial(self):
        """Computes requruired biosafe output for a spatial domain"""

        #-determine a representative points for each floodplain section
        points = pcrr.representativePoint(self.sections)
        clone = pcr.defined(self.sections)
        pcr.setglobaloption('unittrue')
        xcoor = pcr.xcoordinate(clone)
        ycoor = pcr.ycoordinate(clone)
        geoDf = pcrr.getCellValues(points, \
                                mapList = [points, xcoor, ycoor],\
                                columns = ['ID', 'xcoor', 'ycoor'])
        geoDf.set_index('ID', inplace=True, drop=False)
        geoDf.drop(['rowIdx', 'colIdx', 'ID'], axis=1, inplace=True)

        #-compupte the required biosafe parameters for all sections
        sectionIDs = np.unique(pcr.pcr2numpy(self.sections,-9999))[1:]
        ll = []
        for sectionID in sectionIDs:
            ll.append(self.sectionScores(sectionID))
        #~ paramLL = zip(*ll)
        paramLL = list(zip(*ll))

        dfParamLL = []
        for ii in range(len(self.params)):
            bsScores = pd.concat(paramLL[ii], axis=1).T
            bsScores = bsScores.join(geoDf)
            bsScores.index.name = 'ID'
            bsScores.columns.name = self.params[ii]
            dfParamLL.append(bsScores)

        return dfParamLL

class LegendMap(object):
    """
    Class for storing a PCRaster map with legend and legend title.
    -----------
    Parameters:
    pcr_map: PCRaster map object
    legend: pandas DataFrame with map values and labels
    title: string
    """
    def __init__(self, pcr_map, legend_df):
        self.pcr_map = pcr_map
        self.legend_df = legend_df

class biosafe(object):
    """Compute BIOSAFE scores based on paper of De Nooij et al. (2004)

    Parameters:
        linksEco
        linksLaw
        weightsLegal
        speciesPresence
        ecotopeArea
    """

    def __init__(self, weightsLegal, linksLaw, linksEco, speciesPresence, ecotopeArea):
        self.linksEco = linksEco
        self.linksLaw = linksLaw
        self.weightsLegal = weightsLegal
        self.speciesPresence = speciesPresence
        self.ecotopeArea = ecotopeArea

    def SScoresSpecies(self):
        """
        Compute the species specific S-Scores based on laws and regulations
        both potential and actual scores

        Returns
        -------
        A data frame with columns SPotential, SActual, speciesPresence, taxonomicGroup
        """

        weightedLinksLegal = self.linksLaw.iloc[:,2:18].mul(self.weightsLegal.weights, axis = 1)
        SScores = pd.DataFrame(weightedLinksLegal.sum(axis=1), columns = ['SPotential'])
        SScores['SActual'] = SScores.SPotential.mul(self.speciesPresence.speciesPresence, axis=0)
        # add two columns for convenience
        df1 = SScores.join(self.speciesPresence) # add species presence for checking
        df2 = df1.join(self.linksLaw.taxonomicGroup)
        return df2

    def SEcoPot(self):
        """Computes the species specific S scores for each ecotope separately

        Returns
        -------
        SEcoPot:    Potential Species-specific S-scores per ecotope
        """

        links = self.linksEco.iloc[:,0:-1]
        SEcoPot = links.mul(self.SScoresSpecies().SPotential, axis=0)
        SEcoPot['taxonomicGroup'] = self.linksLaw.taxonomicGroup
        return SEcoPot

    def SEcoAct(self):
        """Computes the species specific S scores for each ecotope separately

        Returns
        -------
        SEcoAct:    Actual Species-specific S-scores per ecotope AS_eco
        """

        links = self.linksEco.iloc[:,0:-1]
        SEcoAct = links.mul(self.SScoresSpecies().SActual, axis=0)
        SEcoAct['taxonomicGroup'] = self.linksLaw.taxonomicGroup
        return SEcoAct

    def taxGroupSums(self):
        """Summarizes the total potential and actual biodiversity

        Returns
        -------
        PTB:    Potential Taxonomic group Biodiversity constant,
                based on laws and regulations
        ATB:    Actual Taxonomic group Biodiversity constant,
                based on species presence
        TBS:    Taxonomic group Biodiversity Saturation,
                which is 100 * ATB / PTB
        """
        sums = self.SScoresSpecies().groupby('taxonomicGroup', as_index = False).sum()
        sums.drop('speciesPresence', inplace=True, axis=1)
        sums.columns = ['taxonomicGroup', 'PTB', 'ATB']
        sums['TBS'] = sums['ATB'] * 100 / sums['PTB']
        sums.set_index('taxonomicGroup', drop=True, inplace=True)
        return sums

    def PTE(self):
        """ PTE: Potential Taxonomic group Ecotope constant
            Sum of Potential S_eco scores per taxonomic group for each ecotope
                 = sum of SEcoPot
        """
        PTE = self.SEcoPot().groupby('taxonomicGroup').sum()
        return PTE

    def ATE(self):
        """ PTE: Potential Taxonomic group Ecotope constant
            Sum of Potential S_eco scores per taxonomic group for each ecotope
                 = sum of SEcoPot
        """
        ATE = self.SEcoAct().groupby('taxonomicGroup').sum()
        return ATE

    def TES(self):
        """ TES: Taxonomic group Ecotope Saturation index
        """
        TES = 100 * self.ATE() / self.PTE()
        return TES.fillna(value = 0)

    def TEI(self):
        """ TEI: Taxonomic group Ecotope Importance constant
        """
        TEI = 100 * self.PTE().div(self.taxGroupSums().PTB, axis = 'index')
        return TEI

    def ATEI(self):
        """ ATEI: Actual Taxonomic group Ecotope Importance scores
        """
        return 0.01 * self.TEI() * self.TES()

    def TFI(self):
        """ TFI: Taxonomic group Floodplain Importance Score
        """
        ecoAreas  = self.ecotopeArea.copy()
        totalArea = ecoAreas.area_m2.sum(axis = 0)
        ecoAreas['fraction'] = ecoAreas.area_m2 / totalArea
        fracTEI = self.TEI() * ecoAreas['fraction']
        TFI = fracTEI.sum(axis = 1)
        return pd.DataFrame(TFI, columns = ['TFI'])

    def FI(self):
        """ FI: Floodplain Importance score
        """
        FI = self.TFI().sum()
        FI = pd.DataFrame(FI, columns = ['FI'])
        FI.index = ['FI']
        return FI


    def ATFI(self):
        """ ATFI: Actual Taxonomic group Floodplain Importance Score
        """
        ecoAreas  = self.ecotopeArea.copy()
        totalArea = ecoAreas.area_m2.sum(axis = 0)
        ecoAreas['fraction'] = ecoAreas.area_m2 / totalArea
        fracTEI = self.ATEI() * ecoAreas['fraction']
        ATFI = fracTEI.sum(axis = 1)
        return pd.DataFrame(ATFI, columns = ['ATFI'])

    def AFI(self):
        """ FI: Actual Floodplain Importance score
        """
        AFI = self.ATFI().sum()
        AFI = pd.DataFrame(AFI, columns = ['AFI'])
        AFI.index = ['AFI']
        return AFI

    def TFIS(self):
        """TFIS: Taxonomic Group Floodplain Importance Saturation
        Describes the fraction of actual over potential biodiversity value"""
        TFIS = 100 * self.ATFI().ATFI.values / self.TFI().TFI.values
        return pd.DataFrame(TFIS, columns = ['TFIS'], index=self.TFI().index.values)

    def FIS(self):
        """FIS: Floodplain Importance Saturation
        Describes the fraction of actual over potential biodiversity value per
        taxonomic group."""
        FIS = 100* self.AFI().AFI.values / self.FI().FI.values
        return pd.DataFrame(FIS, columns = ['FIS'], index=['FIS'])

    def TFHS(self):
        """TFHS: Taxonomic Group Floodplain Habitat Saturation.
        Describes the fraction of suitable ecotopes, weighted by law.
        Computed as present floodplain TEI for present ecotopes over
        all possible floodplain ecotopes relevant for the taxonomic group.
        """
        #-FTEI: Floodplain TEI, TEI summed over all ecotopes
        FTEI = self.TEI().sum(axis=1)

        #-PFTEI: Floodplain TEI, TEI summed over the whole area
        # for ecotopes present
        ecoArea = self.ecotopeArea
        ecoPresence = ecoArea.area_m2 > 0
        PFTEI = self.TEI().mul(ecoPresence, axis=1).sum(axis=1)

        #-TFHS: Taxonomic Group Floodplain Habitat Suitability
        TFHS = 100 * PFTEI / FTEI
        return pd.DataFrame(TFHS, columns=['TFHS'])

    def FTEI(self):
        """ FTEI: Floodplain Taxonomic group Ecotope Importance
        """
        ecoAreas  = self.ecotopeArea.copy()
        totalArea = ecoAreas.area_m2.sum(axis = 0)
        ecoAreas['fraction'] = ecoAreas.area_m2 / totalArea
        FTEI = 100 * self.TEI() * ecoAreas['fraction']
        return FTEI.fillna(0)

if __name__ == '__main__':
    #%% initiate biosafe

    # Directory settings
    root_dir = os.path.dirname(os.getcwd())
    # - for testing on eejit
    root_dir = "/scratch/depfg/sutan101/test_biosafe/"

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























