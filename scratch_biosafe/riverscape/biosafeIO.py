# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#~ import pysal as ps # NOT NEEDED
import os, string

class from_xlsx_v3():
    """Read biosafe data from an excel file in version 3 format as produced by
    Alexandra Bloecker.
    This file is located in the inputData folder
    """
    
    def __init__(self, xlsxFile):
        self.xlsxFile = xlsxFile
    
    def weights(self):
        """Read the legal and regulatory weights from the input file
        
        Returns:
            0: weights = the user-specified weights for 16 laws and regulations
        """
        weights = pd.read_excel(self.xlsxFile, 'Weights')
        weights.set_index(weights.codes, inplace=True)
        weights.drop('codes', axis=1, inplace=True)
        return weights
    
    def readTaxGroup(self, taxGroupSheetName):
        """Read the taxonomic group data from a BIOSAFE excel worksheet
        contains per specie both the links to laws as well as to ecotopes"""
        print('\t', taxGroupSheetName)
        speciesData  = pd.read_excel(self.xlsxFile, taxGroupSheetName, 
                                 skiprows = 2, 
                                 parse_cols = range(101))
    
        colNames = ['Scientific', 'Dutch', 'MinArea', \
                    'RL_NT', 'RL_VU', 'RL_EN', 'RL_CR', 'RL_EX', \
                    'HabDir_II', 'HabDir_IV',
                    'BirdDir_I', \
                    'BernConv_1', 'BernConv_2',\
                    'BonnConv_1', 'BonnConv_2',\
                    'FF_Table1' , 'FF_Table2' , 'FF_Table3', 'FF_Birds',\
                    ]
        colNames += list(speciesData.columns[19:102])
        speciesData.columns = colNames
        speciesData.drop(0, inplace = True)
        speciesData.replace(to_replace = 'x', value = 1, inplace = True)
        speciesData.replace(to_replace = ' ', value = 0, inplace = True)
        speciesData.replace(to_replace = 'x foer', value = 1, inplace = True)
        speciesData.replace(to_replace = 'x broed', value = 1, inplace = True)
        speciesData.replace(to_replace = 'xfoer', value = 1, inplace = True)
        speciesData.replace(to_replace = 'xbroed', value = 1, inplace = True)
        speciesData.replace(to_replace = 'x foer/broed', value = 1, inplace = True)
        speciesData.replace(to_replace = 'x broeden', value = 1, inplace = True)
        speciesData.replace(to_replace = 'x broed/foer', value = 1, inplace = True)
        speciesData.replace(to_replace = 'x foer&broed', value = 1, inplace = True)
        speciesData.replace(to_replace = 'x ', value = 1, inplace = True)
        speciesData.index = speciesData['Scientific'].values
        speciesData['taxonomicGroup'] = taxGroupSheetName
        speciesData.fillna(value=0, inplace=True)
        
#        #- test data type of columns
#        3 a.dtypes.isin([np.dtype('float64')]).values
        return speciesData

    def speciesLinks(self):
        """Read BIOSAFE species data per taxonomic group.
        Returns:
            linksLaw:    links between species and laws and regulations
            linksEco:    links between species and ecotope
        Both are binary matrices
        """
        # Loop over the taxonomic groups to compile all data into a single dataframe
        print('\nReading in BIOSAFE taxonomic group data from excel')
        workSheets = ['Mammals', 'Birds', 'Herpetofauna', 'Fish',\
                        'Butterflies', 'DragonDamselflies', 'HigherPlants']
        dfList = []
        for taxGroup in workSheets:
            dfList.append(self.readTaxGroup(taxGroup))
        speciesDataAll = pd.concat(dfList, axis=0)
        
        # extract separate species links with law and ecotopes
        linksLaw = speciesDataAll.iloc[:,0:19].drop('MinArea', axis=1)
        linksLaw['taxonomicGroup'] = speciesDataAll.taxonomicGroup
        linksEco = speciesDataAll.iloc[:,19:]
        return linksLaw, linksEco

def xlsxToCsv(excelFile, outputDir):
    inputData = from_xlsx_v3(excelFile)

    legalWeights = inputData.weights()
    legalWeights.to_csv(os.path.join(outputDir, 'legalWeights.csv'))    
        
    linksLaw, linksEco = inputData.speciesLinks()    
    linksLaw.to_csv(os.path.join(outputDir, 'linksLaw.csv')) 
    linksEco.to_csv(os.path.join(outputDir, 'linksEco.csv')) 

def from_csv(inputDir):
    """
    Read four standard input files from .csv file format
    """
    legalWeights = pd.read_csv(os.path.join(inputDir, 'legalWeights.csv'), index_col = 0)
    linksLaw = pd.read_csv(os.path.join(inputDir, 'linksLaw.csv'), index_col = 0)
    linksEco = pd.read_csv(os.path.join(inputDir, 'linksEco.csv'), index_col = 0)
    return legalWeights, linksLaw, linksEco

def output2xlsx(model, filePath):
    """Writes an instance of the biodiversity model to an excel files
    
    model   : biosafe instance
    filePath: excel (xlsx) file on disk
    
    returns
    -------
    excel file on disk with separate sheets for all parameters
    """
    SScoresSpecies = model.SScoresSpecies()
    summarySScores = model.taxGroupSums()
    SEcoPot = model.SEcoPot()
    SEcoAct = model.SEcoAct()
    PTE = model.PTE()
    ATE = model.ATE()
    TES = model.TES()
    TEI = model.TEI()
    ATEI = model.ATEI()
    TFI = model.TFI()         
    FI = model.FI()
    ATFI = model.ATFI()         
    AFI = model.AFI()
    TFIS = model.TFIS()
    FIS = model.FIS()
    TFHS = model.TFHS()
    
    print('Writing excel output to %s' % filePath)
    writer = pd.ExcelWriter(filePath)
    model.weightsLegal.to_excel(writer, 'weightsLegal')
    model.linksLaw.to_excel(writer, 'linksLaw')
    model.linksEco.to_excel(writer, 'linksEco')
    model.speciesPresence.to_excel(writer, 'speciesPresence')
    model.ecotopeArea.to_excel(writer, 'ecotopeArea')
    SScoresSpecies.to_excel(writer, 'SScoresSpecies')
    summarySScores.to_excel(writer, 'summarySScores')
    SEcoPot.to_excel(writer, 'SEcoPot')
    SEcoAct.to_excel(writer, 'SEcoAct')
    PTE.to_excel(writer, 'PTE')
    ATE.to_excel(writer, 'ATE')
    TES.to_excel(writer, 'TES')
    TEI.to_excel(writer, 'TEI')
    ATEI.to_excel(writer, 'ATEI')
    TFI.to_excel(writer, 'TFI')
    FI.to_excel(writer, 'FI')
    ATFI.to_excel(writer, 'ATFI')
    AFI.to_excel(writer, 'AFI')
    TFIS.to_excel(writer, 'TFIS')
    FIS.to_excel(writer, 'FIS')
    TFHS.to_excel(writer, 'TFHS')
    writer.save()

def output2csv(model, directoryPath):
    """Writes an instance of the biodiversity model to a set of csv files
    
    model        : biosafe instance
    directoryPath: directory on disk
    
    returns
    -------
    a set of separate .csv files all parameters
    """
    try:
        os.makedirs(directoryPath)
    except OSError:
        pass    
    
    SScoresSpecies = model.SScoresSpecies()
    summarySScores = model.taxGroupSums()
    SEcoPot = model.SEcoPot()
    SEcoAct = model.SEcoAct()
    PTE = model.PTE()
    ATE = model.ATE()
    TES = model.TES()
    TEI = model.TEI()
    ATEI = model.ATEI()
    TFI = model.TFI()         
    FI = model.FI()
    ATFI = model.ATFI()         
    AFI = model.AFI()
    TFIS = model.TFIS()
    FIS = model.FIS()
    TFHS = model.TFHS()
    
    print('Writing csv output to %s' % directoryPath)
    
    model.weightsLegal.to_csv(os.path.join(directoryPath, '01_weightsLegal.csv'))
    model.linksLaw.to_csv(os.path.join(directoryPath, '02_linksLaw.csv'))
    model.linksEco.to_csv(os.path.join(directoryPath, '03_linksEco.csv'))
    model.speciesPresence.to_csv(os.path.join(directoryPath, '04_speciesPresence.csv'))
    model.ecotopeArea.to_csv(os.path.join(directoryPath, '05_ecotopeArea.csv'))
    SScoresSpecies.to_csv(os.path.join(directoryPath, '06_SScoresSpecies.csv'))
    summarySScores.to_csv(os.path.join(directoryPath, '07_summarySScores.csv'))
    SEcoPot.to_csv(os.path.join(directoryPath, '08_SEcoPot.csv'))
    SEcoAct.to_csv(os.path.join(directoryPath, '09_SEcoAct.csv'))
    PTE.to_csv(os.path.join(directoryPath, '10_PTE.csv'))
    ATE.to_csv(os.path.join(directoryPath, '11_ATE.csv'))
    TES.to_csv(os.path.join(directoryPath, '12_TES.csv'))
    TEI.to_csv(os.path.join(directoryPath, '13_TEI.csv'))
    ATEI.to_csv(os.path.join(directoryPath, '14_ATEI.csv'))
    TFI.to_csv(os.path.join(directoryPath, '15_TFI.csv'))
    FI.to_csv(os.path.join(directoryPath, '16_FI.csv'))
    ATFI.to_csv(os.path.join(directoryPath, '17_ATFI.csv'))
    AFI.to_csv(os.path.join(directoryPath, '18_AFI.csv'))
    TFIS.to_csv(os.path.join(directoryPath, '19_TFIS.csv'))
    FIS.to_csv(os.path.join(directoryPath, '20_FIS.csv'))
    TFHS.to_csv(os.path.join(directoryPath, '21_TFHS.csv'))
    
def dbf2df(dbf_path, index=None, cols=False, incl_index=False):
    '''
    Read a dbf file as a pandas.DataFrame, optionally selecting the index
    variable and which columns are to be loaded.
    __author__  = "Dani Arribas-Bel <darribas@asu.edu> "
    ...
    Arguments
    ---------
    dbf_path    : str
                  Path to the DBF file to be read
    index       : str
                  Name of the column to be used as the index of the DataFrame
    cols        : list
                  List with the names of the columns to be read into the
                  DataFrame. Defaults to False, which reads the whole dbf
    incl_index  : Boolean
                  If True index is included in the DataFrame as a
                  column too. Defaults to False
    Returns
    -------
    df          : DataFrame
                  pandas.DataFrame object created
    '''
    db = ps.open(dbf_path)
    if cols:
        if incl_index:
            cols.append(index)
        vars_to_read = cols
    else:
        vars_to_read = db.header
    data = dict([(var, db.by_col(var)) for var in vars_to_read])
    if index:
        index = db.by_col(index)
        db.close()
        return pd.DataFrame(data, index=index)
    else:
        db.close()
        return pd.DataFrame(data)   
       
###############################################################################
#-test IO #####################################################################
###############################################################################

#scratchDir  = r'D:\projecten\biodiversityRhine\scratch'
#os.chdir(scratchDir)
#excelFile = 'D:/projecten/biodiversityRhine/biosafe/biodiversity/inputData/BIOSAFE_20150522.xlsx'
#
#inputData = from_xlsx_v3(excelFile)
#legalWeights = inputData.weights()
#linksLaw, linksEco = inputData.speciesLinks()
#xlsxToCsv(excelFile, scratchDir)
#legalWeights, linksLaw, linksEco = from_csv(scratchDir)

















