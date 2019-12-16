import os
from io import StringIO
from collections import OrderedDict
import numpy as np
import pandas as pd

import pcraster as pcr


# we should get this from the original location from riverscape...
def _maps_to_dataframe(maps, columns, MV = np.nan):

    """Convert a set of maps to flattened arrays as columns.

    Input:

        maps: list of PCRaster maps

        columns: list of strings with column names

        MV: missing values, defaults to numpy nan.

    Returns a pandas dataframe with missing values in any column removed.

    """

    data = OrderedDict()

    for name, pcr_map in zip(columns, maps):

        data[name] = pcr.pcr2numpy(pcr_map, MV).flatten()

    return pd.DataFrame(data).dropna()





def involved_stakeholders(path_measures):

  # From user input
  measure_dir_area = os.path.join(path_measures, 'area.map')


  path_restraints = os.path.join(os.getcwd(), '..', 'input_files', 'input', 'restraints')

  # Read inputs
  owner_count_point = pcr.readmap(os.path.join(path_restraints, 'owner_point_number.map'))
  owner_type = pcr.readmap(os.path.join(path_restraints, 'owner_type.map'))
  owner_uid = pcr.readmap(os.path.join(path_restraints, 'owner_nr.map'))

  # Distribute point values to corresponding areas
  owner_count = pcr.areamaximum(owner_count_point, owner_uid)

  # Mask with measure area
  owner_count_measure = pcr.ifthen(measure_dir_area, owner_count)
  owner_type_measure = pcr.ifthen(measure_dir_area, owner_type)
  owner_uid_measure = pcr.ifthen(measure_dir_area, owner_uid)

  df = _maps_to_dataframe([owner_count_measure, owner_type_measure, owner_uid_measure], ['count', 'type', 'uid'])


  owner_types = df['type'].values

  owner_types = np.unique(owner_types)

  owner_types_label = ['Public Works and Water Management', 'Waterboard', 'Municipality', 'Company', 'Citizen', 'State Forestry Servive', 'Province', 'Geldersch and Brabants Landschap', 'Foundation', 'Sand/gravel/clay company', 'other']


  content = 'Owner type,Number of stakeholders involved\n'

  for ot in owner_types:
    newdf = df.loc[df['type'] == ot]

    newdf = newdf.drop_duplicates()

    tot_own = newdf['count'].sum()

    row_content = '{},{}\n'.format(owner_types_label[int(ot) - 1], int(tot_own))

    content = '{}{}'.format(content, row_content)

  df = pd.read_csv(StringIO(content))


  return df

