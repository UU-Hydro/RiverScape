import os


from .utils import input_data_path

from .map_plot import plot


import pcraster






def plot_landowners(measure_path):

  path_area = os.path.join(measure_path, 'area.map')
  path_owner = os.path.join(input_data_path(), 'restraints', 'owner_nr.map')


  area = pcraster.readmap(path_area)
  owner = pcraster.readmap(path_owner)

  display ((plot(area, 'Area affected by measure') + plot(owner, 'Owner existence')).cols(1))
