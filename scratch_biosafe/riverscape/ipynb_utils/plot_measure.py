import ipywidgets
import os


import pcraster

from .map_plot import plot



def _plot_datasets(data_path, measure_dir):

  # this is partly replica of the plot in measures, unify this at some point
  # see if we can plot this without making an entire instance of a measure...
  # Fix this plot anyway, numpy.nan...

  path = os.path.join(data_path, measure_dir)

  area             = pcraster.readmap(os.path.join(path, 'area.map'))
  dem              = pcraster.readmap(os.path.join(path, 'dem.map'))
  ecotopes         = pcraster.readmap(os.path.join(path, 'ecotopes.map'))
  trachytopes      = pcraster.readmap(os.path.join(path, 'trachytopes.map'))
  groyne_height    = pcraster.readmap(os.path.join(path, 'groyne_height.map'))
  minemb_height    = pcraster.readmap(os.path.join(path, 'minemb_height.map'))
  main_dike_height = pcraster.readmap(os.path.join(path, 'main_dike_height.map'))


  display((plot(area, 'Area') + plot(dem, 'Digital elevation map') + plot(ecotopes, 'Ecotopes') + plot(trachytopes, 'Trachytopes') + plot(groyne_height, 'Groyne height') + plot(minemb_height, 'Minor embankment height') + plot(main_dike_height, 'Main dike height')).cols(1))





def plot_measures(data_path):


  w = ipywidgets.Dropdown(
      options=['dikeraising_evr_smooth', 'dikeraising_lrg_smooth', 'groynelowering_evr_smooth', 'groynelowering_lrg_smooth', 'lowering_evr_natural', 'lowering_evr_smooth', 'lowering_lrg_natural', 'lowering_lrg_smooth', 'minemblowering_evr_smooth', 'minemblowering_lrg_smooth', 'sidechannel_evr_natural', 'sidechannel_evr_smooth', 'sidechannel_lrg_natural', 'sidechannel_lrg_smooth', 'smoothing_evr_natural', 'smoothing_evr_smooth', 'smoothing_lrg_natural', 'smoothing_lrg_smooth'],
      description='Measure to display:'
  )



  def on_change(change):
      if change['type'] == 'change' and change['name'] == 'value':
          selected = change['new']

          _plot_datasets(data_path, selected)

  w.observe(on_change)

  display(w)

