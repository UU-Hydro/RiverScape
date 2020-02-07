import numpy

import geoviews
from .map_plot import plot

from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file

import ipywidgets
from ipywidgets import Layout
from ipywidgets import GridspecLayout


import pcraster






def select_area(raster):

  # Get the list of IDs from the raster, use it for the dropdown field
  raster_np = pcraster.pcr2numpy(raster, -9999)

  raster_values = numpy.unique(raster_np)

  raster_unique = numpy.unique(raster_values)

  sections = numpy.delete(raster_unique, numpy.where(raster_unique == -9999))



  hover = HoverTool(tooltips=[
      ("Floodplain section", "@image"),
  ])
  p = plot(raster, hover=hover)

  display(p)

  style = {'description_width': 'initial'}
  w = ipywidgets.SelectMultiple(
    options=sections,
    rows=8,
    description='Sections:', layout=Layout(width="50%"), style=style,
    disabled=False
  )


  display(w)
  return w





def generate_mask(raster, areas):

  indices = areas.value

  if len(indices) == 0:
    print('\x1b[31m \nPlease select one or more floodplain sections before you proceed! \x1b[0m')
    print('\x1b[31m \nNote: you can also continue with the default subset selection.\n\x1b[0m')

    # To enable the 'autorun' of the intervention planning notebook we set some default values
    indices = areas.options[0:5]



  idx = ', '.join(str(index) for index in indices)
  print('Selected floodplain sections: {}'.format(idx))

  selection = ')) | (raster==pcraster.nominal('.join(str(index) for index in indices)

  cmd = 'pcraster.ifthen((raster==pcraster.nominal({})), pcraster.boolean(1))'.format(selection)

  mask = eval(cmd)

  return mask


