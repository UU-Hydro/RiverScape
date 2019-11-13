
import geoviews
from .map_plot import plot

from bokeh.models import HoverTool
from bokeh.plotting import figure, show, output_file

import ipywidgets
from ipywidgets import Layout
from ipywidgets import GridspecLayout


import pcraster as pcr






def select_area(raster):



  hover = HoverTool(tooltips=[
      ("Floodplain section", "@image"),
  ])
  p = plot(raster, hover=hover)

  display(p)

  style = {'description_width': 'initial'}
  w = ipywidgets.SelectMultiple(
    options=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58'],
    rows=8,
    description='Sections:',layout=Layout(width="50%"), style=style,
    disabled=False
  )


  display(w)
  return w





def generate_mask(raster, areas):

  indices = areas.value

  if len(indices) == 0:
    print('\x1b[31m \nPlease select one or more floodplain sections before you proceed! \x1b[0m')
    return


  idx = ', '.join(str(index) for index in indices)
  print('Selected floodplain sections: {}'.format(idx))

  selection = ')) | (raster==pcr.nominal('.join(str(index) for index in indices)

  cmd = 'pcr.ifthen((raster==pcr.nominal({})), pcr.boolean(1))'.format(selection)

  mask = eval(cmd)

  return mask


