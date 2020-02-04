import os
import gdal
import numpy
import holoviews as hv
import pcraster

import bokeh
from holoviews import opts
import geoviews



def _get_raster(raster):
  # Return masked array and colour scheme based on the raster type
  data = None
  colour = None

  if raster.dataType() == pcraster.Scalar:
    data = pcraster.pcr2numpy(raster, numpy.nan)
    mask = numpy.where(data == numpy.nan, True, False)
    data = numpy.ma.masked_array(data, mask)
  elif raster.dataType() == pcraster.Nominal:
    nan_val = -2147483648
    data = pcraster.pcr2numpy(raster, nan_val)
    mask = numpy.where(data == nan_val, True, False)
    data = numpy.ma.masked_array(data, mask)
  elif raster.dataType() == pcraster.Boolean:
    nan_val = 255
    data = pcraster.pcr2numpy(raster, nan_val)
    mask = numpy.where(data == nan_val, True, False)
    data = numpy.ma.masked_array(data, mask)
  else:
    msg = 'Plotting of rasters with data type "{}" is not supported'.format(str(raster.dataType()).split('.')[1])
    raise NotImplementedError(msg)

  return data



def plot(raster, title='', hover=None):
  #geoviews.extension('bokeh')

  # casting should not be necessary...
  np_raster = _get_raster(pcraster.scalar(raster))

  minx = pcraster.clone().west()
  maxy = pcraster.clone().north()
  maxx = minx + pcraster.clone().nrCols() * pcraster.clone().cellSize()
  miny = maxy - pcraster.clone().nrRows() * pcraster.clone().cellSize()
  extent = (minx, maxx, miny, maxy)


  img = hv.Image(np_raster, bounds=extent)




  if hover is None:
    hover = ['hover']
  else:
    hover = [hover]

  s = pcraster.clone().nrCols() / pcraster.clone().nrRows()

  width = 770

  if raster.dataType() == pcraster.Scalar:
    return img.options(cmap='viridis', tools=hover, aspect=s, colorbar=True, frame_width=width, toolbar="above").relabel(title)
  else:
    return img.options(cmap='flag', tools=hover, aspect=s, frame_width=width, toolbar="above").relabel(title)

