import ipywidgets
from ipywidgets import interact, interactive
from ipywidgets.embed import embed_minimal_html, dependency_state
from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as patches


max_channel_width = 200
default_width = 50
default_depth = 1
default_slope = 2


def _width_for_slope(height, slope):

  return height * slope


def _coords(width, depth, slope, y_offset):

  slopewidth = _width_for_slope(depth, slope)

  # left low
  x1 = -width
  y1 = y_offset - depth
  # right low
  x2 = width
  y2 = y_offset - depth
  # right up
  x3 = width + slopewidth
  y3 = y_offset
  # left up
  x4 = -width - slopewidth
  y4 = y_offset

  return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]


def _plot_channel(width, depth, slope):

  # User provides total channel width, we calculate everything from the channel centre
  width = width / 2.0


  max_depth = 5
  min_slope = 10

  height_above_exceedance = 3.0

  # Some estimation of the max channel width, based on 'water' part
  max_width = max_channel_width / 2.0 + _width_for_slope(max_depth, min_slope)


  y_delta = 1.0
  y_min = 0.0 - max_depth - y_delta
  y_max = height_above_exceedance + y_delta



  fig = plt.figure(figsize=(12, 2), dpi=300)
  ax = fig.add_subplot(111)

  # Plot soil
  soil_colour = 'tan'
  plt.fill_between([-max_width, max_width - .1], [y_min, y_min],[ height_above_exceedance - .1, height_above_exceedance - .1], color=soil_colour)



  # Plot 'air' part of channel
  air_colour = 'white'
  user_coords = _coords(width, depth + height_above_exceedance, slope, height_above_exceedance)
  ax.add_patch(patches.Polygon(xy=user_coords, fill=True, color=air_colour))


  # Plot water part of channel
  user_coords = _coords(width, depth, slope, 0)
  ax.add_patch(patches.Polygon(xy=user_coords, fill=True))



  ax.set_ylim([y_min, y_max])
  ax.set_xlim([-max_width, max_width])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  plt.xlabel('Channel width [m]')
  plt.ylabel('Depth [m]')

  ax.plot()




def channel_properties():

  w = ipywidgets.IntSlider(
      value=default_width,
      min=5,
      max=max_channel_width,
      step=5,
      description='Width [m]'
  )

  d = ipywidgets.FloatSlider(
      value=default_depth,
      min=0,
      max=5,
      step=0.25,
      description='Depth [m]'
  )

  s = ipywidgets.IntSlider(
      value=default_slope,
      min=1,
      max=10,
      step=1,
      description='Slope [1:x]'
  )

  res = interactive(_plot_channel, width=w, depth=d, slope=s)
  display(res)

  return res
