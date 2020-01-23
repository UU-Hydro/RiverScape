import os
import pandas as pd

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import Range1d






def plot_water_level_change():

  # evaluate in what respect this type of plot can be generic
  # wrt measures provided by user...


  subplot_width = 800
  subplot_height = 150

  colour_evr_natural = 'green'
  colour_evr_smooth = 'indigo'
  colour_lrg_natural = 'red'
  colour_lrg_smooth = 'blue'

  name_evr_natural = 'evr_natural'
  name_evr_smooth = 'evr_smooth'
  name_lrg_natural = 'lrg_natural'
  name_lrg_smooth = 'lrg_smooth'


  data_path = os.path.join(os.getcwd(), '..', 'input_files', 'input', 'measures')

  data_fname = 'water_level.csv'

  df = pd.read_csv(os.path.join(data_path, data_fname))


  ymin = -1.5
  ymax = 0.5

  xmin = 865
  xmax = 980

  lwidth = 2


  subfig1 = figure(plot_width=subplot_width, plot_height=subplot_height, tools=['pan', 'box_zoom', 'wheel_zoom', 'zoom_in', 'zoom_out', 'reset'], toolbar_location='above')



  subfig1.x_range = Range1d(xmin, xmax)
  subfig1.y_range = Range1d(ymin, ymax)

  subfig1.line(df['km'], df['groynelowering_evr_smooth'], color=colour_evr_smooth, line_width=lwidth, legend_label=name_evr_smooth)
  subfig1.line(df['km'], df['groynelowering_lrg_smooth'], color=colour_lrg_smooth, line_width=lwidth, legend_label=name_lrg_smooth)
  subfig1.line(df['km'], df['dikeraising_evr_smooth'],    color=colour_evr_natural, line_width=lwidth, legend_label=name_evr_natural)
  subfig1.line(df['km'], df['dikeraising_evr_smooth'],    color=colour_lrg_natural, line_width=lwidth, legend_label=name_lrg_natural)



  subfig2 = figure(plot_width=subplot_width, plot_height=subplot_height, x_range=subfig1.x_range, y_range=subfig1.y_range, tools=[])




  subfig2.line(df['km'], df['minemblowering_evr_smooth'], color=colour_evr_smooth, line_width=lwidth)
  subfig2.line(df['km'], df['minemblowering_lrg_smooth'], color=colour_lrg_smooth, line_width=lwidth)




  subfig3 = figure(plot_width=subplot_width, plot_height=subplot_height, x_range=subfig1.x_range, y_range=subfig1.y_range, tools=[])
  subfig3.line(df['km'], df['smoothing_evr_natural'], color=colour_evr_natural, line_width=lwidth)
  subfig3.line(df['km'], df['smoothing_evr_smooth'], color=colour_evr_smooth, line_width=lwidth)
  subfig3.line(df['km'], df['smoothing_lrg_natural'], color=colour_lrg_natural, line_width=lwidth)
  subfig3.line(df['km'], df['smoothing_lrg_smooth'], color=colour_lrg_smooth, line_width=lwidth)



  subfig4 = figure(plot_width=subplot_width, plot_height=subplot_height, x_range=subfig1.x_range, y_range=subfig1.y_range, tools=[])
  subfig4.line(df['km'], df['sidechannel_evr_natural'], color=colour_evr_natural, line_width=lwidth)
  subfig4.line(df['km'], df['sidechannel_evr_smooth'], color=colour_evr_smooth, line_width=lwidth)
  subfig4.line(df['km'], df['sidechannel_lrg_natural'], color=colour_lrg_natural, line_width=lwidth)
  subfig4.line(df['km'], df['sidechannel_lrg_smooth'], color=colour_lrg_smooth, line_width=lwidth)


  subfig5 = figure(plot_width=subplot_width, plot_height=int(subplot_height + 0.2 * subplot_height), x_range=subfig1.x_range, y_range=subfig1.y_range, tools=[])
  subfig5.line(df['km'], df['lowering_evr_natural'], color=colour_evr_natural, line_width=lwidth)
  subfig5.line(df['km'], df['lowering_evr_smooth'], color=colour_evr_smooth, line_width=lwidth)
  subfig5.line(df['km'], df['lowering_lrg_natural'], color=colour_lrg_natural, line_width=lwidth)
  subfig5.line(df['km'], df['lowering_lrg_smooth'], color=colour_lrg_smooth, line_width=lwidth)





  subplots = [subfig1, subfig2, subfig3, subfig4, subfig5]

  for s in subplots:
    s.xaxis.major_label_text_font_size = '0pt'

    s.y_range = Range1d(ymin, ymax, bounds=(ymin, ymax))
    s.yaxis.axis_label = u"\u0394h (m) "



  subplots[-1].xaxis.major_label_text_font_size = '8pt'
  subplots[-1].xaxis.axis_label = "River kilometer"


  subfig1.title.text = "Measure: Groyne lowering"
  subfig2.title.text = "Measure: minemblowering"
  subfig3.title.text = "Measure: Smoothing"
  subfig4.title.text = "Measure: Side channel"
  subfig5.title.text = "Measure: lowering"


  p = gridplot(subplots, ncols=1, toolbar_location='above')



  show(p)
