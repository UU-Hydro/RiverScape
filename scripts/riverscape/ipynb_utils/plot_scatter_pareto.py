import os
import pandas as pd

import numpy as np
import scipy



from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
from bokeh.models import Range1d
from bokeh.transform import factor_cmap, factor_mark

from bokeh.models import HoverTool



def pareto_points(two_col_df):
    """Extract the optimal points in the lower left corner in 2D.
    input:
        two_col_df: DataFrame with two columns.
    """

    points = two_col_df.values
    hull = scipy.spatial.ConvexHull(points)
    vertices = np.hstack((hull.vertices, hull.vertices[0]))
    ll = []
    for ii in range(len(vertices[0:-1])):
        p1 = points[vertices[ii]]
        p2 = points[vertices[ii+1]]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        if (dx >= 0) & (dy <= 0):
            ll.append(p1)
            ll.append(p2)
    optimal_df = pd.DataFrame(data=np.vstack(ll), columns=two_col_df.columns)
    optimal_df.drop_duplicates(inplace=True)
    optimal_df.sort_values(optimal_df.columns[0], inplace=True)
    return optimal_df






def plot_scatter(label=None, water=None, stakeholders=None, costs=None, potAll=None):

  # needs some heavy refactoring...


  data_path = os.path.join(os.getcwd(), '..', 'input_files', 'input', 'measures')



  data_fname = 'stats_measures.csv'

  df = pd.read_csv(os.path.join(data_path, data_fname))


  fill_alpha = 0.7
  line_width = 1

  # Plain adding a column with marker sizes
  marker_size = len(df) * [10]

  df['marker_size'] = marker_size

  # Add the user defined measure, in case
  if not (label is None and water is None and stakeholders is None and costs is None and potAll is None):

    colour = 'gold'
    marker = 'hex'
    marker_size = 20

    row = pd.DataFrame([[label,water,colour,costs,potAll,stakeholders,marker,marker_size]], columns=['labels','dwl_Qref','colour','cost_sum','FI','nr_stakeholders', 'marker','marker_size'])

    df = df.append(row)


  subplot_width = 275
  subplot_height = subplot_width
  min_border = 0
  delta_offset_left = 50


  categories = df['labels']
  markers = df['marker']
  marker_sizes = df['marker_size']

  pot_ymin = 60
  pot_ymax = 180



  colours = df['colour']

  y = 'nr_stakeholders'


  toolset = ['pan', 'box_zoom', 'wheel_zoom', 'zoom_in', 'zoom_out', 'reset']


  subfig11 = figure(plot_width=subplot_width + delta_offset_left, plot_height=subplot_height,
    min_border_left=min_border,
    min_border_bottom=min_border,
                    toolbar_location='above', tools=toolset)
  x = 'dwl_Qref'

  v1 = 'dwl_Qref'
  v2 = 'nr_stakeholders'
  pp = pareto_points(df[[v1, v2]])
  subfig11.line(pp[v1], pp[v2], line_width=20, color='gray', line_alpha=0.25)

  scatter11 = subfig11.scatter(x, y,  source=df, size='marker_size', marker=factor_mark('labels', markers, categories), color=factor_cmap('labels', colours, categories), fill_alpha=fill_alpha, line_width=line_width)
  subfig11.yaxis.axis_label = 'No. of stakeholders (-)'


  subfig11.add_tools(HoverTool(tooltips=[('', '@labels')], renderers=[scatter11]))


  y = 'FI'

  subfig21 = figure(plot_width=subplot_width + delta_offset_left, plot_height=subplot_height,
    min_border_left=min_border,
    min_border_bottom=min_border,
                    tools=toolset,
                    toolbar_location=None, x_range=subfig11.x_range)
  x = 'dwl_Qref'


  v1 = 'dwl_Qref'
  v2 = 'FI'
  pp = pareto_points(pd.concat([df[['dwl_Qref']], -df[['FI']]], axis=1))
  subfig21.line(pp[v1], -pp[v2], line_width=20, color='gray', line_alpha=0.25)

  scatter21 = subfig21.scatter(x, y,  source=df, size='marker_size', marker=factor_mark('labels', markers, categories), color=factor_cmap('labels', colours, categories), fill_alpha=fill_alpha, line_width=line_width)
  subfig21.yaxis.axis_label = 'PotAll (-)'
  subfig21.y_range = Range1d(pot_ymax, pot_ymin)

  subfig21.add_tools(HoverTool(tooltips=[('', '@labels')], renderers=[scatter21]))


  subfig22 = figure(plot_width=subplot_width, plot_height=subplot_height,
    min_border_left=min_border,
    min_border_bottom=min_border,
                    tools=toolset,
                    toolbar_location=None, y_range=subfig21.y_range)
  x = 'nr_stakeholders'

  v1 = 'nr_stakeholders'
  v2 = 'FI'
  pp = pareto_points(pd.concat([df[['nr_stakeholders']], -df[['FI']]], axis=1))
  subfig22.line(pp[v1], -pp[v2], line_width=20, color='gray', line_alpha=0.25)

  scatter22 = subfig22.scatter(x, y,  source=df, size='marker_size', marker=factor_mark('labels', markers, categories), color=factor_cmap('labels', colours, categories), fill_alpha=fill_alpha, line_width=line_width)
  subfig22.yaxis.major_label_text_font_size = '0pt'



  subfig22.add_tools(HoverTool(tooltips=[('', '@labels')], renderers=[scatter22]))

  y = 'cost_sum'

  subfig31 = figure(plot_width=subplot_width + delta_offset_left, plot_height=subplot_height,
    min_border_left=min_border,
    min_border_bottom=min_border,
                    tools=toolset,
                    toolbar_location=None, x_range=subfig11.x_range)
  x = 'dwl_Qref'


  v1 = 'dwl_Qref'
  v2 = 'cost_sum'
  pp = pareto_points(df[[v1, v2]])
  subfig31.line(pp[v1], pp[v2], line_width=20, color='gray', line_alpha=0.25)

  scatter31 = subfig31.scatter(x, y,  source=df, size='marker_size', marker=factor_mark('labels', markers, categories), color=factor_cmap('labels', colours, categories), fill_alpha=fill_alpha, line_width=line_width)
  subfig31.yaxis.axis_label = 'Implementation costs (\u20AC)'
  subfig31.xaxis.axis_label = 'Water level lowering (m)'


  subfig31.add_tools(HoverTool(tooltips=[('', '@labels')], renderers=[scatter31]))


  subfig32 = figure(plot_width=subplot_width, plot_height=subplot_height,
    min_border_left=min_border,
    min_border_bottom=min_border,
                    tools=toolset,
                    toolbar_location=None, x_range=subfig22.x_range, y_range=subfig31.y_range)
  x = 'nr_stakeholders'


  v1 = 'nr_stakeholders'
  v2 = 'cost_sum'
  subfig32.circle( 0,0, line_width=20, fill_color='gray', color='gray', line_alpha=0.25)

  scatter32 = subfig32.scatter(x, y,  source=df, size='marker_size', marker=factor_mark('labels', markers, categories), color=factor_cmap('labels', colours, categories), fill_alpha=fill_alpha, line_width=line_width)
  subfig32.yaxis.major_label_text_font_size = '0pt'
  subfig32.xaxis.axis_label = 'No. of stakeholders (-)'

  subfig32.add_tools(HoverTool(tooltips=[('', '@labels')], renderers=[scatter32]))

  subfig33 = figure(plot_width=subplot_width, plot_height=subplot_height,
    min_border_left=min_border,
    min_border_bottom=min_border,
                    tools=toolset,
                    toolbar_location=None, y_range=subfig31.y_range)
  x = 'FI'

  v1 = 'FI'
  v2 = 'cost_sum'
  pp = pareto_points(df[[v1, v2]])
  subfig33.line(pp[v1], pp[v2], line_width=20, color='gray', line_alpha=0.25)

  scatter33 = subfig33.scatter(x, y,  source=df, size='marker_size', marker=factor_mark('labels', markers, categories), color=factor_cmap('labels', colours, categories), fill_alpha=fill_alpha, line_width=line_width)
  subfig33.yaxis.major_label_text_font_size = '0pt'
  subfig33.x_range = Range1d(pot_ymax, pot_ymin)
  subfig33.xaxis.axis_label = 'PotAll (-)'


  subfig33.add_tools(HoverTool(tooltips=[('', '@labels')], renderers=[scatter33]))

  matrix = gridplot([[subfig11, None, None], [subfig21, subfig22, None], [subfig31, subfig32, subfig33]], toolbar_location='above')


  show(matrix)
