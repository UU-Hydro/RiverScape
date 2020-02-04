import os

from bokeh.models.widgets import Panel, Tabs
from bokeh.io import output_file, show
from bokeh.plotting import figure


def plot_costs(cost_all_msrs):

  rows = cost_all_msrs.shape[0]
  row_labels = cost_all_msrs.index.values
  column_labels = cost_all_msrs.columns[:].tolist()

  # ...
  column_labels[4] = 'groyne_low'

  fig_tabs = []

  for row_label in row_labels:
    # label for the tab
    label = row_label[0]
    cost = cost_all_msrs.loc[row_label,:].tolist()
    p = figure(x_range=column_labels, plot_width=900, plot_height=400, toolbar_location=None, tools='')

    p.vbar(x=column_labels, width=0.5, top=cost)
    p.xaxis.major_label_orientation = 120.0

    tab = Panel(child=p, title=label)

    fig_tabs.append(tab)

  tabs = Tabs(tabs=fig_tabs)

  return show(tabs)
