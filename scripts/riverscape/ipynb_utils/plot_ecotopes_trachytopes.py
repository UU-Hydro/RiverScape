import os

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, HoverTool
from bokeh.plotting import figure, show, ColumnDataSource
from bokeh.layouts import gridplot
import colorcet


import pcraster

from osgeo import ogr


import riverscape.ipynb_utils as rsutils









def get_eco(val):
  ecotopes = {1:'UM-1 ', 2:'RnM ', 3:'RwD ', 4:'REST-H ', 5:'RtM ', 6:'RwM ', 7:'RnD ', 8:'REST-O-U ', 9:'RzM ',10:'RnO ',11:'RtO ',12:'HG-1-2 ',13:'O-UK-1 ',14:'HG-1 ',15:'HG-2 ',16:'VII.1-3 ',17:'RvD ',18:'UG-1-2 ',19:'O-UP-1 ',20:'IV.8-9 ',21:'O-UG-1 ',22:'UG-2 ',23:'UG-1 ',24:'O-UG-2 ',25:'II.1 ',26:'II.2 ',27:'O-UG-1-2 ',28:'RtD ',29:'HM-1 ',30:'HA-2 ',31:'HA-1 ',32:'REST-O ',33:'O-UR-1 ',34:'REST-U ',35:'UR-1 ',36:'RvO ',37:'RvM ',38:'VI.9 ',39:'VI.8 ',40:'V.1-2 ',41:'UP-1 ',42:'O-UB-1 ',43:'HB-5 ',44:'HB-4 ',45:'RwO ',46:'HB-1 ',47:'VI.4 ',48:'HB-3 ',49:'HB-2 ',50:'III.2-3 ',51:'UB-1 ',52:'VI.2-3 ',53:'VII.4 ',54:'RzO ',55:'HP-1 ',56:'RzD ',57:'HR-1 ',58:'I.1 ',59:'IX.a ',60:'UA-1 ',61:'O-UA-2 ',62:'O-UA-1 ',63:'UA-2 ',64:'VII.3 ',65:'VII.1 ',66:'UB-2 ',67:'UB-3 ',68:'O-UB-2 ',69:'O-UB-3 ',70:'O-UB-4 ',71:'O-UB-5 ',72:'UB-4 ',73:'UB-5'}
  try:
    return ecotopes[val]

  except Exception as e:
    return 'Report error please'


def plot_eco_trachy_sec():







  fpath = '' #../input_files/input/reference_maps' # why are we in the reference_maps directory already?


  # we need that for the figure dimensions
  ecot = pcraster.readmap(os.path.join(fpath, 'clone.map'))


  ds_source = ogr.GetDriverByName('GPKG').Open(os.path.join(fpath, 'ecotrachysec.gpkg'), update=0)
  eco_layer = ds_source.GetLayerByName('tmp_ecotrachy')
  trach_layer = ds_source.GetLayerByName('tmp_trachytopes')
  section_layer = ds_source.GetLayerByName('tmp_flpl_wide')


  eco_x = []
  eco_y = []
  eco_class = []
  eco_str = []
  for feature in eco_layer:
    geom = feature.GetGeometryRef()
    ring = geom.GetGeometryRef(0)
    points = ring.GetPointCount()
    xn = []
    xy = []
    for pt in range(points):
      rdx, rdy, z = ring.GetPoint(pt)
      xn.append(rdx)
      xy.append(rdy)

    eco_x.append(xn)
    eco_y.append(xy)

    eco_str.append(get_eco(feature.GetField('uid')))
    eco_class.append(feature.GetField('uid'))


  trach_x = []
  trach_y = []
  trach_class = []
  for feature in trach_layer:
    geom = feature.GetGeometryRef()
    ring = geom.GetGeometryRef(0)
    points = ring.GetPointCount()
    xn = []
    xy = []
    for pt in range(points):
      rdx, rdy, z = ring.GetPoint(pt)
      xn.append(rdx)
      xy.append(rdy)

    trach_x.append(xn)
    trach_y.append(xy)
    trach_class.append(feature.GetField('uid'))



  section_x = []
  section_y = []
  section_class = []
  for feature in section_layer:
    geom = feature.GetGeometryRef()
    ring = geom.GetGeometryRef(0)
    points = ring.GetPointCount()
    xn = []
    xy = []
    for pt in range(points):
      rdx, rdy, z = ring.GetPoint(pt)
      xn.append(rdx)
      xy.append(rdy)

    section_x.append(xn)
    section_y.append(xy)
    section_class.append(feature.GetField('uid'))


  color_mapper = LinearColorMapper(colorcet.b_glasbey_bw_minc_20_hue_150_280[:74])


  minx = pcraster.clone().west()
  maxy = pcraster.clone().north()
  maxx = minx + pcraster.clone().nrCols() * pcraster.clone().cellSize()
  miny = maxy - pcraster.clone().nrRows() * pcraster.clone().cellSize()




  eco_source = ColumnDataSource(data=dict(
    x=eco_x, y=eco_y,
    name=eco_class,
    desc=eco_str
  ))


  # Educated guess of the width...
  fw = 900
  aspect = int(pcraster.clone().nrCols() / pcraster.clone().nrRows())
  fh = int(fw / aspect)

  nr_rows = pcraster.clone().nrRows()
  nr_cols = pcraster.clone().nrCols()

  subfig1 = figure( plot_height=fh, plot_width=fw, title='Ecotopes', x_range=(minx, maxx),y_range=(miny, maxy), tools=['hover', 'pan', 'wheel_zoom', 'zoom_in', 'zoom_out', 'reset'], toolbar_location='above')

  subfig1.patches('x', 'y', source=eco_source
         ,fill_color={'field': 'name', 'transform':color_mapper}
         , fill_alpha=1
         , line_width=0
          )

  hover = subfig1.select_one(HoverTool)
  hover.point_policy = 'follow_mouse'
  hover.tooltips = [('Ecotope value', '@name')]






  subfig2 = figure(title='Trachotypes',plot_width=fw, plot_height=fh, x_range=subfig1.x_range, y_range=subfig1.y_range, tools=['hover', 'pan'])


  trach_source = ColumnDataSource(data=dict(
    x=trach_x, y=trach_y,
    name=trach_class
  ))

  color_mapper = LinearColorMapper(colorcet.b_glasbey_bw_minc_20_hue_150_280[:1911])

  subfig2.patches('x', 'y', source=trach_source
         ,fill_color={'field': 'name', 'transform':color_mapper}
         , fill_alpha=1
         , line_width=0
          )


  trach_hover = subfig2.select_one(HoverTool)
  trach_hover.point_policy = 'follow_mouse'
  trach_hover.tooltips = [('Trachytope value', '@name')]



  subfig3 = figure(title='Floodplain sections',plot_width=fw, plot_height=fh, x_range=subfig1.x_range, y_range=subfig1.y_range, tools=['hover', 'pan'])

  section_source = ColumnDataSource(data=dict(
    x=section_x, y=section_y,
    name=section_class
  ))

  color_mapper = LinearColorMapper(colorcet.b_glasbey_bw_minc_20_hue_150_280[:58])

  subfig3.patches('x', 'y', source=section_source
         ,fill_color={'field': 'name', 'transform':color_mapper}#,
         , fill_alpha=1
         , line_width=0
          )


  eco_hover = subfig3.select_one(HoverTool)
  eco_hover.point_policy = 'follow_mouse'
  eco_hover.tooltips = [('Floodplain section', '@name')]


  subfig1.xgrid.grid_line_color = None
  subfig1.ygrid.grid_line_color = None
  subfig2.xgrid.grid_line_color = None
  subfig2.ygrid.grid_line_color = None
  subfig3.xgrid.grid_line_color = None
  subfig3.ygrid.grid_line_color = None



  fig = gridplot([subfig1, subfig2, subfig3], ncols=1, toolbar_location='above')


  return show(fig)







