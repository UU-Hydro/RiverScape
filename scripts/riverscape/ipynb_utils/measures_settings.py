import ipywidgets

from collections import OrderedDict


from ipywidgets import HBox, VBox
from ipywidgets import Layout


settings = OrderedDict([
                    ('smoothing_percentage', 100),
                    ('smoothing_ecotope', 'UG-2'),
                    ('smoothing_trachytope', 1201),

                    ('lowering_percentage', 100),
                    ('lowering_ecotope', 'UG-2'),
                    ('lowering_trachytope', 1201),
                    ('lowering_height', 'water_level_50d'),

                    ('channel_width', 75),
                    ('channel_depth', 2.5),
                    ('channel_slope', 1./3.),
                    ('channel_ecotope', 'RnM'),
                    ('channel_trachytope', 105),

                    ('relocation_alpha', 10000),
                    ('relocation_depth', 'AHN'),
                    ('relocation_ecotope', 'HG-2'),
                    ('relocation_trachytope', 1201),

                    ('groyne_ref_level', 'wl_exc150d'),
                    ('minemb_ref_level', 'wl_exc50d'),
                    ('main_dike_dh', 0.50),
                    ])


def on_change(v):
    field = v.owner.description

    settings[field] = v['new']



def measures_settings():


  eco_ids = [1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73]
  eco_codes = [' UM-1', 'RnM', 'RwD', 'REST-H', 'RtM', 'RwM', 'RnD', 'REST-O-U', 'RzM', 'RnO', 'RtO', 'HG-1-2', 'O-UK-1', 'HG-1', 'HG-2', 'VII.1-3', 'RvD', 'UG-1-2', 'O-UP-1', 'IV.8-9', 'O-UG-1', 'UG-2', 'UG-1', 'O-UG-2', 'II.1', 'II.2', 'O-UG-1-2', 'RtD', 'HM-1', 'HA-2', 'HA-1', 'REST-O', 'O-UR-1', 'REST-U', 'UR-1', 'RvO', 'RvM', 'VI.9', 'VI.8', 'V.1-2', 'UP-1', 'O-UB-1', 'HB-5', 'HB-4', 'RwO', 'HB-1', 'VI.4', 'HB-3', 'HB-2', 'III.2-3', 'UB-1', 'VI.2-3', 'VII.4', 'RzO', 'HP-1', 'RzD', 'HR-1', 'I.1', 'IX.a', 'UA-1', 'O-UA-2', 'O-UA-1', 'UA-2', 'VII.3', 'VII.1', 'UB-2', 'UB-3', 'O-UB-2', 'O-UB-3', 'O-UB-4', 'O-UB-5', 'UB-4', 'UB-5']

  trach_ids = [102, 105, 106, 111, 113, 114, 121, 1201, 1202, 1212, 1231, 1233, 1242, 1244, 1245, 1246, 1247, 1250, 1804, 1807]

  smoothing_percentage = ipywidgets.BoundedIntText(
    value=settings['smoothing_percentage'],
    min=0,
    max=100,
    step=1,
    description='smoothing_percentage',
    disabled=False,
    style={'description_width': 'initial'}
  )


  smoothing_ecotope = ipywidgets.Dropdown(
    options=eco_codes,
    value=settings['smoothing_ecotope'],
    description='smoothing_ecotope',
    disabled=False,
    style={'description_width': 'initial'}
  )


  smoothing_trachytope = ipywidgets.Dropdown(
    options=trach_ids,
    value=settings['smoothing_trachytope'],
    description='smoothing_trachytope',
    disabled=False,
    style={'description_width': 'initial'}
)


  lowering_percentage = ipywidgets.BoundedIntText(
    value=settings['lowering_percentage'],
    min=0,
    max=100,
    step=1,
    description='lowering_percentage',
    disabled=False,
    style={'description_width': 'initial'}
    )

  lowering_ecotope = ipywidgets.Dropdown(
    options=eco_codes,
    value=settings['lowering_ecotope'],
    description='lowering_ecotope',
    disabled=False,
    style={'description_width': 'initial'}
    )

  lowering_trachytope = ipywidgets.Dropdown(
    options=trach_ids,
    value=settings['lowering_trachytope'],
    description='lowering_trachytope',
    disabled=False,
    style={'description_width': 'initial'}
  )



  relocation_ecotope= ipywidgets.Dropdown(
    options=eco_codes,
    value=settings['relocation_ecotope'],
    description='relocation_ecotope',
    disabled=False,
    style={'description_width': 'initial'}
    )

  relocation_trachytope = ipywidgets.Dropdown(
    options=trach_ids,
    value=settings['relocation_trachytope'],
    description='relocation_trachytope',
    disabled=False,
    style={'description_width': 'initial'}
  )







  smoothing_percentage.observe(on_change, names='value')
  smoothing_ecotope.observe(on_change, names='value')
  smoothing_trachytope.observe(on_change, names='value')

  lowering_percentage.observe(on_change, names='value')
  lowering_ecotope.observe(on_change, names='value')
  lowering_trachytope.observe(on_change, names='value')

  #relocation_alpha.observe(on_change, names='value')
  relocation_ecotope.observe(on_change, names='value')
  relocation_trachytope.observe(on_change, names='value')


  #display(smoothing_percentage)
  #display(smoothing_ecotope)
  #display(smoothing_trachytope)

  #display(lowering_percentage)
  #display(lowering_ecotope)
  #display(lowering_trachytope)


  ##display(relocation_alpha)
  #display(relocation_ecotope)
  #display(relocation_trachytope)



  #grid = ipywidgets.GridspecLayout(6, 2)
  #grid[0, 0] = smoothing_percentage
  #grid[0, 1] = ipywidgets.HTML(value="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

  #grid[1, 0] = smoothing_ecotope
  #grid[1, 1] = ipywidgets.HTML(value="Curabitur pretium tincidunt lacus. Nulla gravida orci a odio. Nullam varius, turpis et commodo pharetra, est eros bibendum elit, nec luctus magna felis sollicitudin mauris. Integer in mauris eu nibh euismod gravida. Duis ac tellus et risus vulputate vehicula. Donec lobortis risus a elit. Etiam tempor. Ut ullamcorper, ligula eu tempor congue, eros est euismod turpis, id tincidunt sapien risus a quam. Maecenas fermentum consequat mi. Donec fermentum. Pellentesque malesuada nulla a mi. Duis sapien sem, aliquet nec, commodo eget, consequat quis, neque. Aliquam faucibus, elit ut dictum aliquet, felis nisl adipiscing sapien, sed malesuada diam lacus eget erat. Cras mollis scelerisque nunc. Nullam arcu. Aliquam consequat. Curabitur augue lorem, dapibus quis, laoreet et, pretium ac, nisi. Aenean magna nisl, mollis quis, molestie eu, feugiat in, orci. In hac habitasse platea dictumst.")

  #grid[2, 0] = smoothing_trachytope
  #grid[2, 1] = ipywidgets.HTML(value="Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.")

  #grid[3, 0] = lowering_percentage
  #grid[3, 1] = ipywidgets.HTML(value="jeez, where is this whitespace coming from?")

  #grid[4, 0] = relocation_ecotope
  #grid[4, 1] = ipywidgets.HTML(value="jeez, where is this whitespace coming from?")


  #grid[5, 0] = relocation_trachytope
  #grid[5, 1] = ipywidgets.HTML(value="jeez, where is this whitespace coming from?")


  #display(grid)

  #grid

  #display(relocation_ecotope)
  #display(relocation_trachytope)


  text1 = 'This user-specified percentage is required for vegetation roughness measure. The score at a specific percentile of the distribution is used as a threshold for positioning the roughness smoothing. Areas where alpha, the product of specific discharge and Nikuradse equivalent roughness length, exceeded the percentile score are selected for roughness smoothing. The percentile is calculated as 100 minus the user-specified percentage smoothing_percentage of the terrestrial floodplain area.'

  hbox1 = HBox([smoothing_percentage, ipywidgets.HTML(value=text1, layout=Layout(width='60%')) ])

  text2 = 'The (new) ecotope unit that is applied for the smoothing measure'
  hbox2 = HBox([smoothing_ecotope, ipywidgets.HTML(value=text2, layout=Layout(width='60%'))])

  text3 = 'The (new) trachytope unit that is applied for the smoothing measure'
  hbox3 = HBox([smoothing_trachytope, ipywidgets.HTML(value=text3, layout=Layout(width='60%'))])

  text4 = 'This user-specified percentage is required for floodplain lowering measure. Floodplain lowering is positioned where water depth exceeded the score at a certain percentile that equals 100 minus the user-specified lowering_percentage.'
  hbox4 = HBox([lowering_percentage, ipywidgets.HTML(value=text4, layout=Layout(width='60%'))])

  text5 = 'The (new) ecotope unit that is applied for the relocation measure'
  hbox5 = HBox([relocation_ecotope, ipywidgets.HTML(value=text5, layout=Layout(width='60%'))])

  text6 = 'The (new) trachytope unit that is applied for the relocation measure'
  hbox6 = HBox([relocation_trachytope, ipywidgets.HTML(value=text6, layout=Layout(width='60%'))])


  grid = VBox([hbox1, hbox2, hbox3, hbox4, hbox5, hbox6])

  display(grid)

  return settings
