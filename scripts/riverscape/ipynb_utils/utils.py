import os




def root_path():

  p = os.path.join(os.getcwd(), '..')


def input_data_path(directory=None):

  if directory is None:
    p = os.path.join(os.getcwd(), '..', 'input_files', 'input')
  else:
    p = os.path.join(os.getcwd(), '..', 'input_files', 'input', directory)

  return p


def example_data_path():

  p = os.path.join(os.getcwd(), '..', 'outputs', 'example_single_measure', 'maps') # 'sidechannel_evr_natural')

  return p
