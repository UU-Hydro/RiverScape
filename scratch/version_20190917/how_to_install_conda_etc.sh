
  803  conda create --name rs-env python=3.6
  804  source activate rs-env
  805  conda install geopandas rasterio seaborn spyder xlrd openpyxl jupyter

  808  PCRASTER=/opt/pcraster/pcraster-4.2.1
  809  export PATH=$PCRASTER/bin:$PATH
  810  export PYTHONPATH=$PCRASTER/python:$PYTHONPATH
