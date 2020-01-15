# RiverScape


A set of notebooks for evaluation of river management measures.

> Note: this document and repository is work in progress and not yet intended for usage.



## How to install

A few steps are required to run the Jupyter notebooks:

 1. You will need a working Python environment, we recommend to install Miniconda. Follow their instructions given at:

    [https://docs.conda.io/en/latest/miniconda.html](URL)

 2. Download the requirements file and use it to install all required modules:

    `conda env create -f requirements.yaml`

 3. Activate the environment in a command prompt:

    `conda activate riverscape`

 4. Clone or download this repository:

    `git clone https://github.com/UU-Hydro/RiverScape.git`


## How to run

Activate the environment in a command prompt:

`conda activate riverscape`

Change to the RiverScape `scripts` directory. You can then run a notebooks e.g. with

`jupyter-notebook intervent_parameter.ipynb`


## Exemplary output set of measures

In case you want to run notebooks without explicitly defining your own set of measures first you can load output data from a pre-defined set of measures. Extract the file `example_measures_waal.zip` in the `outputs` folder. You'll get a `example_measures_waal` subfolder containing outputs of 17 measures.



