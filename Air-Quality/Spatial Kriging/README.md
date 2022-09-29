# Ordinary Kriging

Performs spatial kriging on PM2.5 data. Uses 2-dimentional Ordinary Kriging from pykrige package. Standard variogram models Gaussian, Spherical, and Exponential are used.

### Dataset
Data is derived from Air Quality Dataset from Indian city of Pune. A snapshot (observed data at a particular date and time) of the available data is considered for kriging.

### Dependencies
Numpy\
Pandas\
Matplotlib\
Pykrige\
Scipy\
Datetime\
Gstools\
Folium\
Scikit-gstat

### Files
`Pune_AQI_Interpolation_Gaussian.ipynb` file: Jupyter notebook for spatial interpolation using Ordinary Kriging in PyKrige package.\
`Spatial_Kriging.yml` file: Configuration file for spatial interpolation.\
`avg_aqm_csv.csv` file; Input data set.
