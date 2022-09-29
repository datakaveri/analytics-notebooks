# Data Imputation for Spatio-Temporal Data

This directory contains a jupyter notebook to fill missing values, data imputation, in time series data. The data obtained from [Outliers Detection Directory](https://github.com/datakaveri/analytics-notebooks/tree/main/Air-Quality/Outliers%20Detection) is given as input to the imputation notebook. The notebook uses [Graph Recurrent Imputation Network](https://arxiv.org/pdf/2108.00298.pdf)(GRIN) to fill missing values in the data. The implementation of GRIN model is available in [TORCH SPATIOTEMPORAL](https://torch-spatiotemporal.readthedocs.io/en/latest/index.html). Appropriate changes are made to original model classes to work for available dataset.

### Dataset
Air Quality Dataset from Indian city of Pune. Time synchronized input data is preprocessed to remove outliers using model implemented in [Outliers Detection Directory](https://github.com/datakaveri/analytics-notebooks/tree/main/Air-Quality/Outliers%20Detection).

### Dependencies
- Numpy
- Pandas
- Datetime
- Pathlib
- Torch
- Pytorch_lightning
- Yaml
- Tsl

### Files
`Pune_GRIN.ipynb` file: Jupyter notebook contains implementation of GRIN on Pune air quality dataset.\
`pune_aqm_2020Oct_2022July_TimeSynched_NoOutliers_100.csv` file: Time synchrnized input data file obtained after removing outliers.\
`Data_Imputation.yml` file: Configuration file for GRIN implementation.

## References
<a id="1">[1]</a> 
Andrea Cini, Ivan Marisca, and Cesare Alippi, (2021)
Multivariate Time Series Imputation by Graph Neural Networks,
International Conference on Learning Representations.
