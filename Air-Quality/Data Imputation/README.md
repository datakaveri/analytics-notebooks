# Data Imputation for Spatio-Temporal Data

This directory contains a jupyter notebook to fill missing values, data imputation, in time series data. The data obtained from [Outliers Detection Directory](https://github.com/datakaveri/analytics-notebooks/tree/main/Air-Quality/Outliers%20Detection) is given as input to the imputation notebook. The notebook uses [Graph Recurrent Imputation Network](https://arxiv.org/pdf/2108.00298.pdf)(GRIN) to fill missing values in the data. The implementation of GRIN model is available in [TORCH SPATIOTEMPORAL](https://torch-spatiotemporal.readthedocs.io/en/latest/index.html). Appropriate changes are made to original model classes to work for available dataset.

### Python Packages
- Numpy
- Pandas
- Datetime
- Pathlib
- Torch
- Pytorch_lightning
- Yaml
- Tsl
