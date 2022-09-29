# Spatio-Temporal Data Interpolation Models

This directory contains two major implementations of spatio-temporal interpolation models [Inductive Graph Neural Network for Spatiotemporal Kriging](https://ojs.aaai.org/index.php/AAAI/article/view/16575)(IGNNK) and [Deep Learning based spatio-temporal time series prediction framework](https://www.nature.com/articles/s41598-020-79148-7). 

## Dataset
Air Quality Dataset from Indian city of Pune. Input data is preprocessed to remove outliers and fill all the missing values.


## Inductive Graph Neural Network for Spatiotemporal Kriging


### Dependencies
- Numpy
- Pandas
- Matplotlib
- Torch
- Geopandas
- Scipy
- Seaborn
- Joblib
- Geopy
- Scikit-learn
- [IGNNK](https://github.com/Kaimaoge/IGNNK)

### Files
`GRIN_Imputed_Pune_Data.csv` file: Pune pm2.5 data after performing imputation using GRIN model.\
`IGNNK_Interpolation_pm2p5.ipynb` file: Jupyter notebook contains pytorch implementation of IGNNK interpolation.\
`IGNNK_Model_Training_pm2p5.ipynb` file: Jupyter notebook contains pytorch implementation of IGNNK training.\
`IGNNKKriging.yml` file: Configuration file for IGNNK implementation.

## Deep Learning Framework

### Dependencies
- Numpy
- Pandas
- Geopandas
- Tensorflow
- Keras
- Scikit-learn
- Matplotlib

### Files
`Deep_Learning_Interpolation.ipynb` file: Tensorflow implementation of deep learning framework interpolation.\
`Deep_Learning_Testing.ipynb` file: Tensorflow implementation of deep learning framework training.\
`Deep_Learning_ST.yml` file: Configuration file for deep learning framework implementation.\
`GRIN_Imputed_Pune_Data.csv` file: Pune pm2.5 data after performing imputation using GRIN model



## References
<a id="1">[1]</a> 
Wu, Y., Zhuang, D., Labbe, A., & Sun, L. (2021). Inductive Graph Neural Networks for Spatiotemporal Kriging. Proceedings of the AAAI Conference on Artificial Intelligence, 35(5), 4478-4485. https://doi.org/10.1609/aaai.v35i5.16575.

<a id="1">[2]</a> 
Amato, F., Guignard, F., Robert, S. et al. A novel framework for spatio-temporal prediction of environmental data using deep learning. Sci Rep 10, 22243 (2020). https://doi.org/10.1038/s41598-020-79148-7.
